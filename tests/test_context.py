"""
Tests for the context map stage.

The rule this stage exists to enforce is that a Glean outage stops the write-up
rather than degrading it. Everything else here protects the research loop from
throwing itself away: a stray code fence, a missing key, or one bad query
should not discard work that otherwise succeeded.

No test needs a Glean credential or a Vertex project.

Run with: python -m pytest tests/test_context.py -v
"""

import json

import pytest

from src.podcastnotes import context, glean
from src.podcastnotes.llm import agent


# --- fakes --------------------------------------------------------------------


class _Text:
    type = "text"

    def __init__(self, text):
        self.text = text


class _ToolUse:
    type = "tool_use"

    def __init__(self, tool_input, id="t1", name="search_company_knowledge"):
        self.name = name
        self.input = tool_input
        self.id = id


class _Usage:
    input_tokens = output_tokens = 0
    cache_creation_input_tokens = cache_read_input_tokens = 0


class _Message:
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = _Usage()
        self.stop_details = None


class _FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.sent = []
        outer = self

        class _Stream:
            def __init__(self, kwargs):
                outer.sent.append(kwargs)

            def __enter__(s):
                return s

            def __exit__(s, *e):
                return False

            def get_final_message(s):
                return outer.script.pop(0)

        class _Messages:
            def stream(self, **kw):
                return _Stream(kw)

        self.messages = _Messages()


GOOD_MAP = {
    "people": [{"name": "Dr. Miles Nadeau", "role": "consultant",
                "why_relevant": "guided the trip", "confidence": "high"}],
    "organisations": [{"name": "Lakeside General", "what_it_is": "hospital system"}],
    "terms": [{"term": "PMG", "expansion": "photomagnetography", "note": ""}],
    "projects": [{"name": "Lanternfish", "what_it_is": "derisking", "dates": "2026"}],
    "likely_errors": [{"heard": "Miles Nadoe", "probably": "Miles Nadeau",
                       "evidence": "working doc", "confidence": "high"}],
    "hard_dates": [{"date": "2026-09-08", "what": "the visits"}],
    "open_questions": ["Who is Kestler?"],
}


def _answers(payload):
    """A model that answers immediately with the given JSON."""
    return _FakeClient([_Message([_Text(json.dumps(payload))])])


def _ok_search(query):
    return [{"title": "a doc", "body": "some evidence", "url": "u",
             "source": "gdrive", "created": "", "updated": "", "owner": ""}]


# --- the outage rule ----------------------------------------------------------


def test_an_unreachable_glean_stops_the_stage(monkeypatch):
    """The rule this module exists for. Proceeding produces a finished-looking
    document that is wrong about every name in it."""
    def dead(query):
        raise ConnectionError("no route to host")

    with pytest.raises(context.GleanUnavailable):
        context.build("a trip", "a transcript", search=dead, client=_answers(GOOD_MAP))


def test_a_missing_credential_stops_the_stage():
    def unconfigured(query):
        raise glean.NotConfigured("No Glean credential.")

    with pytest.raises(context.GleanUnavailable, match="credential"):
        context.build("a trip", "a transcript", search=unconfigured, client=_answers(GOOD_MAP))


def test_the_outage_is_found_before_claude_is_paid_for(monkeypatch):
    """Found on the tenth tool call, an outage has already produced half a map
    that looks like a result."""
    client = _answers(GOOD_MAP)

    with pytest.raises(context.GleanUnavailable):
        context.build("t", "t", search=lambda q: (_ for _ in ()).throw(OSError("down")), client=client)

    assert client.sent == [], "Claude was called before Glean was known to work"


def test_a_credential_that_dies_mid_run_stops_rather_than_carrying_on():
    """Distinct from one bad query: every remaining search will fail too."""
    calls = []

    def dies_after_preflight(query):
        calls.append(query)
        if len(calls) == 1:
            return _ok_search(query)
        raise glean.NotConfigured("token revoked")

    client = _FakeClient([
        _Message([_ToolUse({"query": "nadeau"})], stop_reason="tool_use"),
        _Message([_Text("never reached")]),
    ])

    with pytest.raises(context.GleanUnavailable):
        context.build("t", "t", search=dies_after_preflight, client=client)


def test_one_empty_search_is_not_an_outage():
    """No results for a query is ordinary and the loop should carry on."""
    def sometimes_empty(query):
        return [] if "nothing" in query else _ok_search(query)

    client = _FakeClient([
        _Message([_ToolUse({"query": "nothing at all"})], stop_reason="tool_use"),
        _Message([_Text(json.dumps(GOOD_MAP))]),
    ])

    found = context.build("t", "t", search=sometimes_empty, client=client)

    assert found.people[0]["name"] == "Dr. Miles Nadeau"


# --- what Claude is given -----------------------------------------------------


def test_the_transcript_is_handed_over_so_garbles_can_be_searched():
    """Without it the stage can only search the trip description, which never
    contains the mis-heard word that actually needs resolving."""
    client = _answers(GOOD_MAP)

    context.build("Ashford ED visits", "we saw Dr. Ives Vahalee", search=_ok_search, client=client)

    sent = client.sent[0]["messages"][0]["content"]
    assert "Ives Vahalee" in sent
    assert "Ashford ED visits" in sent


def test_the_search_budget_is_passed_through():
    client = _answers(GOOD_MAP)

    context.build("t", "t", max_searches=3, search=_ok_search, client=client)

    # The tool is declared; the budget lives in the loop, so assert the wiring.
    assert client.sent[0]["tools"][0]["name"] == "search_company_knowledge"


def test_queries_are_recorded_for_the_preview():
    """The user has to be able to see what it looked at before trusting it."""
    client = _FakeClient([
        _Message([_ToolUse({"query": "nadeau"}, id="a"), _ToolUse({"query": "lanternfish"}, id="b")],
                 stop_reason="tool_use"),
        _Message([_Text(json.dumps(GOOD_MAP))]),
    ])

    found = context.build("t", "t", search=_ok_search, client=client)

    assert found.searches == ["nadeau", "lanternfish"]


def test_a_blank_query_is_refused_without_calling_glean():
    seen = []

    def counting(query):
        seen.append(query)
        return _ok_search(query)

    client = _FakeClient([
        _Message([_ToolUse({"query": "   "})], stop_reason="tool_use"),
        _Message([_Text(json.dumps(GOOD_MAP))]),
    ])

    context.build("t", "t", search=counting, client=client)

    assert seen == ["podcastnotes context preflight"], "a blank query reached Glean"


# --- reading the answer -------------------------------------------------------


def test_a_code_fence_does_not_discard_the_research():
    client = _FakeClient([
        _Message([_Text("```json\n" + json.dumps(GOOD_MAP) + "\n```")])
    ])

    assert context.build("t", "t", search=_ok_search, client=client).people


def test_preamble_before_the_json_is_tolerated():
    client = _FakeClient([
        _Message([_Text("Here is what I found.\n\n" + json.dumps(GOOD_MAP))])
    ])

    assert context.build("t", "t", search=_ok_search, client=client).hard_dates


def test_a_missing_key_is_an_empty_list_not_a_crash():
    client = _answers({"people": [{"name": "Anya"}]})

    found = context.build("t", "t", search=_ok_search, client=client)

    assert found.people == [{"name": "Anya"}]
    assert found.open_questions == []


def test_unparseable_output_yields_an_empty_map_rather_than_raising():
    client = _FakeClient([_Message([_Text("I could not do this.")])])

    found = context.build("t", "t", search=_ok_search, client=client)

    assert found.is_empty


def test_json_truncated_mid_object_yields_an_empty_map(tmp_path):
    """The realistic failure, and the one the earlier test missed: a response
    cut off at max_tokens has an opening brace and no closing one, so it
    reaches the parser rather than being rejected for having no JSON at all."""
    # A closing brace has to be present, or the object is rejected for having
    # no JSON in it at all and the parser is never reached. That was the flaw
    # in the first version of this test.
    truncated = '{"people": [{"name": "Anya Petrov-Hale", "role": "GM"}'
    client = _FakeClient([_Message([_Text(truncated)])])

    found = context.build("t", "t", search=_ok_search, client=client)

    assert found.is_empty


def test_an_empty_map_says_so():
    assert context.ContextMap().is_empty
    assert not context.ContextMap(people=[{"name": "x"}]).is_empty


# --- the file the user edits --------------------------------------------------


def test_the_map_round_trips_through_disk(tmp_path):
    original = context.ContextMap.from_dict(GOOD_MAP)

    written = original.save(tmp_path)
    reloaded = context.ContextMap.load(tmp_path)

    assert written.endswith(context.FILENAME)
    assert reloaded.people == original.people
    assert reloaded.likely_errors == original.likely_errors


def test_saving_to_an_explicit_filename_is_respected(tmp_path):
    target = tmp_path / "somewhere_else.json"

    context.ContextMap().save(target)

    assert target.exists()


def test_the_file_is_readable_by_a_person(tmp_path):
    """The user is expected to open and edit this, so it is indented."""
    context.ContextMap.from_dict(GOOD_MAP).save(tmp_path)

    assert "\n  " in (tmp_path / context.FILENAME).read_text()


def test_a_hand_edited_file_with_a_wrong_type_does_not_crash_the_load(tmp_path):
    """People edit this by hand, and a JSON object where a list belongs is the
    likeliest way to get it wrong."""
    (tmp_path / context.FILENAME).write_text(json.dumps({"people": {"oops": 1}, "terms": []}))

    reloaded = context.ContextMap.load(tmp_path)

    assert reloaded.people == []


# --- folding a second pass into the first ----------------------------------------


def test_a_second_pass_does_not_lose_the_first():
    base = _map_with(people=[{"name": "Anya"}], terms=[{"term": "PMG"}])
    found = _map_with(people=[{"name": "Dr Okafor"}])

    merged = context.merge(base, found)

    assert [p["name"] for p in merged.people] == ["Anya", "Dr Okafor"]
    assert [t["term"] for t in merged.terms] == ["PMG"]


def test_the_same_person_found_twice_is_listed_once():
    base = _map_with(people=[{"name": "Anya Petrov-Hale"}])
    found = _map_with(people=[{"name": "  anya petrov-hale  "}])

    merged = context.merge(base, found)

    assert len(merged.people) == 1


def test_a_narrower_pass_fills_gaps_and_does_not_overwrite():
    """The second pass was seeded by one sentence, so it knows more about one
    thing and less about the trip. A thinner answer is less knowledge, not
    corrected knowledge."""
    base = _map_with(people=[{"name": "Anya", "role": "GM", "why_relevant": ""}])
    found = _map_with(people=[{"name": "Anya", "role": "", "why_relevant": "ran the trip"}])

    merged = context.merge(base, found)

    assert merged.people[0]["role"] == "GM"
    assert merged.people[0]["why_relevant"] == "ran the trip"


def test_a_correction_for_the_same_words_replaces_rather_than_appends():
    """The one that matters. `correct._proposals` sorts longest-heard first with
    a stable sort, so an appended row for words an existing row already claims
    never fires: the old row matches and the new one is reported as "not found
    in the transcript"."""
    from src.podcastnotes import correct

    base = _map_with(likely_errors=[
        {"heard": "Simon", "probably": "Simone", "confidence": "medium"},
    ])
    found = _map_with(likely_errors=[
        {"heard": "simon", "probably": "Simone Vasari", "confidence": "high"},
    ])

    merged = context.merge(base, found)
    payload = {"segments": [{"text": "we met Simon there", "words": []}]}
    result = correct.apply(payload, merged)

    assert len(merged.likely_errors) == 1
    assert result.payload["segments"][0]["text"] == "we met Simone Vasari there"
    assert result.unmatched == []


def test_a_question_the_person_answered_stops_being_open():
    """`output._shared` feeds these to the writer under "do not present as
    settled", so an answered one left in makes the rewrite hedge about the
    exact thing they just cleared up."""
    base = _map_with(open_questions=["Who is Kestler?", "What is 60-hold?"])

    merged = context.merge(base, _map_with(), answered=["who is kestler?"])

    assert merged.open_questions == ["What is 60-hold?"]


def test_a_newly_raised_question_is_kept():
    merged = context.merge(
        _map_with(open_questions=["Old one"]),
        _map_with(open_questions=["New one", "Old one"]),
    )

    assert merged.open_questions == ["Old one", "New one"]


def test_the_searches_from_both_passes_are_counted():
    merged = context.merge(_map_with(searches=["a", "b"]), _map_with(searches=["c"]))

    assert merged.searches == ["a", "b", "c"]


def test_a_repeated_date_is_not_listed_twice():
    row = {"date": "2026-09-08", "what": "the visits"}

    merged = context.merge(_map_with(hard_dates=[row]), _map_with(hard_dates=[dict(row)]))

    assert len(merged.hard_dates) == 1


def test_merging_nothing_changes_nothing():
    base = _map_with(people=[{"name": "Anya"}], likely_errors=[{"heard": "a", "probably": "b"}])

    merged = context.merge(base, _map_with())

    assert merged.people == base.people
    assert merged.likely_errors == base.likely_errors


def test_the_originals_are_left_alone():
    """A caller showing what a revision changed needs both versions."""
    base = _map_with(people=[{"name": "Anya", "role": "GM"}])

    context.merge(base, _map_with(people=[{"name": "Anya", "role": "", "extra": "x"}]))

    assert base.people == [{"name": "Anya", "role": "GM"}]


def _map_with(**kwargs):
    return context.ContextMap(**kwargs)
