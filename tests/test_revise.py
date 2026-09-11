"""
Tests for the pass that happens after somebody has read the draft.

The rule is that the person who was there is believed, and the mechanism for
that is confidence rather than a special case in the correction stage: what
they state plainly comes back as high, the merge replaces the hedged row, and
`correct.apply` then substitutes it silently with its one rule untouched.

The other property worth defending is that a request the recording cannot
support says so. Doing nothing is indistinguishable from being ignored, and
somebody who thinks they were ignored retypes it and pays for another pass.

Run with: python -m pytest tests/test_revise.py -v
"""

import json

import pytest

from src.podcastnotes import context, revise
from src.podcastnotes.context import ContextMap


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


def _answers(payload):
    return _FakeClient([_Message([_Text(json.dumps(payload))])])


def _ok_search(query):
    return [{"title": "a doc", "body": "evidence", "url": "u",
             "source": "gdrive", "created": "", "updated": "", "owner": ""}]


EXISTING = ContextMap(
    likely_errors=[{"heard": "Simon", "probably": "Simone", "confidence": "medium"}],
    people=[{"name": "Anya Petrov-Hale", "role": "GM"}],
    open_questions=["Who is the St. Bede coordinator?"],
)

TRANSCRIPT = "Speaker 2:\nAnd Simon, I didn't get her last name."


# --- believing the person who was there ------------------------------------------


def test_what_they_state_plainly_comes_back_confident():
    """The mechanism for dropping the [?...] markers. Not a special case in
    correct.py: the row arrives at high confidence, the merge replaces the
    hedged one, and the correction stage keeps its single rule."""
    client = _answers({"likely_errors": [
        {"heard": "Simon", "probably": "Simone Vasari", "confidence": "high",
         "evidence": "you told me after reading the draft"},
    ]})

    found = revise.from_notes(
        EXISTING, "The coordinator is Simone Vasari.", TRANSCRIPT,
        client=client, search=_ok_search,
    )

    assert found.context_map.likely_errors[0]["confidence"] == "high"


def test_the_prompt_says_to_believe_them():
    assert "Believe them" in revise.SYSTEM
    assert "were physically present and you were not" in revise.SYSTEM


def test_a_supplied_name_is_searched_for():
    """A name is a key, not just a spelling. This is the whole reason the pass
    is a fresh Glean loop rather than a substitution."""
    searched = []

    def watching(query):
        searched.append(query)
        return _ok_search(query)

    client = _FakeClient([
        _Message([_ToolUse({"query": "Simone Vasari St Bede"})], stop_reason="tool_use"),
        _Message([_Text(json.dumps({"likely_errors": []}))]),
    ])

    revise.from_notes(EXISTING, "The coordinator is Simone Vasari.", TRANSCRIPT,
                      client=client, search=watching)

    assert any("Simone Vasari" in q for q in searched)


def test_the_raw_transcript_is_what_it_corrects_against():
    """Corrections are re-applied from the original every time, so a row
    written against the polished document matches nothing."""
    client = _answers({"likely_errors": []})

    revise.from_notes(EXISTING, "note", TRANSCRIPT, client=client, search=_ok_search)

    sent = client.sent[0]["messages"][0]["content"]
    assert "And Simon, I didn't get her last name." in sent
    assert "before any correction was applied" in sent


def test_what_is_already_known_is_handed_over():
    """So the pass extends the map rather than deriving it again."""
    client = _answers({"likely_errors": []})

    revise.from_notes(EXISTING, "note", TRANSCRIPT, client=client, search=_ok_search)

    sent = client.sent[0]["messages"][0]["content"]
    assert "Anya Petrov-Hale" in sent
    assert "Who is the St. Bede coordinator?" in sent


def test_the_budget_is_smaller_than_a_cold_pass():
    """A seeded pass starts from a name and a map that exists, so it branches
    far less than one starting from a description."""
    assert revise.REVISE_SEARCHES < context.MAX_SEARCHES


# --- naming a voice rather than fixing a word ------------------------------------


def test_identifying_a_speaker_comes_back_as_a_rename():
    """"The coordinator is Simone Vasari" much more often means a label is her
    than that those syllables are misspelt."""
    client = _answers({"speakers": {"Speaker 2": "Simone Vasari"}})

    found = revise.from_notes(
        EXISTING, "Speaker 2 is Simone Vasari.", TRANSCRIPT,
        speakers=["Speaker 1", "Speaker 2"], client=client, search=_ok_search,
    )

    assert found.speakers == {"Speaker 2": "Simone Vasari"}


def test_a_rename_for_a_speaker_that_does_not_exist_is_dropped():
    """It could never be applied, so carrying it forward is a silent no-op."""
    client = _answers({"speakers": {"Speaker 9": "Nobody"}})

    found = revise.from_notes(EXISTING, "note", TRANSCRIPT,
                              speakers=["Speaker 1"], client=client, search=_ok_search)

    assert found.speakers == {}


def test_a_blank_name_is_not_a_rename():
    client = _answers({"speakers": {"Speaker 1": "   "}})

    found = revise.from_notes(EXISTING, "note", TRANSCRIPT,
                              speakers=["Speaker 1"], client=client, search=_ok_search)

    assert found.speakers == {}


# --- saying what cannot be done --------------------------------------------------


def test_something_the_recording_cannot_support_is_reported():
    """Not inventing it is half the job. Saying so is the other half: doing
    nothing looks exactly like being ignored."""
    client = _answers({
        "impossible": ["Nobody in the recording mentions a Dr Okafor."],
        "likely_errors": [],
    })

    found = revise.from_notes(EXISTING, "Add Dr Okafor from cardiology.", TRANSCRIPT,
                              client=client, search=_ok_search)

    assert found.impossible == ["Nobody in the recording mentions a Dr Okafor."]
    assert found.context_map.likely_errors == []


def test_the_prompt_forbids_inventing_one_instead():
    assert "impossible" in revise.SYSTEM
    assert "fabricated claim" in revise.SYSTEM


# --- questions they answered -----------------------------------------------------


def test_the_questions_their_note_settles_come_back():
    client = _answers({"settled": ["Who is the St. Bede coordinator?"]})

    found = revise.from_notes(EXISTING, "It is Simone Vasari.", TRANSCRIPT,
                              client=client, search=_ok_search)

    assert found.settled == ["Who is the St. Bede coordinator?"]


# --- failures --------------------------------------------------------------------


def test_glean_being_down_stops_the_rewrite():
    """The documents already on screen are fine. Substituting somebody's words
    blind because the research could not run is not an improvement on them."""
    def dead(query):
        raise ConnectionError("no route to host")

    with pytest.raises(context.GleanUnavailable):
        revise.from_notes(EXISTING, "note", TRANSCRIPT,
                          client=_answers({}), search=dead)


def test_unreadable_output_changes_nothing():
    client = _FakeClient([_Message([_Text("I could not work this out.")])])

    found = revise.from_notes(EXISTING, "note", TRANSCRIPT,
                              client=client, search=_ok_search)

    assert found.is_empty


def test_an_empty_revision_says_so():
    assert revise.Revision().is_empty is True
    assert revise.Revision(impossible=["something"]).is_empty is False


# --- the shared rules ------------------------------------------------------------


def test_both_prompts_describe_substitution_the_same_way():
    """The rules are what `correct._proposals` actually consumes. A paraphrase
    in the second prompt drifts out of step with it without anything failing."""
    filled = revise.SYSTEM.replace("{substitution_rules}", context.SUBSTITUTION_RULES)

    assert context.SUBSTITUTION_RULES in context.SYSTEM
    assert context.SUBSTITUTION_RULES in filled
