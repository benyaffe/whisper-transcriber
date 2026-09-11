"""
Tests for the two documents.

Three properties matter, and none of them is "the prose is good", which no test
can check.

House style is enforced rather than requested, because a prompt rule works
nearly always and the one that slips through is published with somebody's name
on it.

Uncertainty markers survive. A writing pass that tidies [?Errol Marchetti] into
Errol Marchetti converts an honest flag into a fabricated fact, and the summary is
where somebody would then act on it.

Meetings come from boundaries.json rather than from file timestamps, which on
the real Ashford trip are all within four seconds of each other because that is
when the files were copied off the device.

Run with: python -m pytest tests/test_output.py -v
"""

import json

import pytest

from src.podcastnotes import output
from src.podcastnotes.context import ContextMap


# --- fakes --------------------------------------------------------------------


class _Text:
    type = "text"

    def __init__(self, text):
        self.text = text


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
    def __init__(self, *texts):
        self.script = [_Message([_Text(t)]) for t in texts]
        self.sent = []
        outer = self

        class _Stream:
            def __init__(self, kw):
                outer.sent.append(kw)

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


def _seg(speaker, text, start=0.0):
    return {"speaker": speaker, "text": text, "start": start, "end": start + 1.0, "words": []}


PAYLOAD = {"segments": [
    _seg("Marcus Ellery", "The first place we went was Ridgeline Northgate.", 0.0),
    _seg("Marcus Ellery", "They saw about eighty thousand people a year.", 1.0),
    _seg("Anya Petrov-Hale", "The medical director was [?Kestler].", 2.0),
    _seg("Devan Shaw", "Internet access was spotty.", 3.0),
]}

BOUNDARIES = {"version": 1, "sources": [
    {"source_name": "Ridgeline Northgate.m4a", "start": 0.0, "end": 2.0},
    {"source_name": "Lakeside StBede.m4a", "start": 2.0, "end": 10.0},
]}


# --- house style, enforced ----------------------------------------------------


@pytest.mark.parametrize("dash", ["—", "--", " – "])
def test_a_dash_that_slipped_past_the_prompt_is_replaced(dash):
    """The prompt asks and the code then checks. Asking works nearly always,
    and nearly is not good enough for a rule somebody cares about."""
    fixed, count = output.enforce(f"the room was fine{dash}mostly")

    assert "—" not in fixed and "--" not in fixed
    assert fixed == "the room was fine, mostly"
    assert count == 1


def test_a_hyphenated_word_is_left_alone():
    """"12-lead" and "rule-out" are not dashes and must survive untouched."""
    fixed, count = output.enforce("a 12-lead ECG and a rule-out pathway")

    assert fixed == "a 12-lead ECG and a rule-out pathway"
    assert count == 0


def test_dashes_are_kept_when_the_style_says_so():
    style = output.HouseStyle(em_dashes=True)

    fixed, count = output.enforce("kept — like this", style)

    assert "—" in fixed and count == 0


def test_the_style_rules_reach_the_prompt():
    client = _FakeClient("transcript", "summary")

    output.write(PAYLOAD, ContextMap(), client=client)

    sent = client.sent[0]["messages"][0]["content"]
    assert "em-dash" in sent
    assert "bullets" in sent
    assert "owner" in sent


def test_the_owner_column_is_asked_against_by_default():
    """The plan's house style: action items as bullets, with no owner column.
    The quality spike produced an owner table, which is what this prevents."""
    assert "owner" in output.HouseStyle().rules()
    assert "owner" not in output.HouseStyle(owner_column=True).rules()


def test_style_fixes_are_counted_across_both_documents():
    client = _FakeClient("a — b", "c — d")

    documents = output.write(PAYLOAD, ContextMap(), client=client)

    assert documents.style_fixes == 2


# --- markers survive ----------------------------------------------------------


def test_a_marker_dropped_by_the_writer_is_reported():
    """Tidying [?Kestler] into Kestler turns an honest flag into a fabricated
    fact, and the summary is where somebody would act on it."""
    client = _FakeClient("The medical director was Kestler.", "summary")

    documents = output.write(PAYLOAD, ContextMap(), client=client)

    assert documents.dropped_markers == ["Kestler"]


def test_a_marker_carried_through_is_not_reported():
    client = _FakeClient("The medical director was [?Kestler].", "summary")

    documents = output.write(PAYLOAD, ContextMap(), client=client)

    assert documents.dropped_markers == []


def test_the_writer_is_told_the_markers_must_survive():
    assert "marker" in output.TRANSCRIPT_SYSTEM
    assert "marker" in output.SUMMARY_SYSTEM


# --- meetings -----------------------------------------------------------------


def test_one_recording_is_one_meeting_with_no_artificial_split():
    assert [name for name, _ in output.meetings(PAYLOAD)] == [""]


def test_several_recordings_split_on_their_boundaries():
    """File timestamps cannot do this: on the real Ashford trip all three
    report creation within four seconds of each other, because that is when
    they were copied off the device."""
    split = output.meetings(PAYLOAD, BOUNDARIES)

    assert [name for name, _ in split] == ["Ridgeline Northgate.m4a", "Lakeside StBede.m4a"]
    assert [len(rows) for _, rows in split] == [2, 2]


def test_a_single_source_is_still_one_meeting():
    one = {"sources": [{"source_name": "only.m4a", "start": 0.0, "end": 10.0}]}

    assert [name for name, _ in output.meetings(PAYLOAD, one)] == [""]


def test_every_segment_lands_in_exactly_one_meeting():
    """A segment appearing twice duplicates content in the document, and one
    appearing nowhere loses it silently."""
    split = output.meetings(PAYLOAD, BOUNDARIES)
    seen = [s["text"] for _, rows in split for s in rows]

    assert sorted(seen) == sorted(s["text"] for s in PAYLOAD["segments"])


def test_a_final_recording_with_no_end_still_collects_its_segments():
    open_ended = {"sources": [
        {"source_name": "first.m4a", "start": 0.0, "end": 2.0},
        {"source_name": "second.m4a", "start": 2.0},
    ]}

    split = output.meetings(PAYLOAD, open_ended)

    assert [len(rows) for _, rows in split] == [2, 2]


# --- the script Claude is given -----------------------------------------------


def test_the_script_uses_the_confirmed_speaker_names():
    script = output.as_script(PAYLOAD)

    assert "Marcus Ellery:" in script
    assert "Speaker 1" not in script


def test_consecutive_turns_from_one_speaker_are_not_repeated():
    script = output.as_script(PAYLOAD)

    assert script.count("Marcus Ellery:") == 1


def test_the_recording_names_appear_when_there_are_several():
    script = output.as_script(PAYLOAD, BOUNDARIES)

    assert "Ridgeline Northgate.m4a" in script
    assert "Lakeside StBede.m4a" in script


def test_open_questions_are_passed_on_so_they_are_not_stated_as_settled():
    client = _FakeClient("transcript", "summary")
    context = ContextMap(open_questions=["Who is Kestler?"])

    output.write(PAYLOAD, context, client=client)

    assert "Who is Kestler?" in client.sent[0]["messages"][0]["content"]


def test_both_documents_get_the_most_effort():
    client = _FakeClient("transcript", "summary")

    output.write(PAYLOAD, ContextMap(), client=client)

    assert [c["output_config"]["effort"] for c in client.sent] == ["max", "max"]


# --- what comes out -----------------------------------------------------------


def test_both_documents_are_returned():
    client = _FakeClient("the transcript", "the summary")

    documents = output.write(PAYLOAD, ContextMap(), client=client)

    assert documents.transcript == "the transcript"
    assert documents.summary == "the summary"


def test_the_combined_document_leads_with_the_summary():
    """It is published as one Doc, and the person opening it wants the
    decisions before the raw record."""
    documents = output.Documents(transcript="TRANSCRIPT", summary="SUMMARY")

    assert documents.combined.index("SUMMARY") < documents.combined.index("TRANSCRIPT")


def test_write_works_without_a_context_map():
    client = _FakeClient("t", "s")

    assert output.write(PAYLOAD, None, client=client).transcript == "t"
