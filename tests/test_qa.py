"""
Tests for the question round.

The budget here is somebody's attention, not tokens. Four questions is a
screenful a person answers; nineteen is a form they abandon, and an abandoned
form leaves every marker in place, which is a worse outcome than never having
asked. So the cap is a correctness property and is tested as one.

The other property worth defending is that an answer settling nothing must
produce no correction. Manufacturing one from "I'm not sure" swaps an honest
marker for a confident error, which nothing downstream can catch.

Run with: python -m pytest tests/test_qa.py -v
"""

import json

import pytest

from src.podcastnotes import qa
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
    def __init__(self, script):
        self.script = list(script)
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


def _answers(payload):
    return _FakeClient([_Message([_Text(json.dumps(payload))])])


def _seg(text, speaker="Anya"):
    return {"speaker": speaker, "text": text, "words": []}


PAYLOAD = {"segments": [
    _seg("75 beds, [?60 hold], I do not know what that means."),
    _seg("the medical director was [?Kestler]", speaker="Marcus"),
    _seg("we spoke to [?Kestler] again later", speaker="Marcus"),
    _seg("down the [?z-lanes] in the hallway", speaker="Devan"),
]}


def _question(marker, ask="What did they mean?"):
    return {"marker": marker, "ask": ask, "why_it_matters": ""}


# --- finding what is still unresolved -----------------------------------------


def test_every_marker_is_found_and_counted():
    """Frequency is most of what decides whether a question earns a slot."""
    assert qa.outstanding(PAYLOAD) == {"60 hold": 1, "Kestler": 2, "z-lanes": 1}


def test_a_transcript_with_no_markers_has_nothing_outstanding():
    assert qa.outstanding({"segments": [_seg("all perfectly clear")]}) == {}


def test_the_excerpt_quotes_the_line_and_who_said_it():
    """So the question can put them back in the moment, which is the
    difference between an answerable question and an unanswerable one."""
    lines = qa.excerpts(PAYLOAD, "60 hold")

    assert lines == ["Anya: 75 beds, [?60 hold], I do not know what that means."]


def test_only_so_many_excerpts_are_quoted():
    assert len(qa.excerpts(PAYLOAD, "Kestler", limit=1)) == 1


# --- the attention budget -----------------------------------------------------


def test_never_more_than_four_questions_in_a_round():
    """A screen of nineteen questions is how the twenty-minute promise breaks,
    and an abandoned form leaves every marker in place."""
    many = {"segments": [_seg(f"a [?marker{i}] here") for i in range(20)]}
    client = _answers({"questions": [_question(f"marker{i}") for i in range(20)]})

    result = qa.next_round(many, ContextMap(), client=client)

    assert len(result.questions) == qa.PER_ROUND


def test_what_was_not_asked_is_reported_as_remaining():
    many = {"segments": [_seg(f"a [?marker{i}] here") for i in range(10)]}
    client = _answers({"questions": [_question(f"marker{i}") for i in range(10)]})

    result = qa.next_round(many, ContextMap(), client=client)

    assert result.remaining == 6


def test_there_is_no_fourth_round():
    client = _answers({"questions": [_question("Kestler")]})

    result = qa.next_round(PAYLOAD, ContextMap(), number=4, client=client)

    assert result.questions == []
    assert client.sent == [], "Claude was called for a round that cannot be asked"


def test_the_third_round_knows_it_is_the_last():
    client = _answers({"questions": [_question("Kestler")]})

    assert qa.next_round(PAYLOAD, ContextMap(), number=3, client=client).is_last is True
    assert qa.Round(number=1).is_last is False


def test_a_marker_already_asked_about_is_not_asked_again():
    """Asking twice because the first answer did not clear the marker is how a
    tool loses somebody's trust."""
    client = _answers({"questions": [_question("Kestler"), _question("z-lanes")]})

    result = qa.next_round(PAYLOAD, ContextMap(), already_asked=["Kestler"], client=client)

    assert [q.marker for q in result.questions] == ["z-lanes"]


def test_nothing_is_asked_when_every_marker_has_been_covered():
    client = _answers({"questions": [_question("Kestler")]})

    result = qa.next_round(
        PAYLOAD, ContextMap(),
        already_asked=["60 hold", "Kestler", "z-lanes"],
        client=client,
    )

    assert result.questions == []
    assert client.sent == [], "Claude was asked to invent questions about nothing"


# --- what comes back ----------------------------------------------------------


def test_a_question_carries_how_often_its_marker_appears():
    client = _answers({"questions": [_question("Kestler")]})

    assert qa.next_round(PAYLOAD, ContextMap(), client=client).questions[0].occurrences == 2


def test_a_marker_echoed_with_its_brackets_still_matches():
    """The prompt shows markers as they appear in the transcript, "[?Simone]",
    so that is the form Claude sends back. Matching that against the bare inner
    text discarded every question, and the round came back empty looking as
    though there had been nothing to ask."""
    client = _answers({"questions": [_question("[?Kestler]")]})

    result = qa.next_round(PAYLOAD, ContextMap(), client=client)

    assert [q.marker for q in result.questions] == ["Kestler"]


def test_a_field_naming_two_markers_is_dropped_rather_than_guessed_at():
    """Only a field that is entirely one marker counts. Picking the first of
    two would attach the question to whichever happened to come first, and a
    question pointed at the wrong marker is worse than one not asked."""
    client = _answers({"questions": [_question("[?Kestler] and [?z-lanes]")]})

    assert qa.next_round(PAYLOAD, ContextMap(), client=client).questions == []


def test_a_marker_echoed_bare_still_matches():
    client = _answers({"questions": [_question("Kestler")]})

    assert qa.next_round(PAYLOAD, ContextMap(), client=client).questions[0].marker == "Kestler"


def test_a_question_about_a_marker_that_does_not_exist_is_dropped():
    client = _answers({"questions": [_question("invented"), _question("Kestler")]})

    result = qa.next_round(PAYLOAD, ContextMap(), client=client)

    assert [q.marker for q in result.questions] == ["Kestler"]


def test_a_question_with_no_text_is_dropped():
    client = _answers({"questions": [{"marker": "Kestler", "ask": "  "}]})

    assert qa.next_round(PAYLOAD, ContextMap(), client=client).questions == []


def test_unparseable_output_asks_nothing_rather_than_crashing():
    client = _FakeClient([_Message([_Text("I have no questions.")])])

    assert qa.next_round(PAYLOAD, ContextMap(), client=client).questions == []


def test_the_established_context_is_sent_so_it_is_not_asked_about_again():
    client = _answers({"questions": []})
    context = ContextMap(people=[{"name": "Dr. Miles Nadeau"}])

    qa.next_round(PAYLOAD, context, client=client)

    assert "Nadeau" in client.sent[0]["messages"][0]["content"]


# --- folding the answers back -------------------------------------------------


def test_an_answer_becomes_a_correction():
    client = _answers({"corrections": [
        {"heard": "60 hold", "probably": "60 hold beds", "evidence": "Anya said so",
         "confidence": "high"},
    ], "still_open": []})

    updated = qa.absorb(ContextMap(), {"60 hold": "It means sixty holding beds."}, client=client)

    assert updated.likely_errors[0]["probably"] == "60 hold beds"
    assert updated.likely_errors[0]["confidence"] == "high"


def test_an_answer_that_settles_nothing_produces_no_correction():
    """Manufacturing one from "I'm not sure" swaps an honest marker for a
    confident error, and nothing downstream can catch that."""
    client = _answers({"corrections": [], "still_open": ["Nobody could recall the name."]})

    updated = qa.absorb(ContextMap(), {"Kestler": "No idea, sorry."}, client=client)

    assert updated.likely_errors == []
    assert updated.open_questions == ["Nobody could recall the name."]


def test_a_correction_missing_either_side_is_ignored():
    client = _answers({"corrections": [
        {"heard": "Kestler", "probably": ""},
        {"heard": "", "probably": "Kessler"},
    ]})

    assert qa.absorb(ContextMap(), {"Kestler": "It is Kessler."}, client=client).likely_errors == []


def test_an_answered_guess_is_superseded_rather_than_added_to():
    """The marker in the transcript is rendered from the original guess, so
    leaving that guess in place means the next correction pass writes the same
    marker straight back and the round achieves nothing a person can see."""
    existing = ContextMap(likely_errors=[
        {"heard": "Simon", "probably": "Simone", "confidence": "medium"},
        {"heard": "Ridgelane", "probably": "Ridgeline", "confidence": "high"},
    ])
    client = _answers({"settled": ["Simone"], "corrections": [
        {"heard": "Simon", "probably": "Simone Vasari", "confidence": "high"},
    ]})

    updated = qa.absorb(existing, {"Simone": "It is Simone Vasari."}, client=client)

    assert [e["probably"] for e in updated.likely_errors] == ["Ridgeline", "Simone Vasari"]


def test_a_superseded_guess_is_matched_through_its_alternatives():
    existing = ContextMap(likely_errors=[
        {"heard": "recess room", "probably": "resus room / resuscitation area",
         "confidence": "medium"},
    ])
    client = _answers({"settled": ["resus room"], "corrections": [
        {"heard": "recess room", "probably": "resus room", "confidence": "high"},
    ]})

    updated = qa.absorb(existing, {"resus room": "Resus room."}, client=client)

    assert [e["probably"] for e in updated.likely_errors] == ["resus room"]


def test_an_answer_that_only_confirms_adds_no_substitution():
    """Dropping the uncertain entry is what clears the marker. An identity
    correction would just be noise in the change log a person reads."""
    existing = ContextMap(likely_errors=[
        {"heard": "Whitlock", "probably": "Curtis Whitlock", "confidence": "medium"},
    ])
    client = _answers({"settled": ["Curtis Whitlock"], "corrections": [
        {"heard": "Curtis Whitlock", "probably": "Curtis Whitlock", "confidence": "high"},
    ]})

    updated = qa.absorb(existing, {"Curtis Whitlock": "Yes, that is right."}, client=client)

    assert updated.likely_errors == []


def test_existing_corrections_are_kept():
    existing = ContextMap(likely_errors=[{"heard": "Ridgelane", "probably": "Ridgeline"}])
    client = _answers({"corrections": [{"heard": "Kestler", "probably": "Kessler"}]})

    updated = qa.absorb(existing, {"Kestler": "Kessler."}, client=client)

    assert [e["heard"] for e in updated.likely_errors] == ["Ridgelane", "Kestler"]


def test_the_original_context_map_is_not_modified():
    """A caller showing what an answer changed needs both versions."""
    original = ContextMap(likely_errors=[])
    client = _answers({"corrections": [{"heard": "Kestler", "probably": "Kessler"}]})

    qa.absorb(original, {"Kestler": "Kessler."}, client=client)

    assert original.likely_errors == []


def test_blank_answers_are_skipped_without_calling_claude():
    client = _answers({"corrections": []})

    updated = qa.absorb(ContextMap(), {"Kestler": "   ", "z-lanes": ""}, client=client)

    assert client.sent == [], "Claude was asked to interpret an empty answer"
    assert updated.likely_errors == []


def test_only_the_answers_given_are_sent():
    """Skipping is always allowed, and a skipped question must not arrive as
    though it had been answered."""
    client = _answers({"corrections": []})

    qa.absorb(ContextMap(), {"Kestler": "It is Kessler.", "z-lanes": ""}, client=client)

    sent = client.sent[0]["messages"][0]["content"]
    assert "Kestler" in sent
    assert "z-lanes" not in sent


def test_a_hedged_answer_keeps_its_marker():
    """Replying "I'm not sure" is not settling it. Dropping the guess there
    would take the marker out of the transcript and leave the mis-heard words
    standing unflagged, which is worse than never having asked."""
    existing = ContextMap(likely_errors=[
        {"heard": "Errol Markety", "probably": "Errol Marchetti", "confidence": "low"},
    ])
    client = _answers({
        "settled": [],
        "corrections": [],
        "still_open": ["Colleague could not recall the first name."],
    })

    updated = qa.absorb(existing, {"Errol Marchetti": "Not sure, could be Gareth."}, client=client)

    assert [e["probably"] for e in updated.likely_errors] == ["Errol Marchetti"]
    assert updated.open_questions == ["Colleague could not recall the first name."]


def test_a_marker_claimed_settled_but_never_asked_about_is_ignored():
    """The model can only settle what the colleague was actually shown."""
    existing = ContextMap(likely_errors=[
        {"heard": "Ridgelane", "probably": "Ridgeline", "confidence": "medium"},
    ])
    client = _answers({"settled": ["Ridgeline"], "corrections": []})

    updated = qa.absorb(existing, {"Kestler": "It is Kessler."}, client=client)

    assert [e["probably"] for e in updated.likely_errors] == ["Ridgeline"]


def test_a_settled_marker_echoed_with_brackets_still_supersedes():
    existing = ContextMap(likely_errors=[
        {"heard": "Simon", "probably": "Simone", "confidence": "medium"},
    ])
    client = _answers({"settled": ["[?Simone]"], "corrections": [
        {"heard": "Simon", "probably": "Simone Vasari", "confidence": "high"},
    ]})

    updated = qa.absorb(existing, {"Simone": "Simone Vasari."}, client=client)

    assert [e["probably"] for e in updated.likely_errors] == ["Simone Vasari"]


def test_the_questions_are_written_about_the_speakers_not_to_them():
    """The person assembling the write-up is often not on the recording. "Anya,
    on the tape you say..." addresses the wrong person and reads as though the
    tool has confused who it is talking to."""
    assert "third person" in qa.SYSTEM
    assert "assembling a write-up" in qa.SYSTEM
    assert "was physically present" not in qa.SYSTEM


def test_an_answer_replaces_an_unsettled_row_it_collides_with():
    """Appending is only safe because the settled rows are dropped first. A
    correction landing on words some *unsettled* row already claims would leave
    the old row firing, since `correct._proposals` sorts longest-heard first
    with a stable sort, and the colleague's own answer would come back reported
    as not found in the transcript. This is the same merge the revision pass
    uses, so the two cannot drift."""
    existing = ContextMap(likely_errors=[
        {"heard": "Kestler", "probably": "Kessler", "confidence": "low"},
    ])
    # They answered about a different marker, and their answer happens to name
    # the same heard words.
    client = _answers({"settled": ["60 hold"], "corrections": [
        {"heard": "Kestler", "probably": "Krakauer", "confidence": "high"},
        {"heard": "60 hold", "probably": "sixty holding beds", "confidence": "high"},
    ]})

    updated = qa.absorb(existing, {"60 hold": "Sixty holding beds, and it is Krakauer."},
                        client=client)

    rows = [e for e in updated.likely_errors if e["heard"] == "Kestler"]
    assert len(rows) == 1
    assert rows[0]["probably"] == "Krakauer"


def test_a_question_they_left_open_is_not_recorded_twice():
    """Three rounds ask about overlapping ground, so the same "still open" note
    comes back more than once and the writing prompt reads it as three
    unresolved things."""
    existing = ContextMap(open_questions=["Who is the St. Bede coordinator?"])
    client = _answers({"settled": [], "corrections": [],
                       "still_open": ["Who is the St. Bede coordinator?"]})

    updated = qa.absorb(existing, {"Simone": "Not sure."}, client=client)

    assert updated.open_questions == ["Who is the St. Bede coordinator?"]
