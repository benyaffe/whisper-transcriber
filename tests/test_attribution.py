"""
Tests for the attribution stage.

The rule worth defending is that nothing is applied without being confirmed.
A wrong attribution is the failure mode with no tell: a transcript that puts
the GM's opinion in the hardware engineer's mouth reads perfectly well and is
completely wrong, and no later stage can detect it.

The second concern is the diarizer's invented speaker, which is not
hypothetical. The real Ashford recording has a fourth label with 22 seconds of
speech who is not a fourth participant.

Run with: python -m pytest tests/test_attribution.py -v
"""

import json

import pytest

from src.podcastnotes import attribution
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


def _seg(speaker, text, start=0.0, end=1.0):
    return {"speaker": speaker, "text": text, "start": start, "end": end, "words": []}


def _payload(segments, speakers=None):
    return {
        "segments": segments,
        "speakers": speakers
        or [
            {"id": "Speaker 1", "pyannote_label": "SPEAKER_03", "total_speech_s": 900.0},
            {"id": "Speaker 2", "pyannote_label": "SPEAKER_01", "total_speech_s": 700.0},
        ],
        "diarization": {"turns": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_03"}]},
    }


ASHFORD = _payload(
    [
        _seg("Speaker 1", "The first place we went to was Ridgeline Health, Northgate, Westvale."),
        _seg("Speaker 2", "The patient demographic seems to be well insured, lots of private insurance."),
        _seg("Speaker 1", "Location did not seem to be an issue and they were willing to transport."),
    ]
)


# --- reading the voices -------------------------------------------------------


def test_every_label_becomes_a_voice():
    found = attribution.voices(ASHFORD)

    assert [v.speaker for v in found] == ["Speaker 1", "Speaker 2"]


def test_voices_are_ordered_by_how_much_they_speak():
    """The person reviewing meets the main voices first and the doubtful
    fragments last."""
    payload = _payload(
        [_seg("Speaker 1", "a"), _seg("Speaker 2", "b")],
        speakers=[
            {"id": "Speaker 1", "pyannote_label": "P0", "total_speech_s": 10.0},
            {"id": "Speaker 2", "pyannote_label": "P1", "total_speech_s": 900.0},
        ],
    )

    assert [v.speaker for v in attribution.voices(payload)] == ["Speaker 2", "Speaker 1"]


def test_the_sample_lines_are_the_longest_ones():
    """An opening "yeah, exactly" identifies nobody. A long turn carries both
    the subject matter and the cadence."""
    payload = _payload([
        _seg("Speaker 1", "Yeah."),
        _seg("Speaker 1", "They did do CCTA but only between eight and five, not at night."),
    ])

    lines = attribution.voices(payload, sample_lines=1)[0].lines

    assert lines == ["They did do CCTA but only between eight and five, not at night."]


def test_a_label_with_little_speech_is_flagged_as_doubtful():
    """The real Ashford recording has one of these: 22 seconds, and not a
    fourth participant."""
    payload = _payload(
        [_seg("Speaker 4", "most of the stuff was a Stryker bed")],
        speakers=[{"id": "Speaker 4", "pyannote_label": "SPEAKER_00", "total_speech_s": 21.8}],
    )

    assert attribution.voices(payload)[0].is_slight is True


def test_speech_time_falls_back_to_the_segments_when_there_are_no_profiles():
    payload = {"segments": [_seg("Speaker 1", "hello", start=0.0, end=4.0)]}

    assert attribution.voices(payload)[0].total_speech_s == 4.0


def test_reading_the_voices_does_not_touch_the_payload():
    before = json.dumps(ASHFORD, sort_keys=True)

    attribution.voices(ASHFORD)

    assert json.dumps(ASHFORD, sort_keys=True) == before


# --- what Claude is asked and what it says ------------------------------------


def test_the_prompt_carries_the_people_from_the_context_map():
    """Without them the stage is guessing at names it has never seen."""
    client = _answers({"speakers": []})
    context = ContextMap(people=[{"name": "Anya Petrov-Hale", "role": "GM"}])

    attribution.suggest(ASHFORD, context, description="Ashford trip", client=client)

    sent = client.sent[0]["messages"][0]["content"]
    assert "Anya Petrov-Hale" in sent
    assert "Ashford trip" in sent


def test_the_hardest_question_gets_the_most_effort():
    client = _answers({"speakers": []})

    attribution.suggest(ASHFORD, ContextMap(), client=client)

    assert client.sent[0]["output_config"]["effort"] == "max"


def test_a_suggestion_comes_back_per_speaker():
    client = _answers({"speakers": [
        {"speaker": "Speaker 1", "name": "Marcus Ellery", "confidence": "high",
         "evidence": "walks through the sites", "same_as": ""},
    ]})

    found = attribution.suggest(ASHFORD, ContextMap(), client=client)

    assert found[0].name == "Marcus Ellery"
    assert found[0].confidence == "high"


def test_a_suggestion_for_a_speaker_that_does_not_exist_is_dropped():
    """It could never be confirmed or applied, and would put a row on the
    review screen that does nothing."""
    client = _answers({"speakers": [
        {"speaker": "Speaker 9", "name": "Nobody", "confidence": "high"},
    ]})

    assert attribution.suggest(ASHFORD, ContextMap(), client=client) == []


def test_an_empty_name_is_kept_as_an_empty_name():
    """Not knowing is a good answer here. A plausible guess is not, because the
    name is published against somebody's words."""
    client = _answers({"speakers": [
        {"speaker": "Speaker 1", "name": "", "confidence": "low", "evidence": "cannot tell"},
    ]})

    assert attribution.suggest(ASHFORD, ContextMap(), client=client)[0].name == ""


def test_unparseable_output_is_no_suggestions_rather_than_a_crash():
    client = _FakeClient([_Message([_Text("I could not work this out.")])])

    assert attribution.suggest(ASHFORD, ContextMap(), client=client) == []


def test_a_speaker_cannot_be_the_same_as_itself():
    client = _answers({"speakers": [
        {"speaker": "Speaker 1", "name": "Anya", "same_as": "Speaker 1"},
    ]})

    assert attribution.suggest(ASHFORD, ContextMap(), client=client)[0].same_as == ""


def test_a_merge_onto_an_unknown_speaker_is_discarded():
    client = _answers({"speakers": [
        {"speaker": "Speaker 1", "name": "Anya", "same_as": "Speaker 7"},
    ]})

    assert attribution.suggest(ASHFORD, ContextMap(), client=client)[0].same_as == ""


# --- confirmation, which is the whole point -----------------------------------


def test_a_confirmed_speaker_is_renamed_everywhere():
    result = attribution.apply(ASHFORD, {"Speaker 1": "Marcus Ellery"})

    assert [s["speaker"] for s in result["segments"]] == [
        "Marcus Ellery", "Speaker 2", "Marcus Ellery",
    ]
    assert result["speakers"][0]["id"] == "Marcus Ellery"


def test_an_unconfirmed_speaker_keeps_its_label():
    """An unreviewed voice stays visibly unreviewed rather than quietly
    becoming somebody."""
    result = attribution.apply(ASHFORD, {"Speaker 1": "Marcus Ellery"})

    assert result["segments"][1]["speaker"] == "Speaker 2"
    assert result["speakers"][1]["id"] == "Speaker 2"


def test_nothing_at_all_is_applied_without_decisions():
    result = attribution.apply(ASHFORD, {})

    assert [s["speaker"] for s in result["segments"]] == ["Speaker 1", "Speaker 2", "Speaker 1"]


def test_an_empty_name_is_not_applied():
    result = attribution.apply(ASHFORD, {"Speaker 1": "   "})

    assert result["segments"][0]["speaker"] == "Speaker 1"


def test_two_labels_can_be_confirmed_as_one_person():
    """How the diarizer's invented speaker is merged back into whoever it was
    split from."""
    payload = _payload([_seg("Speaker 2", "a"), _seg("Speaker 4", "b")])

    result = attribution.apply(payload, {"Speaker 2": "Anya", "Speaker 4": "Anya"})

    assert [s["speaker"] for s in result["segments"]] == ["Anya", "Anya"]


def test_the_pyannote_label_is_never_renamed():
    """It is the link to the raw diarizer output and to the row order of the
    embeddings, so the display name is the only thing that may move."""
    result = attribution.apply(ASHFORD, {"Speaker 1": "Marcus Ellery"})

    assert result["speakers"][0]["pyannote_label"] == "SPEAKER_03"
    assert result["diarization"]["turns"][0]["speaker"] == "SPEAKER_03"


def test_applying_leaves_the_original_alone():
    attribution.apply(ASHFORD, {"Speaker 1": "Marcus Ellery"})

    assert ASHFORD["segments"][0]["speaker"] == "Speaker 1"


def test_a_decision_for_an_unknown_speaker_changes_nothing():
    result = attribution.apply(ASHFORD, {"Speaker 9": "Ghost"})

    assert [s["speaker"] for s in result["segments"]] == ["Speaker 1", "Speaker 2", "Speaker 1"]


# --- merges -------------------------------------------------------------------


def test_a_merge_chain_settles_on_one_target():
    found = [
        attribution.Suggestion("Speaker 4", same_as="Speaker 3"),
        attribution.Suggestion("Speaker 3", same_as="Speaker 2"),
    ]

    assert attribution.merges(found) == {"Speaker 4": "Speaker 2", "Speaker 3": "Speaker 2"}


def test_a_merge_cycle_does_not_hang():
    """Two labels each claiming to be the other. Following the chain naively
    never terminates."""
    found = [
        attribution.Suggestion("Speaker 1", same_as="Speaker 2"),
        attribution.Suggestion("Speaker 2", same_as="Speaker 1"),
    ]

    resolved = attribution.merges(found)

    assert set(resolved) == {"Speaker 1", "Speaker 2"}


def test_speakers_claiming_nothing_are_not_in_the_merge_map():
    found = [attribution.Suggestion("Speaker 1", name="Anya")]

    assert attribution.merges(found) == {}


# --- when a guess would be a coin toss ------------------------------------------


def test_a_confident_suggestion_is_worth_filling_in():
    Suggestion = attribution.Suggestion

    assert Suggestion("Speaker 1", name="Marcus", confidence="high").worth_filling_in
    assert Suggestion("Speaker 1", name="Marcus", confidence="medium").worth_filling_in


def test_a_low_confidence_suggestion_is_not():
    Suggestion = attribution.Suggestion
    """Three runs of the same recording with identical diarization returned
    Anya, Anya, then Devan for the 22-second speaker. The answer flips, so it
    must not arrive looking like the confident ones."""
    assert not Suggestion("Speaker 4", name="Anya", confidence="low").worth_filling_in


def test_an_unstated_confidence_is_not_a_high_one():
    Suggestion = attribution.Suggestion

    assert not Suggestion("Speaker 1", name="Marcus").worth_filling_in


def test_no_name_is_never_worth_filling_in():
    Suggestion = attribution.Suggestion

    assert not Suggestion("Speaker 1", name="", confidence="high").worth_filling_in


def test_claude_is_told_that_choosing_between_two_people_is_not_knowing():
    """It returned a low-confidence name rather than an empty one, which is
    the behaviour this instruction exists to change."""
    assert "choosing between two people" in attribution.SYSTEM
