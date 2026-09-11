"""
Tests for the correction stage.

The property that matters is that word timings survive. Everything that plays
audio against text depends on them, and a rewrite that silently drops or
scrambles them still looks perfectly good in a Markdown document, which is the
only place anyone would look.

The second property is that an uncertain correction stays visibly uncertain.
Applied silently it is indistinguishable from a confident one and the reader
has no way back to what was actually said.

Run with: python -m pytest tests/test_correct.py -v
"""

import pytest

from src.podcastnotes import correct
from src.podcastnotes.context import ContextMap


def _words(*pairs):
    """(" word", start, end) triples into recogniser-shaped entries."""
    return [
        {"word": w, "start": s, "end": e, "probability": 0.9}
        for w, s, e in pairs
    ]


def _payload(text, words=None, speaker="Speaker 1"):
    return {
        "metadata": {"duration": 10.0},
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": text,
                "speaker": speaker,
                "confidence": 0.9,
                "words": words or [],
            }
        ],
    }


def _map(**kwargs):
    return ContextMap(likely_errors=[kwargs])


def _err(heard, probably, confidence="high"):
    return {"heard": heard, "probably": probably, "evidence": "e", "confidence": confidence}


# --- the substitution itself --------------------------------------------------


def test_a_confident_correction_is_applied():
    payload = _payload("We saw Dr. Ives Vahalee today.")

    result = correct.apply(payload, ContextMap(likely_errors=[_err("Ives Vahalee", "Idris Vahali")]))

    assert result.payload["segments"][0]["text"] == "We saw Dr. Idris Vahali today."
    assert result.applied == 1


def test_an_unsure_correction_is_marked_rather_than_applied_silently():
    """Applied silently it reads exactly like a confident one, and the reader
    has no way back to what was said."""
    payload = _payload("We met Errol Markety.")

    result = correct.apply(payload, ContextMap(likely_errors=[_err("Errol Markety", "Errol Marchetti", "low")]))

    assert result.payload["segments"][0]["text"] == "We met [?Errol Marchetti]."


@pytest.mark.parametrize("confidence", ["medium", "low", "", "HIGHISH"])
def test_anything_short_of_high_confidence_gets_marked(confidence):
    payload = _payload("the geneticists trial")

    result = correct.apply(payload, ContextMap(likely_errors=[_err("geneticists", "Helivar", confidence)]))

    assert "[?Helivar]" in result.payload["segments"][0]["text"]


def test_matching_ignores_case_but_the_replacement_does_not():
    payload = _payload("ridgelane and Ridgelane")

    result = correct.apply(payload, ContextMap(likely_errors=[_err("Ridgelane", "Ridgeline")]))

    assert result.payload["segments"][0]["text"] == "Ridgeline and Ridgeline"
    assert result.applied == 2


def test_punctuation_between_words_does_not_prevent_a_match():
    """The transcript writes "St. Bede's Fairmont" and the map may write the same
    thing with different separators. Tolerance is between tokens, not inside
    them: "John's" and "Johns" are deliberately not treated as the same token,
    because loosening that far starts matching things nobody intended in a
    document that gets published."""
    payload = _payload("we went to St. Bede's Fairmont road")

    result = correct.apply(payload, ContextMap(likely_errors=[
        _err("St Bede s Fairmont", "Lakeside St. Bede, Fairmount"),
    ]))

    assert "Fairmount" in result.payload["segments"][0]["text"]


def test_a_match_must_be_a_whole_word():
    """"OPS" inside "OPSEC" is the failure this prevents."""
    payload = _payload("the OPSEC briefing and the OPS unit")

    result = correct.apply(payload, ContextMap(likely_errors=[_err("OPS", "OBS")]))

    assert result.payload["segments"][0]["text"] == "the OPSEC briefing and the OBS unit"


def test_a_match_cannot_start_inside_a_longer_word():
    """Not hypothetical. Stanford is a candidate site in the Lanternfish
    documents, and it ends in "ford", so a correction keyed on Ford would
    rewrite it into nonsense. The trailing boundary does not catch this one,
    because the match sits at the end of the word rather than the start."""
    payload = _payload("we also looked at Stanford and Sutter")

    result = correct.apply(payload, ContextMap(likely_errors=[_err("Ford", "Lakeside General")]))

    assert result.payload["segments"][0]["text"] == "we also looked at Stanford and Sutter"


def test_variants_separated_by_slashes_are_each_tried():
    payload = _payload("abnormal trope and a second trope")

    result = correct.apply(payload, ContextMap(likely_errors=[_err("trope / abnormal trope", "troponin")]))

    assert "trope" not in result.payload["segments"][0]["text"]


def test_the_longest_variant_is_tried_first():
    """Otherwise the short one fires inside the long phrase and the long
    correction can never apply."""
    payload = _payload("we visited St. Bede's Lakesides yesterday")
    context = ContextMap(likely_errors=[
        _err("St. Bede's", "St. Bede"),
        _err("St. Bede's Lakesides", "Lakeside St. Bede"),
    ])

    result = correct.apply(payload, context)

    assert "Lakeside St. Bede" in result.payload["segments"][0]["text"]


def test_a_variant_too_short_to_be_safe_is_skipped():
    """One and two character variants only. Three is the floor rather than the
    ceiling because the corrections that matter most are acronyms: OPS to OBS,
    and CTA and EKG alongside them. Rejecting those to be cautious would throw
    away the substitutions this stage exists for."""
    payload = _payload("an X ray of the OPS unit")

    result = correct.apply(payload, ContextMap(likely_errors=[
        _err("X", "chest"),
        _err("OPS", "OBS"),
    ]))

    text = result.payload["segments"][0]["text"]
    assert text == "an X ray of the OBS unit"


def test_a_three_letter_acronym_is_still_corrected():
    payload = _payload("they read the EKG first")

    result = correct.apply(payload, ContextMap(likely_errors=[_err("EKG", "ECG")]))

    assert result.payload["segments"][0]["text"] == "they read the ECG first"


# --- word timings, the whole point --------------------------------------------


def test_a_one_for_one_substitution_keeps_every_timing():
    words = _words((" Ridgelane", 1.0, 1.5), (" Health", 1.5, 2.0))
    payload = _payload(" Ridgelane Health", words)

    result = correct.apply(payload, ContextMap(likely_errors=[_err("Ridgelane", "Ridgeline")]))

    got = result.payload["segments"][0]["words"]
    assert [w["word"].strip() for w in got] == ["Ridgeline", "Health"]
    assert (got[0]["start"], got[0]["end"]) == (1.0, 1.5)
    assert (got[1]["start"], got[1]["end"]) == (1.5, 2.0)


def test_two_words_becoming_one_keeps_the_span():
    """"12th lead" to "12-lead". The span is what the audio occupies; the
    boundary inside it was always an estimate."""
    words = _words((" the", 1.0, 1.2), (" 12th", 1.2, 1.6), (" lead", 1.6, 2.0))
    payload = _payload(" the 12th lead", words)

    result = correct.apply(payload, ContextMap(likely_errors=[_err("12th lead", "12-lead")]))

    got = result.payload["segments"][0]["words"]
    assert [w["word"].strip() for w in got] == ["the", "12-lead"]
    assert got[-1]["start"] == 1.2 and got[-1]["end"] == 2.0


def test_one_word_becoming_two_shares_the_span():
    words = _words((" abnormal", 1.0, 1.4), (" trope", 1.4, 2.0))
    payload = _payload(" abnormal trope", words)

    result = correct.apply(payload, ContextMap(likely_errors=[_err("trope", "cardiac troponin")]))

    got = result.payload["segments"][0]["words"]
    assert [w["word"].strip() for w in got] == ["abnormal", "cardiac", "troponin"]
    assert got[1]["start"] == 1.4
    assert got[-1]["end"] == 2.0


def test_timings_stay_in_order_after_a_substitution():
    words = _words((" the", 1.0, 1.2), (" 12th", 1.2, 1.6), (" lead", 1.6, 2.0))
    payload = _payload(" the 12th lead", words)

    result = correct.apply(payload, ContextMap(likely_errors=[_err("12th lead", "a b c d")]))

    got = result.payload["segments"][0]["words"]
    times = [t for w in got for t in (w["start"], w["end"])]
    assert times == sorted(times), "word timings went backwards"


def test_the_words_still_spell_the_segment_text():
    """They are two views of the same thing and must not drift apart."""
    words = _words((" the", 1.0, 1.2), (" 12th", 1.2, 1.6), (" lead", 1.6, 2.0))
    payload = _payload(" the 12th lead", words)

    result = correct.apply(payload, ContextMap(likely_errors=[_err("12th lead", "12-lead")]))

    segment = result.payload["segments"][0]
    assert "".join(w["word"] for w in segment["words"]).strip() == segment["text"].strip()


def test_trailing_punctuation_on_a_replaced_word_survives():
    words = _words((" Ridgelane,", 1.0, 1.5), (" then", 1.5, 2.0))
    payload = _payload(" Ridgelane, then", words)

    result = correct.apply(payload, ContextMap(likely_errors=[_err("Ridgelane", "Ridgeline")]))

    got = result.payload["segments"][0]["words"]
    assert got[0]["word"].strip() == "Ridgeline,"


def test_the_run_ends_where_the_recogniser_said_even_when_the_span_is_degenerate():
    """The last word's end is pinned to the recorded end rather than to wherever
    the proportional arithmetic happens to land. On ordinary input the two
    agree, which is exactly why this needs a case where they cannot: faster
    whisper occasionally emits a word whose end precedes its start, and without
    the pin the rebuilt run would finish after the segment it belongs to."""
    words = [{"word": " trope", "start": 2.0, "end": 1.0, "probability": 0.9}]
    payload = _payload(" trope", words)

    result = correct.apply(payload, ContextMap(likely_errors=[_err("trope", "cardiac troponin")]))

    got = result.payload["segments"][0]["words"]
    assert got[-1]["end"] == 1.0, "the run ran past the end the recogniser gave"


def test_a_run_with_no_recogniser_timings_is_not_given_invented_ones():
    words = [{"word": " trope", "start": None, "end": None, "probability": 0.5}]
    payload = _payload(" trope", words)

    result = correct.apply(payload, ContextMap(likely_errors=[_err("trope", "troponin")]))

    got = result.payload["segments"][0]["words"][0]
    assert got["start"] is None and got["end"] is None


def test_a_segment_with_no_words_still_gets_its_text_corrected():
    """Word timings are optional in the export; the text is not."""
    payload = _payload("Ridgelane Health", words=[])

    result = correct.apply(payload, ContextMap(likely_errors=[_err("Ridgelane", "Ridgeline")]))

    assert result.payload["segments"][0]["text"] == "Ridgeline Health"


# --- the account of what changed ----------------------------------------------


def test_every_change_is_reported_with_where_it_landed():
    payload = {
        "segments": [
            {"text": "Ridgelane Northgate", "words": []},
            {"text": "back to Ridgelane", "words": []},
        ]
    }

    result = correct.apply(payload, ContextMap(likely_errors=[_err("Ridgelane", "Ridgeline")]))

    change = result.changes[0]
    assert change.occurrences == 2
    assert change.segments == [0, 1]
    assert change.marked is False


def test_a_correction_that_matched_nothing_is_reported_not_dropped():
    """Silently discarding it hides a context map that is drifting away from
    the recording it was built for."""
    payload = _payload("nothing relevant here")

    result = correct.apply(payload, ContextMap(likely_errors=[_err("Ridgelane", "Ridgeline")]))

    assert [c.heard for c in result.unmatched] == ["Ridgelane"]
    assert result.changes == []


def test_the_original_payload_is_left_alone():
    """A caller showing a before and after has no other way back."""
    payload = _payload("Ridgelane Health", _words((" Ridgelane", 1.0, 1.5)))

    correct.apply(payload, ContextMap(likely_errors=[_err("Ridgelane", "Ridgeline")]))

    assert payload["segments"][0]["text"] == "Ridgelane Health"
    assert payload["segments"][0]["words"][0]["word"] == " Ridgelane"


def test_an_empty_context_map_changes_nothing():
    payload = _payload("Ridgelane Health")

    result = correct.apply(payload, ContextMap())

    assert result.payload["segments"][0]["text"] == "Ridgelane Health"
    assert result.applied == 0


def test_a_correction_missing_either_side_is_ignored():
    payload = _payload("Ridgelane Health")

    result = correct.apply(payload, ContextMap(likely_errors=[
        {"heard": "Ridgelane", "probably": "", "confidence": "high"},
        {"heard": "", "probably": "Ridgeline", "confidence": "high"},
    ]))

    assert result.payload["segments"][0]["text"] == "Ridgelane Health"


def test_speaker_and_other_segment_fields_are_untouched():
    payload = _payload("Ridgelane", speaker="Speaker 2")

    result = correct.apply(payload, ContextMap(likely_errors=[_err("Ridgelane", "Ridgeline")]))

    segment = result.payload["segments"][0]
    assert segment["speaker"] == "Speaker 2"
    assert segment["confidence"] == 0.9
    assert result.payload["metadata"]["duration"] == 10.0
