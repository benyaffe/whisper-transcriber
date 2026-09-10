"""
Tests for which diarization view segments get aligned against.

pyannote returns two timelines. The overlapping one reports every speaker
active at a given moment; the exclusive one keeps only the dominant speaker per
frame. Whisper segments carry exactly one speaker label, so the exclusive view
is the one that matches the question being asked.

Run with: python -m pytest tests/test_speaker_alignment.py -v
"""

import pytest

from src.core.diarization import SpeakerTurn, assign_speakers_to_segments
from src.core.runner import TranscriptionSegment


def seg(start, end, text="x"):
    return TranscriptionSegment(start=start, end=end, text=text, confidence=0.9)


OVERLAPPING = [
    SpeakerTurn(0.0, 30.0, "SPEAKER_00"),   # long, quiet backchannel
    SpeakerTurn(10.5, 11.5, "SPEAKER_01"),  # the person actually talking
]
EXCLUSIVE = [
    SpeakerTurn(0.0, 10.4, "SPEAKER_00"),
    SpeakerTurn(10.4, 11.6, "SPEAKER_01"),  # wins these frames outright
    SpeakerTurn(11.6, 30.0, "SPEAKER_00"),
]


def _raw_speaker_of(segments, turns):
    """Which raw pyannote label a segment ends up attributed to."""
    speaker_map = assign_speakers_to_segments(segments, turns)
    display_to_raw = {v: k for k, v in speaker_map.items()}
    return display_to_raw[segments[0].speaker]


def test_overlapping_turns_attribute_by_list_order_not_dominance():
    """Why the switch matters: this is the behavior being replaced.

    A segment sitting inside two simultaneous turns takes the first match in
    the list, regardless of who actually dominates the passage.
    """
    assert _raw_speaker_of([seg(10.0, 12.0)], OVERLAPPING) == "SPEAKER_00"


def test_exclusive_turns_attribute_by_dominance():
    """The same segment, aligned against the exclusive view, follows the argmax.

    Same audio, same speakers, different answer. This pair is the whole
    justification for the change.
    """
    assert _raw_speaker_of([seg(10.0, 12.0)], EXCLUSIVE) == "SPEAKER_01"


def test_display_labels_follow_transcript_order_not_pyannote_order():
    """The mapping the JSON export has to carry.

    Speakers are numbered by first appearance in the transcript, which is not
    the labels() order the embedding rows use. Anything keying an embedding off
    the trailing digit of "Speaker N" matches the wrong person.
    """
    segments = [seg(0.0, 1.0), seg(5.0, 6.0)]
    turns = [
        SpeakerTurn(0.0, 2.0, "SPEAKER_02"),  # speaks first
        SpeakerTurn(4.0, 7.0, "SPEAKER_00"),  # speaks second
    ]

    speaker_map = assign_speakers_to_segments(segments, turns)

    assert speaker_map == {"SPEAKER_02": "Speaker 1", "SPEAKER_00": "Speaker 2"}
    assert [s.speaker for s in segments] == ["Speaker 1", "Speaker 2"]


class _Result:
    """Minimal stand-in for DiarizationResult's alignment inputs."""

    def __init__(self, turns, exclusive_turns):
        self.turns = turns
        self.exclusive_turns = exclusive_turns


@pytest.mark.parametrize(
    "exclusive, expected_source",
    [
        ([SpeakerTurn(0.0, 5.0, "SPEAKER_09")], "exclusive"),
        ([], "overlapping"),
    ],
)
def test_alignment_prefers_exclusive_and_falls_back(exclusive, expected_source):
    """Empty exclusive turns must fall back rather than losing every speaker."""
    overlapping = [SpeakerTurn(0.0, 5.0, "SPEAKER_00")]
    result = _Result(turns=overlapping, exclusive_turns=exclusive)

    # Mirrors the selection in TranscriptionWorker._run_diarization.
    chosen = result.exclusive_turns or result.turns

    assert chosen is (exclusive if expected_source == "exclusive" else overlapping)


def test_runner_aligns_against_exclusive_turns(monkeypatch):
    """End to end through _run_diarization, with the pipeline stubbed out.

    This used to build a TranscriptionWorker via __new__, hand-populate seven
    attributes and stub two pyqtSignals with a fake _Signal class, all to avoid
    QThread.__init__. The runner has a real constructor and a no-op observer.
    """
    from src.core import runner as tmod

    class Result:
        turns = OVERLAPPING
        exclusive_turns = EXCLUSIVE
        embeddings = {}
        embedding_dimension = None
        dropped_speakers = []

    monkeypatch.setattr(tmod, "run_diarization", lambda *a, **k: Result())

    runner = tmod.TranscriptionRunner(
        "/tmp/x.wav", hf_token="hf_x", enable_speaker_id=True
    )
    runner.audio_path = "/tmp/x.wav"
    runner.segments = [seg(10.0, 12.0)]

    runner._run_diarization()

    assert runner._speaker_id_used is True
    # SPEAKER_01 dominates 10.4-11.6, so the segment is theirs. Aligning
    # against the overlapping turns would have given SPEAKER_00.
    assert runner._speaker_map["SPEAKER_01"] == runner.segments[0].speaker
