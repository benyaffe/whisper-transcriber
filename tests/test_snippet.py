"""
Tests for the per-speaker audio clips used to put a name to a voice.

Selection is pure data and tested without touching ffmpeg. Extraction is tested
against the one-second WAV the conftest fixture builds, plus a mocked argv check
for the one detail nothing else would notice if it regressed.

Run with: python -m pytest tests/test_snippet.py -v
"""

import json
import os

import pytest

from src.core.snippet import (
    DEFAULT_CLIP_SECONDS,
    ClipExtractionError,
    SpeakerClip,
    extract_clip,
    extract_speaker_clips,
    pick_speaker_clips,
)


def payload(*, turns=None, exclusive=None, speakers=None, segments=None):
    return {
        "version": 1,
        "segments": segments or [],
        "speakers": speakers if speakers is not None else [
            {"id": "Speaker 1", "pyannote_label": "SPEAKER_00"},
        ],
        "diarization": {
            "turns": turns or [],
            "exclusive_turns": exclusive if exclusive is not None else [],
        },
    }


def turn(start, end, speaker="SPEAKER_00"):
    return {"start": start, "end": end, "speaker": speaker}


def segment(start, end, text, speaker="Speaker 1"):
    return {"start": start, "end": end, "text": text, "speaker": speaker}


# --- selection ----------------------------------------------------------------


def test_longest_turn_wins():
    clips = pick_speaker_clips(payload(exclusive=[
        turn(0.0, 2.0), turn(10.0, 19.0), turn(30.0, 33.0),
    ]))

    assert len(clips) == 1
    assert clips[0].start == 10.0


def test_exclusive_turns_are_preferred_over_overlapping_ones():
    """The whole reason this reads the exclusive timeline.

    A window where two people talk at once is the worst possible clip to hand
    somebody identifying a voice. In the real Ashford recording this is a 1.2s
    difference for Speaker 3.
    """
    clips = pick_speaker_clips(payload(
        turns=[turn(100.0, 130.0)],       # long, but overlapped
        exclusive=[turn(200.0, 210.0)],   # shorter, but clean
    ))

    assert clips[0].start == 200.0


def test_falls_back_to_overlapping_turns_when_there_is_no_exclusive_view():
    """Otherwise a pipeline without an exclusive timeline yields no clips."""
    clips = pick_speaker_clips(payload(turns=[turn(5.0, 25.0)], exclusive=[]))

    assert clips[0].start == 5.0


def test_a_long_turn_is_truncated_to_the_cap():
    clips = pick_speaker_clips(payload(exclusive=[turn(100.0, 200.0)]), max_seconds=6.0)

    assert clips[0].start == 100.0
    assert clips[0].duration == 6.0
    assert clips[0].end == 106.0


def test_a_short_turn_gives_a_short_clip_rather_than_nothing():
    """The agreed behavior for someone who only ever says "mm-hmm".

    They are the hardest person to identify, so a 1.5s clip beats no clip.
    """
    clips = pick_speaker_clips(payload(exclusive=[turn(4.0, 5.5)]), max_seconds=6.0)

    assert clips[0].duration == pytest.approx(1.5)


def test_a_speaker_with_no_turns_gets_no_clip():
    """Do not invent a window for someone who never had a clean turn."""
    clips = pick_speaker_clips(payload(
        exclusive=[turn(0.0, 5.0, "SPEAKER_00")],
        speakers=[
            {"id": "Speaker 1", "pyannote_label": "SPEAKER_00"},
            {"id": "Speaker 2", "pyannote_label": "SPEAKER_01"},  # never speaks alone
        ],
    ))

    assert [c.speaker for c in clips] == ["Speaker 1"]


def test_zero_length_turns_are_ignored():
    clips = pick_speaker_clips(payload(exclusive=[
        turn(1.0, 1.0), turn(2.0, 2.0), turn(5.0, 8.0),
    ]))

    assert clips[0].start == 5.0


def test_a_speaker_with_only_zero_length_turns_gets_no_clip():
    """Otherwise ffmpeg is asked for a zero-second cut and writes an empty file.

    Degenerate turns are rare but reachable: pyannote can emit a boundary of
    zero width, and min_duration_off is 0.0 in the shipped config.
    """
    clips = pick_speaker_clips(payload(exclusive=[turn(1.0, 1.0), turn(4.0, 4.0)]))

    assert clips == []


def test_transcript_text_for_the_window_is_attached():
    clips = pick_speaker_clips(payload(
        exclusive=[turn(10.0, 16.0)],
        segments=[
            segment(0.0, 5.0, "before the window"),
            segment(9.0, 12.0, "overlapping the start"),
            segment(12.0, 15.0, "fully inside"),
            segment(30.0, 35.0, "long after"),
        ],
    ))

    assert clips[0].text == "overlapping the start fully inside"


def test_a_window_with_no_transcript_gives_empty_text():
    clips = pick_speaker_clips(payload(
        exclusive=[turn(100.0, 106.0)],
        segments=[segment(0.0, 5.0, "elsewhere")],
    ))

    assert clips[0].text == ""


def test_pyannote_label_is_carried_through():
    """Display numbering follows transcript order, not pyannote's label order.

    Anything deriving the raw label from the digit in "Speaker 3" reads the
    wrong person, which is why the mapping travels with the clip.
    """
    clips = pick_speaker_clips(payload(
        exclusive=[turn(0.0, 8.0, "SPEAKER_02")],
        speakers=[{"id": "Speaker 1", "pyannote_label": "SPEAKER_02"}],
    ))

    assert (clips[0].speaker, clips[0].pyannote_label) == ("Speaker 1", "SPEAKER_02")


def test_default_cap_is_six_seconds():
    assert DEFAULT_CLIP_SECONDS == 6.0


# --- extraction ---------------------------------------------------------------


def test_fast_seek_puts_ss_before_the_input(monkeypatch, tmp_path):
    """A silent 2x that nothing else would catch.

    Measured on the real recording at a 588s offset: 0.29s with -ss before -i,
    0.63s after it. ffmpeg jumps to the offset instead of decoding up to it.
    """
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        out = cmd[-1]
        with open(out, "wb") as f:
            f.write(b"RIFF____WAVEfmt ")
        from types import SimpleNamespace
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr("src.core.snippet.subprocess.run", fake_run)

    clip = SpeakerClip("Speaker 1", "SPEAKER_00", start=588.0, duration=6.0)
    extract_clip("/tmp/audio.m4a", clip, str(tmp_path))

    cmd = seen["cmd"]
    assert cmd.index("-ss") < cmd.index("-i"), " ".join(cmd)
    assert cmd[cmd.index("-ss") + 1] == "588.000"
    assert cmd[cmd.index("-t") + 1] == "6.000"


def test_extraction_produces_a_playable_file(temp_audio_file, tmp_path):
    """Against the real bundled ffmpeg and the 1s WAV from conftest."""
    clip = SpeakerClip("Speaker 2", "SPEAKER_01", start=0.0, duration=0.5)

    extract_clip(temp_audio_file, clip, str(tmp_path / "clips"))

    assert clip.path is not None
    assert os.path.exists(clip.path)
    assert os.path.getsize(clip.path) > 44  # bigger than a bare WAV header
    assert clip.path.endswith("Speaker_2.wav")


def test_a_failing_ffmpeg_raises_and_leaves_no_file(monkeypatch, tmp_path):
    """Never leave a truncated clip where a caller might play it."""
    def fake_run(cmd, **kwargs):
        out = cmd[-1]
        with open(out, "wb") as f:
            f.write(b"")  # zero-byte leftover
        from types import SimpleNamespace
        return SimpleNamespace(returncode=1, stderr=b"Invalid data found")

    monkeypatch.setattr("src.core.snippet.subprocess.run", fake_run)

    clip = SpeakerClip("Speaker 1", "SPEAKER_00", start=0.0, duration=6.0)

    with pytest.raises(ClipExtractionError, match="Invalid data found"):
        extract_clip("/tmp/audio.m4a", clip, str(tmp_path))

    assert os.listdir(tmp_path) == []


def test_end_to_end_from_a_json_file(temp_audio_file, tmp_path):
    json_path = tmp_path / "run.json"
    json_path.write_text(json.dumps(payload(
        exclusive=[turn(0.0, 0.4, "SPEAKER_00"), turn(0.5, 0.9, "SPEAKER_01")],
        speakers=[
            {"id": "Speaker 1", "pyannote_label": "SPEAKER_00"},
            {"id": "Speaker 2", "pyannote_label": "SPEAKER_01"},
        ],
        segments=[segment(0.0, 0.4, "hello")],
    )), encoding="utf-8")

    clips = extract_speaker_clips(
        str(json_path), temp_audio_file, str(tmp_path / "clips"), max_seconds=6.0
    )

    assert [c.speaker for c in clips] == ["Speaker 1", "Speaker 2"]
    assert all(os.path.exists(c.path) for c in clips)
    assert clips[0].text == "hello"


def test_snippet_module_imports_no_qt():
    """Same headless guarantee as runner.py."""
    import ast
    import inspect

    from src.core import snippet

    tree = ast.parse(inspect.getsource(snippet))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    qt = sorted(m for m in imported if m.split(".")[0] in {"PyQt6", "PyQt5", "PySide6"})
    assert qt == [], f"snippet imports Qt: {qt}"
