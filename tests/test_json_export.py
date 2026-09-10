"""
Tests for the machine-readable transcript export.

This file is the contract between Whisper Transcriber and everything
downstream, so the shape matters as much as the values. It also has to survive
json.dumps, which the numpy scalars pyannote returns do not.

Run with: python -m pytest tests/test_json_export.py -v
"""

import json

import pytest

from src.core.diarization import DiarizationResult, SpeakerTurn
from src.core.json_export import SCHEMA_VERSION, build_payload, write_json
from src.core.transcriber import TranscriptionSegment, WordTiming
from src.utils.file_utils import generate_output_paths


def segment(start, end, text, speaker=None, words=()):
    return TranscriptionSegment(
        start=start,
        end=end,
        text=text,
        confidence=0.91234,
        speaker=speaker,
        words=[WordTiming(*w) for w in words],
    )


@pytest.fixture
def diarized():
    """A two-speaker result with one speaker lacking a usable profile."""
    return DiarizationResult(
        turns=[
            SpeakerTurn(0.0, 4.0, "SPEAKER_01"),
            SpeakerTurn(4.0, 10.0, "SPEAKER_00"),
        ],
        exclusive_turns=[
            SpeakerTurn(0.0, 4.0, "SPEAKER_01"),
            SpeakerTurn(4.0, 10.0, "SPEAKER_00"),
        ],
        embeddings={"SPEAKER_01": [0.1] * 256},
        embedding_dimension=256,
        dropped_speakers=["SPEAKER_00"],
    )


@pytest.fixture
def payload(diarized):
    return build_payload(
        source_path="/some/where/Lakeside StBede.m4a",
        segments=[
            segment(0.0, 2.0, "Hello there", "Speaker 1",
                    words=[(0.0, 0.5, " Hello", 0.98), (0.5, 2.0, " there", 0.87)]),
            segment(4.0, 6.0, "Second speaker", "Speaker 2"),
        ],
        duration=756.202667,
        model="medium",
        language="en",
        device="mps",
        speaker_id_used=True,
        # SPEAKER_01 speaks first, so it becomes Speaker 1 despite sorting second.
        speaker_map={"SPEAKER_01": "Speaker 1", "SPEAKER_00": "Speaker 2"},
        diarization=diarized,
    )


def test_payload_is_json_serializable(payload):
    """numpy floats would raise here, which is the whole reason for the casts."""
    text = json.dumps(payload)
    assert json.loads(text) == payload


def test_top_level_shape(payload):
    assert payload["version"] == SCHEMA_VERSION
    assert set(payload) == {"version", "metadata", "segments", "speakers", "diarization"}


def test_metadata_block(payload):
    meta = payload["metadata"]
    assert meta["source_file"] == "Lakeside StBede.m4a"   # basename, not full path
    assert meta["duration"] == pytest.approx(756.203)
    assert meta["model"] == "medium"
    assert meta["language"] == "en"
    assert meta["device"] == "mps"
    assert meta["speaker_id_used"] is True
    assert meta["created_at"].endswith("+00:00")
    assert meta["app_version"] != ""


def test_app_version_comes_from_the_version_file(payload):
    """Same single source of truth build.sh and the spec read."""
    import os
    from src.core.json_export import _REPO_ROOT

    with open(os.path.join(_REPO_ROOT, "VERSION"), encoding="utf-8") as f:
        assert payload["metadata"]["app_version"] == f.read().strip()


def test_word_timings_round_trip(payload):
    words = payload["segments"][0]["words"]
    assert [w["word"] for w in words] == [" Hello", " there"]
    assert words[0]["start"] == 0.0
    assert words[0]["probability"] == pytest.approx(0.98)


def test_segment_without_words_still_serializes(payload):
    assert payload["segments"][1]["words"] == []


def test_speaker_entries_carry_the_label_mapping(payload):
    """The field that stops Speaker Library lookups matching the wrong person.

    Display numbering follows first appearance in the transcript, while
    embedding rows follow pyannote's sorted labels(). Here the two disagree:
    Speaker 1 is SPEAKER_01, not SPEAKER_00.
    """
    by_id = {s["id"]: s for s in payload["speakers"]}

    assert by_id["Speaker 1"]["pyannote_label"] == "SPEAKER_01"
    assert by_id["Speaker 2"]["pyannote_label"] == "SPEAKER_00"


def test_speaker_without_a_usable_profile_has_no_embedding_key(payload):
    """Absent, not null. A consumer must not mistake a padded row for a voice."""
    by_id = {s["id"]: s for s in payload["speakers"]}

    assert len(by_id["Speaker 1"]["embedding"]) == 256
    assert "embedding" not in by_id["Speaker 2"]
    assert payload["diarization"]["speakers_without_profile"] == ["SPEAKER_00"]


def test_total_speech_is_summed_per_speaker(payload):
    by_id = {s["id"]: s for s in payload["speakers"]}

    assert by_id["Speaker 1"]["total_speech_s"] == pytest.approx(4.0)
    assert by_id["Speaker 2"]["total_speech_s"] == pytest.approx(6.0)


def test_both_timelines_are_exported(payload):
    diar = payload["diarization"]

    assert len(diar["turns"]) == 2
    assert len(diar["exclusive_turns"]) == 2
    assert diar["embedding_dimension"] == 256
    assert diar["turns"][0] == {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_01"}


def test_speaker_id_off_still_produces_a_valid_file():
    """The JSON is written on every run, so this path has to hold up."""
    payload = build_payload(
        source_path="/tmp/x.m4a",
        segments=[segment(0.0, 2.0, "No speakers here")],
        duration=2.0,
        model="medium",
        language=None,
        device="cpu",
        speaker_id_used=False,
    )

    json.dumps(payload)
    assert payload["metadata"]["speaker_id_used"] is False
    assert payload["metadata"]["language"] is None
    assert "speakers" not in payload
    assert "diarization" not in payload
    assert payload["segments"][0]["speaker"] is None


def test_speaker_labels_suppressed_when_speaker_id_failed():
    """A stale label on a segment must not leak out as if it were real."""
    payload = build_payload(
        source_path="/tmp/x.m4a",
        segments=[segment(0.0, 2.0, "text", speaker="Speaker 1")],
        duration=2.0,
        model="medium",
        language="en",
        device="cpu",
        speaker_id_used=False,
    )

    assert payload["segments"][0]["speaker"] is None


def test_write_json_round_trips(tmp_path, payload):
    path = tmp_path / "out.json"
    write_json(str(path), payload)

    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_generate_output_paths_includes_json():
    vtt, txt, js = generate_output_paths("/a/b/Recording.m4a")

    assert (vtt, txt, js) == ("/a/b/Recording.vtt", "/a/b/Recording.txt", "/a/b/Recording.json")


def test_non_ascii_text_is_preserved(tmp_path):
    """ensure_ascii=False, so the file stays readable rather than escaped."""
    payload = build_payload(
        source_path="/tmp/x.m4a",
        segments=[segment(0.0, 1.0, "Café Détroit naïve")],
        duration=1.0,
        model="medium",
        language="fr",
        device="cpu",
        speaker_id_used=False,
    )
    path = tmp_path / "out.json"
    write_json(str(path), payload)

    raw = path.read_text(encoding="utf-8")
    assert "Café Détroit naïve" in raw
    assert json.loads(raw)["segments"][0]["text"] == "Café Détroit naïve"
