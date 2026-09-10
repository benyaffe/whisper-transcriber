"""
Tests for pulling per-speaker voice embeddings out of pyannote's output.

These guard the Speaker Library's input. A bad vector stored here becomes a
voice profile that silently matches the wrong person on a later trip, which is
close to impossible to debug from the far end.

Run with: python -m pytest tests/test_speaker_embeddings.py -v
"""

import numpy as np
import pytest

from src.core.diarization import _extract_speaker_embeddings, run_diarization
from tests.conftest import FakeAnnotationWithLabels, FakeDiarizeOutput, make_embeddings

DIM = 256


def annotation(*speakers):
    """An annotation whose labels() are exactly the given speakers."""
    return FakeAnnotationWithLabels(
        [(float(i), float(i + 1), s) for i, s in enumerate(speakers)]
    )


def noop_log(_msg):
    pass


def test_healthy_rows_become_profiles():
    rows = make_embeddings(0.1, 0.2, dimension=DIM)

    embeddings, dimension, unusable = _extract_speaker_embeddings(
        rows, annotation("SPEAKER_00", "SPEAKER_01"), noop_log
    )

    assert dimension == DIM
    assert unusable == []
    assert set(embeddings) == {"SPEAKER_00", "SPEAKER_01"}
    assert len(embeddings["SPEAKER_00"]) == DIM
    assert embeddings["SPEAKER_00"][0] == pytest.approx(0.1)


def test_rows_are_paired_with_labels_in_sorted_order():
    """pyannote aligns embedding rows with labels(), which is a str sort."""
    rows = make_embeddings(0.1, 0.2, 0.3, dimension=DIM)

    # Deliberately construct the annotation out of order.
    ann = FakeAnnotationWithLabels(
        [(0.0, 1.0, "SPEAKER_02"), (1.0, 2.0, "SPEAKER_00"), (2.0, 3.0, "SPEAKER_01")]
    )
    embeddings, _, _ = _extract_speaker_embeddings(rows, ann, noop_log)

    assert embeddings["SPEAKER_00"][0] == pytest.approx(0.1)
    assert embeddings["SPEAKER_01"][0] == pytest.approx(0.2)
    assert embeddings["SPEAKER_02"][0] == pytest.approx(0.3)


def test_all_zero_row_is_rejected():
    """pyannote pads phantom speakers with zeros, then reorders.

    So a zero row can sit at any index. Cosine similarity against it is
    undefined, and storing it would poison the Speaker Library.
    """
    rows = make_embeddings(0.1, 0.0, 0.3, dimension=DIM)

    embeddings, _, unusable = _extract_speaker_embeddings(
        rows, annotation("SPEAKER_00", "SPEAKER_01", "SPEAKER_02"), noop_log
    )

    assert unusable == ["SPEAKER_01"]
    assert "SPEAKER_01" not in embeddings
    assert set(embeddings) == {"SPEAKER_00", "SPEAKER_02"}


def test_zero_row_is_rejected_wherever_it_lands():
    """The padded row is reordered, so position must not matter."""
    for zero_at in range(3):
        fills = [0.1, 0.2, 0.3]
        fills[zero_at] = 0.0
        labels = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]

        _, _, unusable = _extract_speaker_embeddings(
            make_embeddings(*fills, dimension=DIM), annotation(*labels), noop_log
        )
        assert unusable == [labels[zero_at]], f"missed a zero row at index {zero_at}"


def test_all_nan_row_is_rejected():
    """Reachable via pyannote's max_clusters<2 shortcut over an empty slice."""
    rows = make_embeddings(0.1, float("nan"), dimension=DIM)

    embeddings, _, unusable = _extract_speaker_embeddings(
        rows, annotation("SPEAKER_00", "SPEAKER_01"), noop_log
    )

    assert unusable == ["SPEAKER_01"]
    assert "SPEAKER_01" not in embeddings


def test_partially_nan_row_is_rejected():
    rows = make_embeddings(0.1, 0.2, dimension=DIM)
    rows[1][17] = np.nan

    _, _, unusable = _extract_speaker_embeddings(
        rows, annotation("SPEAKER_00", "SPEAKER_01"), noop_log
    )

    assert unusable == ["SPEAKER_01"]


def test_none_embeddings_is_not_an_error():
    """OracleClustering produces no centroids at all."""
    embeddings, dimension, unusable = _extract_speaker_embeddings(
        None, annotation("SPEAKER_00"), noop_log
    )

    assert (embeddings, dimension, unusable) == ({}, None, [])


def test_empty_embedding_array_is_not_an_error():
    """A file with no speech gives the documented (0, dim) shape."""
    embeddings, dimension, unusable = _extract_speaker_embeddings(
        np.zeros((0, DIM), dtype=np.float32), annotation(), noop_log
    )

    assert embeddings == {}
    assert dimension == DIM
    assert unusable == []


def test_more_labels_than_rows_pairs_what_lines_up():
    """Defensive: never raise, never invent a profile for the unpaired label."""
    rows = make_embeddings(0.1, dimension=DIM)

    embeddings, _, _ = _extract_speaker_embeddings(
        rows, annotation("SPEAKER_00", "SPEAKER_01"), noop_log
    )

    assert set(embeddings) == {"SPEAKER_00"}


def test_output_is_plain_floats_not_numpy():
    """np.float32 is not JSON-serializable, and this feeds a JSON export."""
    import json

    embeddings, _, _ = _extract_speaker_embeddings(
        make_embeddings(0.1, dimension=DIM), annotation("SPEAKER_00"), noop_log
    )

    assert all(type(x) is float for x in embeddings["SPEAKER_00"])
    json.dumps(embeddings)  # raises TypeError on numpy scalars


# --- through run_diarization --------------------------------------------------


def test_run_diarization_surfaces_embeddings(patched_diarization):
    patched_diarization.output = FakeDiarizeOutput(
        turns=[(0.0, 1.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")],
        embeddings=make_embeddings(0.1, 0.2, dimension=DIM),
    )

    result = run_diarization("/tmp/x.wav", "hf_token")

    assert result.embedding_dimension == DIM
    assert set(result.embeddings) == {"SPEAKER_00", "SPEAKER_01"}
    assert result.dropped_speakers == []


def test_run_diarization_reports_a_speaker_missing_from_exclusive_turns():
    """Someone always talked over never wins the argmax and vanishes.

    Segments get aligned against the exclusive turns, so losing a speaker there
    silently is exactly the failure worth surfacing.
    """
    from types import SimpleNamespace
    from unittest.mock import patch

    import pyannote.audio
    from src.core import diarization
    from tests.conftest import FakePipeline

    pipeline = FakePipeline(
        output=FakeDiarizeOutput(
            turns=[(0.0, 1.0, "SPEAKER_00"), (0.2, 0.8, "SPEAKER_01")],
            exclusive_turns=[(0.0, 1.0, "SPEAKER_00")],  # 01 always overlapped
            embeddings=make_embeddings(0.1, 0.2, dimension=DIM),
        )
    )

    with patch.object(diarization, "validate_hf_token", lambda t: (True, "ok")), \
         patch.object(diarization, "_ensure_models_downloaded", lambda t, log: None), \
         patch.object(
             diarization, "_decode_audio_to_tensor",
             lambda p, sample_rate=16000: SimpleNamespace(shape=(1, sample_rate)),
         ), \
         patch.object(
             pyannote.audio, "Pipeline",
             SimpleNamespace(from_pretrained=lambda *a, **k: pipeline),
         ):
        result = run_diarization("/tmp/x.wav", "hf_token")

    assert {t.speaker for t in result.turns} == {"SPEAKER_00", "SPEAKER_01"}
    assert {t.speaker for t in result.exclusive_turns} == {"SPEAKER_00"}
    assert "SPEAKER_01" in result.dropped_speakers
    # The embedding is still usable even though the speaker lost every frame.
    assert "SPEAKER_01" in result.embeddings
