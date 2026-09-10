"""
Tests that the source-audio snapshot is cleaned up.

_snapshot_audio copies (or hardlinks) the source into a temp directory so a run
survives the file being moved underneath it. Nothing ever removed that
directory: 11 of them, 56 MB, were sitting in /var/folders when this was found,
and several had a link count of 1, meaning they were real copies rather than
cheap hardlinks.

Run with: python -m pytest tests/test_temp_cleanup.py -v
"""

import os

import pytest

from src.core.transcriber import TranscriptionWorker


def make_worker(tmp_path):
    """A worker with the state _snapshot_audio and cleanup touch, no QThread."""
    worker = TranscriptionWorker.__new__(TranscriptionWorker)
    worker._temp_dir = None
    worker._logger = _NullLogger()
    return worker


class _NullLogger:
    def info(self, *a, **k):
        pass


def test_snapshot_records_the_directory_it_created(tmp_path):
    source = tmp_path / "audio.m4a"
    source.write_bytes(b"x" * 2048)
    worker = make_worker(tmp_path)

    snapshot = worker._snapshot_audio(str(source))

    assert os.path.exists(snapshot)
    assert worker._temp_dir is not None
    assert os.path.basename(worker._temp_dir).startswith("wt_source_")
    assert os.path.dirname(snapshot) == worker._temp_dir


def test_cleanup_removes_the_directory(tmp_path):
    source = tmp_path / "audio.m4a"
    source.write_bytes(b"x" * 2048)
    worker = make_worker(tmp_path)
    snapshot = worker._snapshot_audio(str(source))
    temp_dir = worker._temp_dir

    worker._cleanup_temp_audio()

    assert not os.path.exists(temp_dir), "snapshot directory leaked"
    assert not os.path.exists(snapshot)
    assert worker._temp_dir is None
    # The original must survive; only the snapshot goes.
    assert source.exists()


def test_cleanup_is_idempotent(tmp_path):
    """run()'s finally can fire after an earlier cleanup; must not raise."""
    source = tmp_path / "audio.m4a"
    source.write_bytes(b"x" * 2048)
    worker = make_worker(tmp_path)
    worker._snapshot_audio(str(source))

    worker._cleanup_temp_audio()
    worker._cleanup_temp_audio()  # second call, nothing left to do


def test_cleanup_with_no_snapshot_is_a_noop(tmp_path):
    """The video path uses extract_audio and never snapshots."""
    worker = make_worker(tmp_path)

    worker._cleanup_temp_audio()

    assert worker._temp_dir is None


def test_cleanup_survives_a_vanished_directory(tmp_path):
    """Never let a cleanup failure take down a run that otherwise succeeded."""
    import shutil

    source = tmp_path / "audio.m4a"
    source.write_bytes(b"x" * 2048)
    worker = make_worker(tmp_path)
    worker._snapshot_audio(str(source))
    shutil.rmtree(worker._temp_dir)  # something else got there first

    worker._cleanup_temp_audio()

    assert worker._temp_dir is None


def test_run_cleans_up_even_when_transcription_raises(tmp_path, monkeypatch):
    """The finally in run() is the thing under test."""
    source = tmp_path / "audio.m4a"
    source.write_bytes(b"x" * 2048)

    worker = make_worker(tmp_path)
    worker.filepath = str(source)
    worker._cancelled = True  # suppresses the error signalling path
    snapshot_dir = {}

    def fake_transcribe():
        worker._snapshot_audio(str(source))
        snapshot_dir["path"] = worker._temp_dir
        raise RuntimeError("boom")

    monkeypatch.setattr(worker, "_transcribe", fake_transcribe)

    worker.run()

    assert not os.path.exists(snapshot_dir["path"]), "leaked on the failure path"


def test_run_cleans_up_on_success(tmp_path, monkeypatch):
    source = tmp_path / "audio.m4a"
    source.write_bytes(b"x" * 2048)

    worker = make_worker(tmp_path)
    worker.filepath = str(source)
    worker._cancelled = False
    snapshot_dir = {}

    def fake_transcribe():
        worker._snapshot_audio(str(source))
        snapshot_dir["path"] = worker._temp_dir

    monkeypatch.setattr(worker, "_transcribe", fake_transcribe)

    worker.run()

    assert not os.path.exists(snapshot_dir["path"]), "leaked on the success path"


# --- model stability ----------------------------------------------------------


def test_model_size_is_never_mutated_mid_run():
    """The auto-upgrade used to rebind self.model_size half way through a run.

    It never actually took effect, because it also rebound segments_gen inside
    `for segment in segments_gen`, and Python binds that iterator once. The net
    result was a discarded two minutes of transcript, a large model loaded and
    never used, and a UI claiming an upgrade that did not happen.

    Guarding it here because the JSON metadata and the diarization device
    selection both read model_size, and a mid-run change makes both wrong.
    """
    import ast
    import inspect
    import textwrap

    from src.core import transcriber

    tree = ast.parse(textwrap.dedent(inspect.getsource(transcriber.TranscriptionWorker)))

    # Real assignment statements only. A substring match would also catch
    # `check_memory_available(self.model_size, ...)`, which is a read.
    assigning_methods = []
    for method in ast.walk(tree):
        if not isinstance(method, ast.FunctionDef):
            continue
        for node in ast.walk(method):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            for t in targets:
                if (
                    isinstance(t, ast.Attribute)
                    and t.attr == "model_size"
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                ):
                    assigning_methods.append(method.name)

    assert assigning_methods == ["__init__"], (
        f"model_size is assigned outside __init__, in: {assigning_methods}"
    )


def test_low_confidence_warns_rather_than_switching_models():
    """The quality check still fires; it just no longer tries to act on it."""
    import inspect

    from src.core import transcriber

    source = inspect.getsource(transcriber.TranscriptionWorker._transcribe)

    assert "quality_warning.emit" in source
    assert "WhisperModel(\"large\"" not in source, "the dead upgrade path is back"
