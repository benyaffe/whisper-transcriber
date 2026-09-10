"""
Tests for the stall check in the transcription loop.

It is not a hang detector and the tests say so. It only runs when the segment
generator yields, so it reports a gap between two segments after the fact. A
genuinely wedged blocking call never reaches it.

What it *was* doing wrong: the clock started in __init__, so a slow model load
or a first-run model download counted against the timeout and killed the run
before the first segment was ever attempted.

Run with: python -m pytest tests/test_watchdog.py -v
"""

import time

import pytest

from src.core.transcriber import TranscriptionSegment, TranscriptionWorker


class _Signal:
    def __init__(self):
        self.emissions = []

    def emit(self, *args):
        self.emissions.append(args)

    def connect(self, slot):
        pass


class _NullLogger:
    def info(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def fake_segment(start, end):
    """Duck-types faster-whisper's Segment closely enough for the loop."""
    from types import SimpleNamespace

    return SimpleNamespace(start=start, end=end, text=" text", words=[])


def build_worker(monkeypatch, segments, *, model_load_seconds=0.0):
    """A worker wired to a scripted generator, with a controllable clock."""
    from src.core import transcriber as tmod

    clock = {"now": 1000.0}
    monkeypatch.setattr(tmod.time, "time", lambda: clock["now"])

    worker = TranscriptionWorker.__new__(TranscriptionWorker)
    worker.filepath = "/tmp/x.m4a"
    worker.model_size = "medium"
    worker.language = "en"
    worker._cancelled = False
    worker.segments = []
    worker._logger = _NullLogger()
    worker.audio_path = "/tmp/x.m4a"
    worker._temp_dir = None
    worker._will_diarize = False
    worker._progress_ceiling = 100.0
    worker._diarization = None
    worker._speaker_map = {}
    worker._speaker_id_used = False
    worker._last_segment_time = clock["now"]  # as __init__ would leave it

    for name in ("progress", "diarization_progress", "status_message", "segment_ready",
                 "language_detected", "quality_warning", "hardware_info", "audio_ready",
                 "completed", "error"):
        setattr(worker, name, _Signal())

    # Everything before the segment loop, collapsed into one slow step.
    monkeypatch.setattr(tmod, "detect_optimal_settings", lambda: ("cpu", "int8", "Test"))
    monkeypatch.setattr(tmod, "detect_diarization_device", lambda: "cpu")
    monkeypatch.setattr(tmod, "get_file_info", lambda p: {"duration": 600.0, "has_video": False, "has_audio": True})
    monkeypatch.setattr(tmod, "check_memory_available", lambda m, d: (True, ""))
    monkeypatch.setattr(worker, "_prepare_audio", lambda: "/tmp/x.m4a")
    monkeypatch.setattr(worker, "_run_diarization", lambda: None)
    monkeypatch.setattr(worker, "_save_outputs", lambda: ("v", "t", "j"))

    class FakeModel:
        def __init__(self, *a, **k):
            clock["now"] += model_load_seconds  # loading burns wall clock

        def transcribe(self, *a, **k):
            from types import SimpleNamespace
            return iter(segments), SimpleNamespace(language="en", language_probability=0.99)

    import faster_whisper
    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)

    return worker, clock


def test_slow_model_load_does_not_trip_the_stall_check(monkeypatch):
    """The regression. A first-run model download can take many minutes."""
    segments = [fake_segment(0.0, 5.0), fake_segment(5.0, 10.0)]
    worker, clock = build_worker(
        monkeypatch, segments,
        model_load_seconds=TranscriptionWorker.SEGMENT_TIMEOUT * 3,
    )

    worker._transcribe()  # must not raise

    assert len(worker.segments) == 2


def test_a_real_gap_between_segments_still_trips_it(monkeypatch):
    """The check has to keep doing the one thing it can actually do."""
    slow = TranscriptionWorker.SEGMENT_TIMEOUT + 60

    class StallingIterator:
        """Yields one segment, then burns more than the timeout before the next."""

        def __init__(self, clock):
            self.clock = clock
            self.n = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.n += 1
            if self.n > 2:
                raise StopIteration
            if self.n == 2:
                self.clock["now"] += slow
            return fake_segment((self.n - 1) * 5.0, self.n * 5.0)

    worker, clock = build_worker(monkeypatch, [])
    it = StallingIterator(clock)

    import faster_whisper
    from types import SimpleNamespace

    class FakeModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, *a, **k):
            return it, SimpleNamespace(language="en", language_probability=0.99)

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)

    with pytest.raises(RuntimeError, match="stalled"):
        worker._transcribe()


def test_normal_pacing_never_trips(monkeypatch):
    segments = [fake_segment(i * 5.0, (i + 1) * 5.0) for i in range(20)]
    worker, _ = build_worker(monkeypatch, segments)

    worker._transcribe()

    assert len(worker.segments) == 20


def test_the_clock_is_reset_at_the_segment_loop_not_in_init(monkeypatch):
    """Pin the actual fix, so it cannot quietly move back into __init__."""
    import ast
    import inspect
    import textwrap

    from src.core import transcriber

    source = textwrap.dedent(inspect.getsource(transcriber.TranscriptionWorker._transcribe))
    tree = ast.parse(source)

    assigns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Attribute) and t.attr == "_last_segment_time"
    ]
    assert assigns, "_transcribe no longer resets the stall clock before the loop"
