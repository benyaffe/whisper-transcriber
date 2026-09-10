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

from src.core.runner import TranscriptionRunner


def fake_segment(start, end):
    """Duck-types faster-whisper's Segment closely enough for the loop."""
    from types import SimpleNamespace

    return SimpleNamespace(start=start, end=end, text=" text", words=[])


def build_worker(monkeypatch, segments, *, model_load_seconds=0.0):
    """A runner wired to a scripted generator, with a controllable clock.

    This used to hand-populate fourteen attributes on a
    TranscriptionWorker.__new__ instance and stub ten pyqtSignals, purely
    because QThread.__init__ could not be called in a test. The runner has a
    normal constructor and a no-op observer, so all of that is gone.
    """
    from src.core import runner as tmod

    clock = {"now": 1000.0}
    monkeypatch.setattr(tmod.time, "time", lambda: clock["now"])

    worker = TranscriptionRunner("/tmp/x.m4a", model="medium", language="english")

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
        model_load_seconds=TranscriptionRunner.SEGMENT_TIMEOUT * 3,
    )

    worker._transcribe()  # must not raise

    assert len(worker.segments) == 2


def test_a_real_gap_between_segments_still_trips_it(monkeypatch):
    """The check has to keep doing the one thing it can actually do."""
    slow = TranscriptionRunner.SEGMENT_TIMEOUT + 60

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
    """Pin the actual fix, so it cannot quietly move back into the constructor."""
    import ast
    import inspect
    import textwrap

    from src.core import runner

    source = textwrap.dedent(inspect.getsource(runner.TranscriptionRunner._transcribe))
    tree = ast.parse(source)

    assigns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Attribute) and t.attr == "_last_segment_time"
    ]
    assert assigns, "_transcribe no longer resets the stall clock before the loop"


# --- headless operation -------------------------------------------------------


def test_runner_works_with_no_qapplication(monkeypatch):
    """The path PodcastNotesWT depends on: a real run, no Qt object anywhere.

    The observer here is a plain class, not a QObject, and nothing constructs a
    QApplication. Before the split this was impossible: the pipeline lived on a
    QThread.
    """
    from src.core.runner import TranscriptionObserver, TranscriptionRunner

    seen = {"segments": [], "progress": [], "status": []}

    class Recorder(TranscriptionObserver):
        def segment(self, start, end, text, speaker):
            seen["segments"].append((start, end, text))

        def progress(self, percent, eta_seconds):
            seen["progress"].append(percent)

        def status(self, message):
            seen["status"].append(message)

    segments = [fake_segment(i * 5.0, (i + 1) * 5.0) for i in range(4)]
    worker, _ = build_worker(monkeypatch, segments)
    worker.observer = Recorder()

    worker._transcribe()

    assert len(seen["segments"]) == 4
    assert seen["progress"] == sorted(seen["progress"])
    assert any("Hardware" in m for m in seen["status"])


def test_speaker_id_is_driven_by_arguments_not_saved_settings(monkeypatch):
    """Two runners, same machine, opposite speaker-ID decisions."""
    from src.core.runner import TranscriptionRunner

    on = TranscriptionRunner("/tmp/x.wav", hf_token="hf_x", enable_speaker_id=True)
    off = TranscriptionRunner("/tmp/x.wav", hf_token="hf_x", enable_speaker_id=False)
    no_token = TranscriptionRunner("/tmp/x.wav", hf_token="", enable_speaker_id=True)

    assert (on.enable_speaker_id, bool(on.hf_token)) == (True, True)
    assert off.enable_speaker_id is False
    assert bool(no_token.hf_token) is False


# --- GUI-side stall warning ---------------------------------------------------
#
# The runner's own check cannot fire while faster-whisper is wedged inside a
# blocking call, because the loop body is never reached. The window polls from
# the GUI event loop, which keeps running when the worker thread does not.


class _FakeWorker:
    def __init__(self, quiet_for, running=True):
        self._quiet = quiet_for
        self._running = running

    def seconds_since_last_segment(self):
        return self._quiet

    def isRunning(self):
        return self._running


def make_window(qt_app):
    from src.ui.main_window import MainWindow

    return MainWindow()


def test_no_warning_while_segments_are_arriving(qt_app):
    window = make_window(qt_app)
    try:
        window.transcription_worker = _FakeWorker(quiet_for=5)
        window._stall_warned = False

        window._check_for_stall()

        assert "No new speech" not in window.preview_text.toPlainText()
    finally:
        window.close()


def test_warns_once_when_the_pipeline_goes_quiet(qt_app):
    from src.ui.main_window import MainWindow

    window = make_window(qt_app)
    try:
        window.transcription_worker = _FakeWorker(
            quiet_for=MainWindow.STALL_WARN_AFTER_S + 60
        )
        window._stall_warned = False

        window._check_for_stall()
        window._check_for_stall()  # still stalled; must not spam
        window._check_for_stall()

        shown = window.preview_text.toPlainText()
        assert shown.count("No new speech") == 1
        assert "Still working" in shown
    finally:
        window.close()


def test_warning_rearms_after_recovery(qt_app):
    from src.ui.main_window import MainWindow

    window = make_window(qt_app)
    try:
        stalled = _FakeWorker(quiet_for=MainWindow.STALL_WARN_AFTER_S + 60)
        window.transcription_worker = stalled
        window._stall_warned = False
        window._check_for_stall()

        stalled._quiet = 1  # segments start flowing again
        window._check_for_stall()
        stalled._quiet = MainWindow.STALL_WARN_AFTER_S + 60  # and stall again
        window._check_for_stall()

        assert window.preview_text.toPlainText().count("No new speech") == 2
    finally:
        window.close()


def test_watch_stops_itself_when_the_worker_finishes(qt_app):
    window = make_window(qt_app)
    try:
        window._start_stall_watch()
        assert window._stall_timer.isActive()

        window.transcription_worker = _FakeWorker(quiet_for=0, running=False)
        window._check_for_stall()

        assert not window._stall_timer.isActive()
    finally:
        window.close()
