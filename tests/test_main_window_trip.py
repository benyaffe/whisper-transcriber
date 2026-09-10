"""
Tests for the window that now holds the trip flow.

Guards the wiring: a signal nobody connected is as silent as one nobody emits,
and that is exactly the kind of gap tests of the two screens in isolation
cannot see.

Run with: python -m pytest tests/test_main_window_trip.py -v
"""

import pytest


@pytest.fixture
def audio_files(tmp_path):
    paths = []
    for name in ("one.m4a", "two.m4a"):
        p = tmp_path / name
        p.write_bytes(b"pretend audio " * 100)
        paths.append(str(p))
    return paths


@pytest.fixture
def window(qt_app, monkeypatch):
    from src.ui.main_window import MainWindow
    from src.ui.podcastnotes.intake_view import IntakeView

    monkeypatch.setattr(IntakeView, "_hf_token", staticmethod(lambda: "hf_test"))
    w = MainWindow()
    yield w
    w.close()


class _FakeWorker:
    """A TripWorker that records what got connected and never runs anything."""

    SIGNALS = ("progress", "status", "segment", "audio_ready", "completed", "failed")

    def __init__(self, trip, identify_speakers, hf_token=""):
        self.trip = trip
        self.identify_speakers = identify_speakers
        self.hf_token = hf_token
        self.started = False
        self.cancelled = False
        for name in self.SIGNALS:
            setattr(self, name, _RecordingSignal())

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def wait(self, _ms):
        return True

    def isRunning(self):
        return self.started and not self.cancelled


class _RecordingSignal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)


@pytest.fixture
def fake_worker(monkeypatch):
    from src.ui import main_window as mod

    made = {}

    def build(trip, identify_speakers, hf_token=""):
        made["worker"] = _FakeWorker(trip, identify_speakers, hf_token)
        return made["worker"]

    monkeypatch.setattr(mod, "TripWorker", build)
    monkeypatch.setattr(mod, "check_ffmpeg_health", lambda: (True, ""))
    return made


# --- what opens ---------------------------------------------------------------


def test_the_app_opens_on_the_intake_screen(window):
    assert window.stack.currentWidget() is window.intake


def test_there_is_no_queue_any_more(window):
    """The old framing is gone: several recordings are one trip, not N jobs."""
    for gone in ("queue", "queue_list", "queue_files", "process_next", "retry_selected"):
        assert not hasattr(window, gone), f"{gone} survived"


# --- starting a trip ----------------------------------------------------------


def test_starting_a_trip_switches_to_the_run_screen(window, fake_worker, audio_files):
    window._start_trip("Ashford", "a tour", audio_files, True)

    assert window.stack.currentWidget() is window.run_view
    assert window.run_view.title.text() == "Ashford"


def test_starting_a_trip_connects_every_worker_signal(window, fake_worker, audio_files):
    """A correctly emitted signal nobody connected is still silent."""
    window._start_trip("Ashford", "", audio_files, True)
    worker = fake_worker["worker"]

    for name in _FakeWorker.SIGNALS:
        assert getattr(worker, name).slots, f"{name} was never connected"
    assert worker.started is True


def test_the_trip_carries_its_own_speaker_choice(window, fake_worker, audio_files):
    window._start_trip("Lecture", "", audio_files[:1], False)

    assert fake_worker["worker"].identify_speakers is False
    assert fake_worker["worker"].hf_token == "", "no token should be fetched when off"


def test_speakers_on_passes_the_token(window, fake_worker, audio_files, monkeypatch):
    from src.ui import main_window as mod

    monkeypatch.setattr(mod, "get_hf_token", lambda: "hf_real")

    window._start_trip("Ashford", "", audio_files, True)

    assert fake_worker["worker"].hf_token == "hf_real"


def test_the_description_reaches_the_trip(window, fake_worker, audio_files):
    window._start_trip("Ashford", "Ridgeline and Lakeside General", audio_files, True)

    assert fake_worker["worker"].trip.description == "Ridgeline and Lakeside General"


def test_a_broken_ffmpeg_stops_the_trip_before_it_starts(window, fake_worker, audio_files, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from src.ui import main_window as mod

    monkeypatch.setattr(mod, "check_ffmpeg_health", lambda: (False, "FFmpeg not found"))
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: None)

    window._start_trip("Ashford", "", audio_files, True)

    assert "worker" not in fake_worker
    assert window.stack.currentWidget() is window.intake


# --- finishing ----------------------------------------------------------------


def test_finishing_hands_the_outputs_to_the_run_screen(window, fake_worker, audio_files, tmp_path):
    from src.ui.podcastnotes.trip_worker import TripOutputs

    window._start_trip("Ashford", "", audio_files, True)
    outputs = TripOutputs(work_dir=str(tmp_path / "ashford"), audio_path="x", speaker_count=2)

    window._on_trip_finished(outputs)

    assert window.run_view.progress_bar.value() == 100
    assert "2 speakers" in window.run_view.summary.text()


def test_a_failure_is_explained_with_a_suggestion(window, fake_worker, audio_files):
    window._start_trip("Ashford", "", audio_files, True)

    window._on_trip_failed("No internet connection")

    shown = window.run_view.summary.text()
    assert "No internet connection" in shown
    # get_error_suggestion adds guidance for known failures.
    assert len(shown) > len("No internet connection")


def test_cancelling_stops_the_worker(window, fake_worker, audio_files):
    window._start_trip("Ashford", "", audio_files, True)

    window._cancel_trip()

    assert fake_worker["worker"].cancelled is True
    assert "Cancelled" in window.run_view.summary.text()


def test_new_trip_returns_to_intake(window, fake_worker, audio_files, tmp_path):
    from src.ui.podcastnotes.trip_worker import TripOutputs

    window._start_trip("Ashford", "", audio_files, True)
    window._on_trip_finished(TripOutputs(work_dir=str(tmp_path), audio_path="x"))

    window.run_view.new_trip_button.click()

    assert window.stack.currentWidget() is window.intake


# --- files from outside -------------------------------------------------------


def test_files_dropped_on_the_icon_join_the_trip(window, audio_files):
    """Previously these were silently discarded once the app was running."""
    window.open_files(audio_files)

    assert window.stack.currentWidget() is window.intake
    assert window.intake.recordings.sources() == audio_files


def test_files_arriving_mid_trip_are_refused_out_loud(window, fake_worker, audio_files, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    told = {}
    monkeypatch.setattr(QMessageBox, "information", lambda p, t, m, *a, **k: told.update(msg=m))
    window._start_trip("Ashford", "", audio_files, True)

    window.open_files(audio_files)

    assert "already running" in told["msg"]
    assert window.stack.currentWidget() is window.run_view
