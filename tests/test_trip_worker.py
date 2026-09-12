"""
Tests for the worker that runs a trip: fetch, join, transcribe.

The transcription itself is stubbed. What is under test is the orchestration:
that stages run in order, that the bar spans the whole run rather than
restarting three times, that a trip's settings beat the app's saved ones, and
that a trip's files land in the trip's folder.

Run with: python -m pytest tests/test_trip_worker.py -v
"""

import os

import pytest

from src.podcastnotes.project import REMOTE_TRIPS_DIR, SourceRecording, TripProject
from src.ui.podcastnotes.trip_worker import (
    STAGE_SPANS,
    TRIP_LANGUAGE,
    TRIP_MODEL,
    TripOutputs,
    TripWorker,
)


@pytest.fixture
def recordings(tmp_path):
    folder = tmp_path / "Audio Files"
    folder.mkdir()
    paths = []
    for name in ("one.m4a", "two.m4a"):
        p = folder / name
        p.write_bytes(b"pretend audio " + name.encode())
        paths.append(str(p))
    return paths


def make_worker(trip, monkeypatch, *, identify_speakers=True, hf_token="hf_x"):
    """A worker with transcription stubbed and the calls recorded."""
    from src.ui.podcastnotes import trip_worker as mod

    seen = {"runner_kwargs": None, "concat": None, "progress": [], "status": []}

    class FakeOutputs:
        audio_path = "/tmp/audio.m4a"
        vtt_path = "/tmp/out.vtt"
        txt_path = "/tmp/out.txt"
        json_path = "/tmp/out.json"
        segments = []

    class FakeRunner:
        def __init__(self, path, **kwargs):
            seen["runner_kwargs"] = dict(kwargs, path=path)
            self.last_segment_time = 0.0

        def run(self):
            return FakeOutputs()

        def cancel(self):
            seen["cancelled"] = True

    monkeypatch.setattr(mod, "TranscriptionRunner", FakeRunner)

    worker = TripWorker(trip, identify_speakers=identify_speakers, hf_token=hf_token)
    worker.progress.connect(lambda pct, label: seen["progress"].append((pct, label)))
    worker.status.connect(seen["status"].append)
    return worker, seen


# --- what the trip decides, not the app ---------------------------------------


def test_speaker_id_comes_from_the_trip_not_the_saved_setting(recordings, monkeypatch):
    """The reason TranscriptionRunner takes these as arguments.

    A lecture turns speakers off and a group recording leaves them on, and
    neither touches what the application remembers.
    """
    trip = TripProject.create("Lecture", recordings[:1])
    worker, seen = make_worker(trip, monkeypatch, identify_speakers=False)

    worker._run_trip()

    assert seen["runner_kwargs"]["enable_speaker_id"] is False


def test_speaker_id_on_is_passed_through_with_the_token(recordings, monkeypatch):
    trip = TripProject.create("Ashford", recordings[:1])
    worker, seen = make_worker(trip, monkeypatch, identify_speakers=True, hf_token="hf_abc")

    worker._run_trip()

    assert seen["runner_kwargs"]["enable_speaker_id"] is True
    assert seen["runner_kwargs"]["hf_token"] == "hf_abc"


def test_model_and_language_are_pinned(recordings, monkeypatch):
    """The pickers are gone; a trip is about content, not transcriber tuning."""
    trip = TripProject.create("Ashford", recordings[:1])
    worker, seen = make_worker(trip, monkeypatch)

    worker._run_trip()

    assert seen["runner_kwargs"]["model"] == TRIP_MODEL == "medium"
    assert seen["runner_kwargs"]["language"] == TRIP_LANGUAGE == "english"


# --- one bar for the whole trip -----------------------------------------------


def test_progress_spans_the_whole_run_and_never_goes_backwards(recordings, monkeypatch):
    """Three consecutive 0-to-100% bars would be worse than one honest one."""
    trip = TripProject.create("Ashford", recordings[:1])
    worker, seen = make_worker(trip, monkeypatch)

    worker._run_trip()

    percents = [p for p, _ in seen["progress"]]
    assert percents == sorted(percents), f"progress went backwards: {percents}"
    assert percents[-1] == 100.0
    assert all(0.0 <= p <= 100.0 for p in percents)


def test_each_stage_stays_inside_its_own_slice():
    """Guards the arithmetic rather than the run."""
    assert STAGE_SPANS["download"][0] == 0.0
    assert STAGE_SPANS["transcribe"][1] == 100.0

    ordered = sorted(STAGE_SPANS.values())
    for (_, earlier_end), (later_start, _) in zip(ordered, ordered[1:]):
        assert earlier_end == later_start, "a gap or overlap between stages"


def test_transcription_gets_the_largest_share():
    """It is by far the longest stage, so it should own most of the bar."""
    floor, ceiling = STAGE_SPANS["transcribe"]
    assert ceiling - floor >= 80.0


# --- where the files go -------------------------------------------------------


def test_a_single_recording_is_linked_into_the_trip_folder(recordings, monkeypatch):
    """Otherwise the transcript lands beside the original, not with the trip."""
    trip = TripProject.create("Solo", recordings[:1])
    worker, _ = make_worker(trip, monkeypatch)
    os.makedirs(trip.work_dir, exist_ok=True)

    audio, boundaries = worker._join([recordings[0]])

    assert os.path.dirname(audio) == trip.work_dir
    assert os.path.basename(audio) == "one.m4a"
    assert boundaries == ""
    # The original is still there, and still the same file.
    assert os.path.exists(recordings[0])
    assert os.path.samefile(audio, recordings[0]), "should be a link, not a copy"


def test_linking_twice_is_harmless(recordings, monkeypatch):
    trip = TripProject.create("Solo", recordings[:1])
    worker, _ = make_worker(trip, monkeypatch)
    os.makedirs(trip.work_dir, exist_ok=True)

    first = worker._link_into_trip(recordings[0])
    second = worker._link_into_trip(recordings[0])

    assert first == second


def test_several_recordings_are_joined(recordings, monkeypatch):
    from src.ui.podcastnotes import trip_worker as mod

    trip = TripProject.create("Ashford", recordings)
    worker, _ = make_worker(trip, monkeypatch)
    os.makedirs(trip.work_dir, exist_ok=True)

    class FakeResult:
        output_path = trip.combined_audio_path
        boundaries = []
        stream_copied = True
        drift = 0.0
        reencode_reason = ""

    joined = {}
    monkeypatch.setattr(mod, "concat", lambda paths, out, on_status=None: (
        joined.update(paths=paths, out=out), FakeResult())[1])
    monkeypatch.setattr(mod, "write_boundaries", lambda r, d: os.path.join(d, "boundaries.json"))

    audio, boundaries = worker._join(recordings)

    assert joined["paths"] == recordings
    assert audio == trip.combined_audio_path
    assert boundaries.endswith("boundaries.json")


# --- URL sources --------------------------------------------------------------


def test_a_url_source_is_not_mangled_by_abspath():
    """os.path.abspath("https://x") gives "<cwd>/https:/x", which is nonsense."""
    source = SourceRecording.from_path("https://youtube.com/watch?v=abc123")

    assert source.path == "https://youtube.com/watch?v=abc123"
    assert source.remote is True
    assert source.duration == 0.0


def test_a_trip_of_only_urls_lives_somewhere_sensible():
    trip = TripProject.create("Webinar", ["https://youtube.com/watch?v=abc"])

    assert trip.work_dir.startswith(REMOTE_TRIPS_DIR)


def test_a_url_alongside_a_local_file_uses_the_local_folder(recordings):
    trip = TripProject.create(
        "Mixed", [recordings[0], "https://youtube.com/watch?v=abc"]
    )

    assert trip.work_dir.startswith(os.path.dirname(recordings[0]))


def test_downloads_land_in_the_trip_folder_not_downloads(recordings, monkeypatch):
    """VideoDownloader defaults to ~/Downloads; a trip's fetches are its own."""
    from src.core import downloader as dl

    trip = TripProject.create("Webinar", [recordings[0], "https://example.com/talk"])
    worker, seen = make_worker(trip, monkeypatch)
    os.makedirs(trip.work_dir, exist_ok=True)

    asked = {}

    class FakeDownloader:
        def __init__(self, url, output_dir):
            asked["url"] = url
            asked["output_dir"] = output_dir
            self.completed = _Sig()
            self.error = _Sig()

        def run(self):
            self.completed.fire(os.path.join(asked["output_dir"], "talk.mp4"))

    monkeypatch.setattr(dl, "VideoDownloader", FakeDownloader)

    resolved = worker._fetch_remote_sources()

    assert asked["output_dir"] == trip.work_dir
    assert resolved[0] == recordings[0]
    assert resolved[1].endswith("talk.mp4")


def test_a_failed_download_stops_the_trip(recordings, monkeypatch):
    from src.core import downloader as dl

    trip = TripProject.create("Webinar", ["https://example.com/gone"])
    worker, _ = make_worker(trip, monkeypatch)

    class FakeDownloader:
        def __init__(self, url, output_dir):
            self.completed = _Sig()
            self.error = _Sig()

        def run(self):
            self.error.fire("Video unavailable")

    monkeypatch.setattr(dl, "VideoDownloader", FakeDownloader)

    with pytest.raises(RuntimeError, match="Video unavailable"):
        worker._fetch_remote_sources()


def test_no_urls_means_no_downloader_at_all(recordings, monkeypatch):
    trip = TripProject.create("Ashford", recordings)
    worker, _ = make_worker(trip, monkeypatch)

    resolved = worker._fetch_remote_sources()

    assert resolved == recordings


class _Sig:
    """Enough of a pyqtSignal for the fake downloader."""

    def __init__(self):
        self._slots = []

    def connect(self, slot):
        self._slots.append(slot)

    def fire(self, value):
        for slot in self._slots:
            slot(value)


# --- cancellation -------------------------------------------------------------


def test_cancel_reaches_the_transcriber(recordings, monkeypatch):
    trip = TripProject.create("Ashford", recordings[:1])
    worker, seen = make_worker(trip, monkeypatch)
    worker._run_trip()

    worker.cancel()

    assert worker._cancelled is True
    assert seen.get("cancelled") is True


def test_cancel_before_any_stage_is_safe(recordings, monkeypatch):
    trip = TripProject.create("Ashford", recordings[:1])
    worker, _ = make_worker(trip, monkeypatch)

    worker.cancel()  # no runner exists yet

    assert worker.seconds_since_last_segment() == 0.0


def test_cancelling_stops_between_stages(recordings, monkeypatch):
    trip = TripProject.create("Ashford", recordings[:1])
    worker, seen = make_worker(trip, monkeypatch)
    worker._cancelled = True

    assert worker._run_trip() is None
    assert seen["runner_kwargs"] is None, "transcription started despite cancellation"


# --- how long is left ----------------------------------------------------------


def test_the_estimate_is_in_whole_minutes():
    """The estimate comes from throughput measured so far and swings by a
    minute between updates, so seconds claim a precision it does not have and
    watching them jump reads as the app changing its mind twice a second."""
    from src.ui.podcastnotes.trip_worker import remaining

    assert remaining(785) == "Transcribing, about 14 minutes left"
    assert "s left" not in remaining(785).replace("minutes left", "")


def test_the_last_stretch_does_not_count_down_from_fifty_nine():
    from src.ui.podcastnotes.trip_worker import remaining

    assert remaining(40) == "Transcribing, less than a minute left"


def test_one_minute_is_singular():
    from src.ui.podcastnotes.trip_worker import remaining

    assert remaining(60) == "Transcribing, about 1 minute left"


def test_it_rounds_up_so_it_never_stalls_on_one_minute():
    """Rounding down means saying "1 minute left" and then running for another
    two, which is the one thing an estimate must not do."""
    from src.ui.podcastnotes.trip_worker import remaining

    assert remaining(61) == "Transcribing, about 2 minutes left"
    assert remaining(119) == "Transcribing, about 2 minutes left"


def test_no_estimate_yet_just_says_what_it_is_doing():
    from src.ui.podcastnotes.trip_worker import remaining

    assert remaining(0) == "Transcribing"
    assert remaining(-1) == "Transcribing"
