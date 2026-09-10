"""
Runs one trip: fetch anything remote, join the recordings, transcribe the result.

Three stages behind one progress bar. Reporting a trip as three consecutive
0-to-100% bars would be worse than useless, so each stage owns a slice of the
whole and the bar only ever moves forward.

Speaker identification and the model are decided by the trip, not by the saved
application settings. That is the whole reason TranscriptionRunner takes them as
arguments: a lecture can turn speakers off and a group recording can leave them
on, without either touching what the app remembers.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from src.core.runner import TranscriptionObserver, TranscriptionRunner
from src.podcastnotes.concat import ConcatError, concat, write_boundaries
from src.podcastnotes.project import TripProject
from src.utils.file_utils import is_url
from src.utils.logger import get_logger, log_exception

# Trips are pinned to these. The pickers that used to expose them are gone:
# a trip is about the content, not about tuning the transcriber.
TRIP_MODEL = "medium"
TRIP_LANGUAGE = "english"

# How the bar is divided. Downloading is skipped when nothing is remote and
# joining is skipped for a trip of one, in which case the earlier slices are
# simply passed through rather than left as a stall.
STAGE_SPANS = {
    "download": (0.0, 10.0),
    "join": (10.0, 15.0),
    "transcribe": (15.0, 100.0),
}


@dataclass
class TripOutputs:
    """Everything a finished trip produced."""

    work_dir: str
    audio_path: str
    vtt_path: str = ""
    txt_path: str = ""
    json_path: str = ""
    boundaries_path: str = ""
    speaker_count: int = 0
    sources: list = field(default_factory=list)


class TripWorker(QThread):
    """Drives one trip end to end on a background thread."""

    progress = pyqtSignal(float, str)   # overall percent, stage label
    status = pyqtSignal(str)            # free-text line for the transcript pane
    segment = pyqtSignal(float, float, str, str)  # start, end, text, speaker
    audio_ready = pyqtSignal(str)
    completed = pyqtSignal(object)      # TripOutputs
    failed = pyqtSignal(str)

    def __init__(self, trip: TripProject, identify_speakers: bool, hf_token: str = ""):
        super().__init__()
        self.trip = trip
        self.identify_speakers = identify_speakers
        self.hf_token = hf_token
        self._logger = get_logger()
        self._runner: Optional[TranscriptionRunner] = None
        self._cancelled = False

    # --- lifecycle ------------------------------------------------------------

    def cancel(self):
        """Stop as soon as the current stage allows."""
        self._cancelled = True
        if self._runner is not None:
            self._runner.cancel()

    def seconds_since_last_segment(self) -> float:
        """For the window's stall watch. Zero when no transcription is running."""
        import time

        if self._runner is None:
            return 0.0
        return time.time() - self._runner.last_segment_time

    def run(self):
        try:
            outputs = self._run_trip()
        except Exception as e:
            if not self._cancelled:
                log_exception(e, "trip")
                self.status.emit(f"[Error: {type(e).__name__}: {e}]")
                self.failed.emit(f"{type(e).__name__}: {e}")
            return

        if outputs is not None:
            self.completed.emit(outputs)

    # --- stages ---------------------------------------------------------------

    def _emit_stage_progress(self, stage: str, fraction: float, label: str):
        """Map a stage's own 0.0-1.0 onto its slice of the overall bar."""
        floor, ceiling = STAGE_SPANS[stage]
        self.progress.emit(floor + fraction * (ceiling - floor), label)

    def _run_trip(self) -> Optional[TripOutputs]:
        os.makedirs(self.trip.work_dir, exist_ok=True)
        self.trip.save()

        local_paths = self._fetch_remote_sources()
        if self._cancelled:
            return None

        audio_path, boundaries_path = self._join(local_paths)
        if self._cancelled:
            return None
        self.audio_ready.emit(audio_path)

        return self._transcribe(audio_path, boundaries_path)

    def _fetch_remote_sources(self) -> list[str]:
        """Download any URL sources into the working folder.

        Downloads are derived files, so they belong beside the rest of the
        trip's work rather than in ~/Downloads where the downloader would
        otherwise put them.
        """
        sources = [s.path for s in self.trip.sources]
        remote = [p for p in sources if is_url(p)]
        if not remote:
            self._emit_stage_progress("download", 1.0, "Ready")
            return sources

        from src.core.downloader import VideoDownloader

        resolved = []
        for index, source in enumerate(sources):
            if self._cancelled:
                return resolved
            if not is_url(source):
                resolved.append(source)
                continue

            self.status.emit(f"[Downloading {source}]")
            done = {}
            downloader = VideoDownloader(source, self.trip.work_dir)
            downloader.completed.connect(lambda p: done.update(path=p))
            downloader.error.connect(lambda e: done.update(error=e))
            # Called directly rather than started: this is already a background
            # thread, and the trip's stages are strictly sequential.
            downloader.run()

            if "error" in done:
                raise RuntimeError(f"Could not download {source}: {done['error']}")
            if "path" not in done:
                if self._cancelled:
                    return resolved
                raise RuntimeError(f"Download produced no file: {source}")

            resolved.append(done["path"])
            self.status.emit(f"[Downloaded {os.path.basename(done['path'])}]")
            self._emit_stage_progress(
                "download", (index + 1) / len(sources), "Downloading"
            )

        return resolved

    def _join(self, paths: list[str]) -> tuple[str, str]:
        """Join the recordings, or link a lone recording into the trip."""
        if len(paths) == 1:
            # Nothing to join, but the transcriber writes its outputs beside
            # its input, and a trip's outputs belong in the trip's folder
            # rather than scattered next to the original recording. A hard link
            # costs nothing and leaves the original untouched.
            self._emit_stage_progress("join", 1.0, "Ready")
            return self._link_into_trip(paths[0]), ""

        self._emit_stage_progress("join", 0.0, f"Joining {len(paths)} recordings")
        try:
            result = concat(paths, self.trip.combined_audio_path, on_status=self.status.emit)
        except ConcatError as e:
            raise RuntimeError(str(e)) from e

        boundaries_path = write_boundaries(result, self.trip.work_dir)
        self._emit_stage_progress("join", 1.0, "Joined")
        return result.output_path, boundaries_path

    def _link_into_trip(self, source: str) -> str:
        """Give the trip its own name for a single recording, without copying it.

        Falls back to a copy across volumes, where hard links are impossible.
        Either way the original is only ever read.
        """
        import shutil

        destination = self.trip.path_for(os.path.basename(source))
        if os.path.exists(destination):
            return destination
        try:
            os.link(source, destination)
        except OSError:
            shutil.copyfile(source, destination)
        return destination

    def _transcribe(self, audio_path: str, boundaries_path: str) -> Optional[TripOutputs]:
        worker = self

        class _Relay(TranscriptionObserver):
            def status(self, message):
                worker.status.emit(message)

            def segment(self, start, end, text, speaker):
                worker.segment.emit(start, end, text, speaker)

            def hardware(self, description):
                worker.status.emit(f"[{description}]")

            def quality_warning(self, message):
                worker.status.emit(f"[Warning: {message}]")

            def language_detected(self, language, confidence):
                worker.status.emit(f"[Detected: {language} ({confidence:.0%})]")

            def progress(self, percent, eta_seconds):
                label = "Transcribing"
                if eta_seconds > 0:
                    mins, secs = divmod(eta_seconds, 60)
                    label = f"Transcribing, about {mins}m {secs:02d}s left"
                worker._emit_stage_progress("transcribe", percent / 100.0, label)

            def diarization_progress(self, percent, stage):
                worker._emit_stage_progress("transcribe", percent / 100.0, stage)

        self._runner = TranscriptionRunner(
            audio_path,
            model=TRIP_MODEL,
            language=TRIP_LANGUAGE,
            hf_token=self.hf_token,
            enable_speaker_id=self.identify_speakers,
            observer=_Relay(),
        )
        result = self._runner.run()
        if result is None:
            return None  # cancelled

        speakers = {s.speaker for s in result.segments if s.speaker}
        self._emit_stage_progress("transcribe", 1.0, "Done")
        return TripOutputs(
            work_dir=self.trip.work_dir,
            audio_path=result.audio_path,
            vtt_path=result.vtt_path,
            txt_path=result.txt_path,
            json_path=result.json_path,
            boundaries_path=boundaries_path,
            speaker_count=len(speakers),
            sources=[s.name for s in self.trip.sources],
        )
