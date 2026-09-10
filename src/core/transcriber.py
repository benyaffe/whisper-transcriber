"""
Qt wrapper around the transcription pipeline.

All the work lives in src/core/runner.py, which has no Qt in it. This module
is the adapter: it reads the saved settings, runs the pipeline on a QThread,
and turns observer callbacks into signals.

Names from runner are re-exported here because callers and tests import them
from this module, and because monkeypatching them as attributes of this module
is the established test idiom.
"""

import time
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from src.core.config import get_hf_token, is_speaker_id_enabled
from src.core.runner import (  # noqa: F401  (re-exported for callers and tests)
    LANGUAGE_CODES,
    MODEL_MEMORY_REQUIREMENTS,
    QualityMetrics,
    TranscriptionObserver,
    TranscriptionOutputs,
    TranscriptionRunner,
    TranscriptionSegment,
    WordTiming,
    check_memory_available,
    detect_diarization_device,
    detect_optimal_settings,
    format_vtt_time,
)
from src.utils.logger import get_logger, log_exception


class _SignalObserver(TranscriptionObserver):
    """Forwards runner callbacks to the worker's signals.

    Qt queues signal delivery across threads, so emitting from the worker
    thread is safe and slots run on the GUI thread.
    """

    def __init__(self, worker: "TranscriptionWorker"):
        self._w = worker

    def hardware(self, description):
        self._w.hardware_info.emit(description)

    def status(self, message):
        self._w.status_message.emit(message)

    def progress(self, percent, eta_seconds):
        self._w.progress.emit(percent, eta_seconds)

    def diarization_progress(self, percent, stage):
        self._w.diarization_progress.emit(percent, stage)

    def segment(self, start, end, text, speaker):
        self._w.segment_ready.emit(start, end, text, speaker)

    def language_detected(self, language, confidence):
        self._w.language_detected.emit(language, confidence)

    def quality_warning(self, message):
        self._w.quality_warning.emit(message)

    def audio_ready(self, path):
        self._w.audio_ready.emit(path)


class TranscriptionWorker(QThread):
    """Background worker for transcription with quality assessment and speaker ID."""

    # Signals
    progress = pyqtSignal(float, int)  # percent, eta_seconds
    diarization_progress = pyqtSignal(float, str)  # overall percent, stage label
    status_message = pyqtSignal(str)  # status updates (rendered as gray italic)
    segment_ready = pyqtSignal(float, float, str, str)  # start, end, text, speaker
    language_detected = pyqtSignal(str, float)  # language, confidence
    quality_warning = pyqtSignal(str)  # warning message
    hardware_info = pyqtSignal(str)  # hardware description
    audio_ready = pyqtSignal(str)  # audio path for playback
    completed = pyqtSignal(str, str, str, str)  # vtt_path, txt_path, json_path, audio_path
    error = pyqtSignal(str)  # error message

    def __init__(self, filepath: str, initial_model: str = "medium", language: Optional[str] = None):
        super().__init__()
        self.filepath = filepath
        self._logger = get_logger()
        # Settings are read here, in the Qt layer, and handed to the runner as
        # plain values. The runner never consults QSettings, which is what lets
        # a caller pin speaker ID on regardless of the Transcribe-mode checkbox.
        self.runner = TranscriptionRunner(
            filepath,
            model=initial_model,
            language=language,
            hf_token=get_hf_token(),
            enable_speaker_id=is_speaker_id_enabled(),
            observer=_SignalObserver(self),
        )

    # Read-through properties, so existing callers keep working.
    @property
    def segments(self):
        return self.runner.segments

    @property
    def audio_path(self):
        return self.runner.audio_path

    @property
    def model_size(self):
        return self.runner.model_size

    @property
    def language(self):
        return self.runner.language

    def run(self):
        try:
            outputs = self.runner.run()
        except Exception as e:
            if not self.runner.cancelled:
                import traceback

                log_exception(e, "transcription")
                self.status_message.emit(f"[Error: {type(e).__name__}: {e}]")
                for line in traceback.format_exc().rstrip().splitlines():
                    self.status_message.emit(f"[  {line}]")
                self.error.emit(f"{type(e).__name__}: {e}")
            return

        if outputs is None:
            return  # cancelled
        self.completed.emit(
            outputs.vtt_path, outputs.txt_path, outputs.json_path, outputs.audio_path
        )

    def cancel(self):
        """Cancel transcription."""
        self.runner.cancel()

    def seconds_since_last_segment(self) -> float:
        """How long the pipeline has been quiet.

        The runner's own stall check only fires when the segment generator
        yields, so it cannot notice faster-whisper wedged inside a blocking
        call. A caller with an event loop can poll this to spot that case.
        """
        return time.time() - self.runner.last_segment_time
