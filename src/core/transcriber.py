"""
Transcription engine using faster-whisper.
Handles model loading, transcription, quality assessment, speaker diarization, and output.
"""

import os
import time
import platform
import psutil
from dataclasses import dataclass
from typing import Optional
from PyQt6.QtCore import QThread, pyqtSignal

from src.utils.file_utils import extract_audio, generate_output_paths, get_file_info
from src.core.diarization import run_diarization, assign_speakers_to_segments, DiarizationError, TokenValidationError
from src.ui.settings_dialog import get_hf_token, is_speaker_id_enabled
from src.utils.logger import get_logger, log_exception


# Memory requirements per model (approximate, in GB)
MODEL_MEMORY_REQUIREMENTS = {
    "tiny": 1.0,
    "base": 1.5,
    "small": 2.5,
    "medium": 5.0,
    "large": 10.0,
}


def check_memory_available(model_size: str, file_duration_minutes: float) -> tuple[bool, str]:
    """
    Check if sufficient memory is available for transcription.
    Returns (is_ok, warning_message).
    """
    try:
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)

        required_gb = MODEL_MEMORY_REQUIREMENTS.get(model_size, 5.0)
        # Add buffer for audio processing (roughly 0.1GB per 10 minutes)
        required_gb += file_duration_minutes * 0.01

        if available_gb < required_gb:
            return False, (
                f"Low memory: {available_gb:.1f}GB available, ~{required_gb:.1f}GB needed. "
                f"Try closing other apps or using a smaller model."
            )
        elif available_gb < required_gb * 1.5:
            return True, f"Memory is tight ({available_gb:.1f}GB available). Large files may be slow."
        else:
            return True, ""
    except Exception:
        return True, ""  # Don't block on errors


@dataclass
class TranscriptionSegment:
    """A segment of transcribed text with timing, confidence, and speaker."""
    start: float
    end: float
    text: str
    confidence: float
    speaker: Optional[str] = None


@dataclass
class QualityMetrics:
    """Quality metrics for transcription assessment."""
    avg_confidence: float
    low_confidence_ratio: float
    repetition_score: float


def detect_optimal_settings() -> tuple[str, str, str]:
    """
    Auto-detect hardware and return optimal device/compute settings.
    Returns (device, compute_type, description)
    """
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    # CUDA GPU
    if has_torch and torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        compute = "float16" if gpu_mem >= 8 else "int8"
        return "cuda", compute, f"CUDA GPU ({gpu_mem:.0f}GB)"

    # Apple Silicon
    if platform.system() == "Darwin" and platform.processor() == "arm":
        return "cpu", "int8", "Apple Silicon (optimized)"

    # Fallback
    return "cpu", "int8", "CPU"


# Language code mapping
LANGUAGE_CODES = {
    "english": "en", "spanish": "es", "french": "fr", "german": "de",
    "italian": "it", "portuguese": "pt", "dutch": "nl", "russian": "ru",
    "chinese": "zh", "japanese": "ja", "korean": "ko", "arabic": "ar",
    "hindi": "hi", "turkish": "tr", "polish": "pl", "ukrainian": "uk",
    "vietnamese": "vi", "thai": "th", "indonesian": "id", "malay": "ms",
    "swedish": "sv", "norwegian": "no", "danish": "da", "finnish": "fi",
    "greek": "el", "czech": "cs", "romanian": "ro", "hungarian": "hu",
    "hebrew": "he",
}


class TranscriptionWorker(QThread):
    """Background worker for transcription with quality assessment and speaker ID."""

    # Signals
    progress = pyqtSignal(float, int)  # percent, eta_seconds
    status_message = pyqtSignal(str)  # status updates (rendered as gray italic)
    segment_ready = pyqtSignal(float, float, str, str)  # start, end, text, speaker
    language_detected = pyqtSignal(str, float)  # language, confidence
    model_upgraded = pyqtSignal(str, str, str)  # old_model, new_model, reason
    quality_warning = pyqtSignal(str)  # warning message
    hardware_info = pyqtSignal(str)  # hardware description
    audio_ready = pyqtSignal(str)  # audio path for playback
    completed = pyqtSignal(str, str, str)  # vtt_path, txt_path, audio_path
    error = pyqtSignal(str)  # error message

    # Quality thresholds
    CONFIDENCE_THRESHOLD = 0.6
    LOW_CONFIDENCE_RATIO_THRESHOLD = 0.25
    ASSESSMENT_DURATION = 120  # seconds

    # Watchdog: timeout if no progress for this many seconds
    SEGMENT_TIMEOUT = 300  # 5 minutes

    def __init__(self, filepath: str, initial_model: str = "medium", language: Optional[str] = None):
        super().__init__()
        self.filepath = filepath
        self.model_size = initial_model
        self.language = LANGUAGE_CODES.get(language, language) if language else None
        self._cancelled = False
        self.segments: list[TranscriptionSegment] = []
        self._last_segment_time = time.time()
        self._logger = get_logger()
        self.audio_path: Optional[str] = None
        self._temp_audio = False

    def run(self):
        try:
            self._logger.info(f"Starting transcription: {self.filepath}")
            self._transcribe()
            self._logger.info("Transcription completed successfully")
        except Exception as e:
            if not self._cancelled:
                log_exception(e, "transcription")
                self.error.emit(str(e))

    def _transcribe(self):
        """Main transcription workflow."""
        from faster_whisper import WhisperModel

        # Track if speaker ID was successfully used (for VTT output)
        self._speaker_id_used = False

        # Step 1: Hardware detection
        device, compute_type, hw_desc = detect_optimal_settings()
        self.hardware_info.emit(f"Using {hw_desc}")
        self.status_message.emit(f"[Hardware: {hw_desc}]")
        self._logger.info(f"Hardware: {hw_desc}, device={device}, compute={compute_type}")

        # Step 2: Prepare audio
        self.status_message.emit("[Preparing audio...]")
        self.audio_path = self._prepare_audio()
        self.audio_ready.emit(self.audio_path)

        # Step 2.5: Memory check
        info = get_file_info(self.audio_path)
        duration_minutes = info.get('duration', 0) / 60
        mem_ok, mem_warning = check_memory_available(self.model_size, duration_minutes)
        if not mem_ok:
            self.quality_warning.emit(mem_warning)
            self._logger.warning(f"Memory warning: {mem_warning}")
        elif mem_warning:
            self.status_message.emit(f"[{mem_warning}]")

        # Step 3: Load model (check cancellation before slow operation)
        if self._cancelled:
            return
        self.status_message.emit(f"[Loading {self.model_size} model ({compute_type})...]")
        self._logger.info(f"Loading model: {self.model_size}")
        model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
        if self._cancelled:
            return

        # Step 4: Get duration
        info = get_file_info(self.audio_path)
        total_duration = info.get('duration', 0)

        # Step 5: Transcribe
        lang_info = f"language: {self.language}" if self.language else "auto-detecting language"
        self.status_message.emit(f"[Starting transcription ({lang_info})...]")

        segments_gen, trans_info = model.transcribe(
            self.audio_path,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            language=self.language
        )

        self.language_detected.emit(trans_info.language, trans_info.language_probability)

        # Process segments
        start_time = time.time()
        assessed = False

        for segment in segments_gen:
            if self._cancelled:
                return

            # Watchdog: check for timeout (no progress)
            now = time.time()
            if now - self._last_segment_time > self.SEGMENT_TIMEOUT:
                self._logger.error(f"Watchdog timeout: no segment for {self.SEGMENT_TIMEOUT}s")
                raise RuntimeError(
                    f"Transcription appears stuck (no progress for {self.SEGMENT_TIMEOUT // 60} minutes). "
                    "The file may be corrupted or incompatible."
                )
            self._last_segment_time = now

            # Calculate confidence
            word_confs = [w.probability for w in segment.words] if segment.words else []
            avg_conf = sum(word_confs) / len(word_confs) if word_confs else 0.8

            trans_seg = TranscriptionSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
                confidence=avg_conf,
                speaker=None
            )
            self.segments.append(trans_seg)
            self.segment_ready.emit(segment.start, segment.end, segment.text.strip(), "")

            # Progress
            if total_duration > 0:
                percent = (segment.end / total_duration) * 100
                elapsed = time.time() - start_time
                if segment.end > 0:
                    rate = elapsed / segment.end
                    remaining = (total_duration - segment.end) * rate
                    self.progress.emit(percent, int(remaining))
                else:
                    self.progress.emit(percent, 0)

            # Quality assessment at 2 minutes
            if not assessed and segment.end >= self.ASSESSMENT_DURATION:
                assessed = True
                quality = self._assess_quality()

                if (quality.avg_confidence < self.CONFIDENCE_THRESHOLD or
                        quality.low_confidence_ratio > self.LOW_CONFIDENCE_RATIO_THRESHOLD):

                    if self.model_size != "large":
                        old_model = self.model_size
                        self.model_size = "large"
                        self.model_upgraded.emit(
                            old_model, "large",
                            f"Low confidence ({quality.avg_confidence:.0%})"
                        )
                        self.segments = []
                        model = WhisperModel("large", device=device, compute_type=compute_type)
                        segments_gen, trans_info = model.transcribe(
                            self.audio_path,
                            beam_size=5,
                            word_timestamps=True,
                            vad_filter=True,
                            language=self.language
                        )
                        continue
                    else:
                        self.quality_warning.emit(
                            f"Quality issues detected ({quality.avg_confidence:.0%} confidence)"
                        )

        # Step 6: Speaker diarization
        if self._cancelled:
            return
        self.progress.emit(95, -1)
        self._run_diarization()

        # Step 7: Save outputs
        if self._cancelled:
            return
        vtt_path, txt_path = self._save_outputs()
        self.completed.emit(vtt_path, txt_path, self.audio_path)

    def _prepare_audio(self) -> str:
        """Extract audio from video if needed."""
        info = get_file_info(self.filepath)
        if info.get('has_video') and info.get('has_audio'):
            self._temp_audio = True
            return extract_audio(self.filepath)
        return self.filepath

    def _run_diarization(self):
        """Run speaker diarization and assign speakers to segments."""
        # Check if speaker ID is enabled
        if not is_speaker_id_enabled():
            self.status_message.emit("[Speaker ID: Disabled]")
            # Don't assign speakers - leave them as None for VTT output
            self._speaker_id_used = False
            return

        # Get HF token
        hf_token = get_hf_token()
        if not hf_token:
            self.status_message.emit("[Speaker ID: No token configured - see Settings]")
            self._speaker_id_used = False
            return

        # Status callback
        def status_cb(msg: str):
            self.status_message.emit(msg)

        try:
            self.status_message.emit("[Speaker ID: Starting...]")
            turns = run_diarization(
                self.audio_path,
                hf_token,
                status_callback=status_cb
            )

            if not turns:
                self.status_message.emit("[Speaker ID: No speakers detected]")
                self._speaker_id_used = False
                return

            speaker_map = assign_speakers_to_segments(self.segments, turns)
            self._speaker_id_used = True
            # Debug: verify speakers were assigned
            assigned = sum(1 for s in self.segments if s.speaker)
            self.status_message.emit(f"[Speaker ID complete: {len(speaker_map)} speaker(s), {assigned}/{len(self.segments)} segments tagged]")

        except TokenValidationError as e:
            self.status_message.emit(f"[Speaker ID: Token error - {str(e)[:40]}]")
            self._speaker_id_used = False

        except DiarizationError as e:
            self.status_message.emit(f"[Speaker ID failed: {str(e)[:40]}]")
            self._speaker_id_used = False

        except Exception as e:
            self.status_message.emit(f"[Speaker ID error: {str(e)[:40]}]")
            self._speaker_id_used = False

    def _assess_quality(self) -> QualityMetrics:
        """Assess transcription quality from collected segments."""
        if not self.segments:
            return QualityMetrics(0.8, 0.0, 0.0)

        confidences = [s.confidence for s in self.segments]
        avg_conf = sum(confidences) / len(confidences)
        low_count = sum(1 for c in confidences if c < self.CONFIDENCE_THRESHOLD)
        low_ratio = low_count / len(confidences)

        # Repetition check
        texts = [s.text.lower() for s in self.segments]
        reps = sum(1 for i in range(1, len(texts)) if texts[i] == texts[i-1] and len(texts[i]) > 10)
        rep_score = reps / len(texts) if texts else 0

        return QualityMetrics(avg_conf, low_ratio, rep_score)

    def _save_outputs(self) -> tuple[str, str]:
        """Save transcription to VTT and TXT with speaker labels."""
        vtt_path, txt_path = generate_output_paths(self.filepath)

        # Debug: log speaker ID state
        self._logger.info(f"Saving outputs: _speaker_id_used={self._speaker_id_used}, segments={len(self.segments)}")
        if self.segments:
            sample = self.segments[0]
            self._logger.info(f"First segment speaker: '{sample.speaker}', type: {type(sample.speaker)}")

        # VTT - only include speaker tags if speaker ID was successfully used
        with open(vtt_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            for i, seg in enumerate(self.segments, 1):
                start = self._format_vtt_time(seg.start)
                end = self._format_vtt_time(seg.end)
                f.write(f"{i}\n{start} --> {end}\n")
                if self._speaker_id_used and seg.speaker:
                    f.write(f"<v {seg.speaker}>{seg.text}</v>\n\n")
                else:
                    f.write(f"{seg.text}\n\n")

        # TXT - only include speaker labels if speaker ID was successfully used
        with open(txt_path, 'w', encoding='utf-8') as f:
            current_speaker = None
            for seg in self.segments:
                if self._speaker_id_used and seg.speaker and seg.speaker != current_speaker:
                    current_speaker = seg.speaker
                    f.write(f"\n{current_speaker}:\n")
                f.write(f"{seg.text} ")

        return vtt_path, txt_path

    def _format_vtt_time(self, seconds: float) -> str:
        """Format seconds as VTT timestamp."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    def cancel(self):
        """Cancel transcription."""
        self._cancelled = True
