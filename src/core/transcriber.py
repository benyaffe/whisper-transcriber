"""
Transcription engine using faster-whisper.
Handles model selection, confidence analysis, and output generation.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional
from PyQt6.QtCore import QThread, pyqtSignal

from utils.file_utils import extract_audio, generate_output_paths, get_file_info


@dataclass
class TranscriptionSegment:
    """A segment of transcribed text with timing and confidence."""
    start: float
    end: float
    text: str
    confidence: float  # Average word confidence


@dataclass
class QualityMetrics:
    """Quality metrics for transcription assessment."""
    avg_confidence: float
    low_confidence_ratio: float  # Ratio of words below threshold
    repetition_score: float  # Higher = more repetition (bad)


class TranscriptionWorker(QThread):
    """
    Background worker for transcription.
    Handles model loading, transcription, quality assessment, and output.
    """

    # Signals
    progress = pyqtSignal(float, int)  # percent, eta_seconds
    text_chunk = pyqtSignal(str)  # New transcribed text
    language_detected = pyqtSignal(str, float)  # language, confidence
    model_upgraded = pyqtSignal(str, str, str)  # old_model, new_model, reason
    quality_warning = pyqtSignal(str)  # warning message
    completed = pyqtSignal(str, str)  # vtt_path, txt_path
    error = pyqtSignal(str)  # error message

    # Quality thresholds
    CONFIDENCE_THRESHOLD = 0.6  # Below this = low confidence
    LOW_CONFIDENCE_RATIO_THRESHOLD = 0.25  # More than 25% low confidence = upgrade
    ASSESSMENT_DURATION = 120  # Assess quality after 120 seconds of audio

    def __init__(self, filepath: str, initial_model: str = "medium"):
        super().__init__()
        self.filepath = filepath
        self.model_size = initial_model
        self._cancelled = False
        self.segments: list[TranscriptionSegment] = []
        self.audio_path: Optional[str] = None
        self._temp_audio = False

    def run(self):
        try:
            self._transcribe()
        except Exception as e:
            if not self._cancelled:
                self.error.emit(str(e))

    def _transcribe(self):
        """Main transcription workflow."""
        from faster_whisper import WhisperModel

        # Step 1: Prepare audio
        self.text_chunk.emit("Preparing audio...\n")
        self.audio_path = self._prepare_audio()

        # Step 2: Load model
        self.text_chunk.emit(f"Loading {self.model_size} model...\n")
        model = self._load_model(self.model_size)

        # Step 3: Get audio duration for progress calculation
        info = get_file_info(self.audio_path)
        total_duration = info.get('duration', 0)

        # Step 4: Transcribe
        self.text_chunk.emit("Starting transcription...\n\n")
        segments_generator, info = model.transcribe(
            self.audio_path,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True
        )

        # Emit detected language
        detected_lang = info.language
        lang_prob = info.language_probability
        self.language_detected.emit(detected_lang, lang_prob)

        # Process segments
        start_time = time.time()
        assessed = False

        for segment in segments_generator:
            if self._cancelled:
                return

            # Calculate confidence from word-level data
            word_confidences = []
            if segment.words:
                word_confidences = [w.probability for w in segment.words]

            avg_conf = sum(word_confidences) / len(word_confidences) if word_confidences else 0.8

            trans_segment = TranscriptionSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
                confidence=avg_conf
            )
            self.segments.append(trans_segment)

            # Emit text for preview
            self.text_chunk.emit(segment.text.strip() + " ")

            # Update progress
            if total_duration > 0:
                percent = (segment.end / total_duration) * 100
                elapsed = time.time() - start_time
                if segment.end > 0:
                    rate = elapsed / segment.end
                    remaining = (total_duration - segment.end) * rate
                    self.progress.emit(percent, int(remaining))
                else:
                    self.progress.emit(percent, 0)

            # Quality assessment after ASSESSMENT_DURATION
            if not assessed and segment.end >= self.ASSESSMENT_DURATION:
                assessed = True
                quality = self._assess_quality()

                if quality.avg_confidence < self.CONFIDENCE_THRESHOLD or \
                   quality.low_confidence_ratio > self.LOW_CONFIDENCE_RATIO_THRESHOLD:

                    if self.model_size != "large":
                        # Upgrade to large and restart
                        old_model = self.model_size
                        self.model_size = "large"
                        self.model_upgraded.emit(
                            old_model,
                            "large",
                            f"Low confidence ({quality.avg_confidence:.0%}), upgrading for better accuracy"
                        )

                        # Restart with large model
                        self.segments = []
                        model = self._load_model("large")
                        segments_generator, info = model.transcribe(
                            self.audio_path,
                            beam_size=5,
                            word_timestamps=True,
                            vad_filter=True
                        )
                        self.text_chunk.emit("\n\n[Restarting with large model...]\n\n")
                        assessed = True  # Don't re-assess
                        continue

                    else:
                        # Already on large, just warn
                        self.quality_warning.emit(
                            f"Quality issues detected (confidence: {quality.avg_confidence:.0%}). "
                            "Results may be incomplete. Consider checking audio quality."
                        )

        # Step 5: Generate output files
        vtt_path, txt_path = self._save_outputs()

        # Cleanup temp audio
        self._cleanup()

        self.completed.emit(vtt_path, txt_path)

    def _prepare_audio(self) -> str:
        """Extract audio from video if needed, return path to audio file."""
        info = get_file_info(self.filepath)

        # If it's video with audio, extract audio for faster processing
        if info.get('has_video') and info.get('has_audio'):
            self._temp_audio = True
            return extract_audio(self.filepath)

        # Already audio-only
        return self.filepath

    def _load_model(self, model_size: str):
        """Load Whisper model."""
        from faster_whisper import WhisperModel

        # Use CPU on Mac (or cuda if available on other platforms)
        # faster-whisper handles device selection
        compute_type = "int8"  # Good balance of speed/accuracy

        model = WhisperModel(
            model_size,
            device="auto",
            compute_type=compute_type
        )

        return model

    def _assess_quality(self) -> QualityMetrics:
        """Assess transcription quality from segments so far."""
        if not self.segments:
            return QualityMetrics(0.8, 0.0, 0.0)

        confidences = [s.confidence for s in self.segments]
        avg_conf = sum(confidences) / len(confidences)

        low_conf_count = sum(1 for c in confidences if c < self.CONFIDENCE_THRESHOLD)
        low_conf_ratio = low_conf_count / len(confidences)

        # Check for repetition (common hallucination sign)
        texts = [s.text.lower() for s in self.segments]
        repetition_score = self._calculate_repetition(texts)

        return QualityMetrics(
            avg_confidence=avg_conf,
            low_confidence_ratio=low_conf_ratio,
            repetition_score=repetition_score
        )

    def _calculate_repetition(self, texts: list[str]) -> float:
        """Calculate repetition score. Higher = more repetition."""
        if len(texts) < 3:
            return 0.0

        # Look for repeated consecutive segments
        repetitions = 0
        for i in range(1, len(texts)):
            if texts[i] == texts[i-1] and len(texts[i]) > 10:
                repetitions += 1

        return repetitions / len(texts)

    def _save_outputs(self) -> tuple[str, str]:
        """Save transcription to VTT and TXT files."""
        vtt_path, txt_path = generate_output_paths(self.filepath)

        # Write VTT
        with open(vtt_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")

            for i, seg in enumerate(self.segments, 1):
                start = self._format_timestamp(seg.start)
                end = self._format_timestamp(seg.end)
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{seg.text}\n\n")

        # Write TXT (clean prose)
        with open(txt_path, 'w', encoding='utf-8') as f:
            full_text = ' '.join(seg.text for seg in self.segments)
            # Basic sentence formatting
            full_text = full_text.replace('  ', ' ')
            f.write(full_text)

        return vtt_path, txt_path

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as VTT timestamp (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def _cleanup(self):
        """Clean up temporary files."""
        if self._temp_audio and self.audio_path and os.path.exists(self.audio_path):
            try:
                os.remove(self.audio_path)
            except OSError:
                pass

    def cancel(self):
        """Cancel the transcription."""
        self._cancelled = True
        self._cleanup()
