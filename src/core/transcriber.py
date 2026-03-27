"""
Transcription engine using faster-whisper.
Handles model selection, confidence analysis, speaker diarization, and output generation.
"""

import os
import time
import platform
from dataclasses import dataclass, field
from typing import Optional
from PyQt6.QtCore import QThread, pyqtSignal

from src.utils.file_utils import extract_audio, generate_output_paths, get_file_info


@dataclass
class TranscriptionSegment:
    """A segment of transcribed text with timing, confidence, and speaker."""
    start: float
    end: float
    text: str
    confidence: float
    speaker: Optional[str] = None  # Speaker label (e.g., "Speaker 1")


@dataclass
class QualityMetrics:
    """Quality metrics for transcription assessment."""
    avg_confidence: float
    low_confidence_ratio: float
    repetition_score: float


def detect_optimal_settings():
    """
    Auto-detect hardware and return optimal device/compute settings.
    Returns (device, compute_type, description)
    """
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    # Check for Apple Silicon
    if platform.system() == "Darwin" and platform.processor() == "arm":
        if has_torch and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Apple Silicon with MPS - use float16 for speed
            return "auto", "float16", "Apple Silicon (Metal)"
        else:
            # Fallback for bundled app without torch - use int8 on CPU
            return "cpu", "int8", "Apple Silicon (CPU)"

    # Check for CUDA
    if has_torch and torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_mem >= 8:
            return "cuda", "float16", f"CUDA GPU ({gpu_mem:.0f}GB)"
        else:
            return "cuda", "int8", f"CUDA GPU ({gpu_mem:.0f}GB, using int8)"

    # Fallback to CPU
    return "cpu", "int8", "CPU"


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
    hardware_info = pyqtSignal(str)  # Hardware acceleration info
    completed = pyqtSignal(str, str, str)  # vtt_path, txt_path, audio_path
    error = pyqtSignal(str)  # error message

    # Quality thresholds
    CONFIDENCE_THRESHOLD = 0.6
    LOW_CONFIDENCE_RATIO_THRESHOLD = 0.25
    ASSESSMENT_DURATION = 120

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

    def __init__(self, filepath: str, initial_model: str = "medium", language: Optional[str] = None):
        super().__init__()
        self.filepath = filepath
        self.model_size = initial_model
        self.language = self.LANGUAGE_CODES.get(language, language) if language else None
        self._cancelled = False
        self.segments: list[TranscriptionSegment] = []
        self.audio_path: Optional[str] = None
        self._temp_audio = False
        self._keep_audio = False  # Keep audio for playback

    def run(self):
        try:
            self._transcribe()
        except Exception as e:
            if not self._cancelled:
                self.error.emit(str(e))

    def _transcribe(self):
        """Main transcription workflow."""
        from faster_whisper import WhisperModel

        # Step 1: Detect optimal hardware settings
        device, compute_type, hw_desc = detect_optimal_settings()
        self.hardware_info.emit(f"Using {hw_desc}")
        self.text_chunk.emit(f"[Hardware: {hw_desc}]\n")

        # Step 2: Prepare audio (keep for playback)
        self.text_chunk.emit("Preparing audio...\n")
        self.audio_path = self._prepare_audio()
        self._keep_audio = True  # Keep for timestamp playback

        # Step 3: Load model with optimal settings
        self.text_chunk.emit(f"Loading {self.model_size} model ({compute_type})...\n")
        model = self._load_model(self.model_size, device, compute_type)

        # Step 4: Get audio duration
        info = get_file_info(self.audio_path)
        total_duration = info.get('duration', 0)

        # Step 5: Transcribe
        lang_info = f" (language: {self.language})" if self.language else " (auto-detecting language)"
        self.text_chunk.emit(f"Starting transcription{lang_info}...\n\n")

        segments_generator, info = model.transcribe(
            self.audio_path,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            language=self.language
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

            word_confidences = []
            if segment.words:
                word_confidences = [w.probability for w in segment.words]

            avg_conf = sum(word_confidences) / len(word_confidences) if word_confidences else 0.8

            trans_segment = TranscriptionSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
                confidence=avg_conf,
                speaker=None  # Will be filled by diarization
            )
            self.segments.append(trans_segment)

            self.text_chunk.emit(segment.text.strip() + " ")

            if total_duration > 0:
                percent = (segment.end / total_duration) * 100
                elapsed = time.time() - start_time
                if segment.end > 0:
                    rate = elapsed / segment.end
                    remaining = (total_duration - segment.end) * rate
                    self.progress.emit(percent, int(remaining))
                else:
                    self.progress.emit(percent, 0)

            # Quality assessment
            if not assessed and segment.end >= self.ASSESSMENT_DURATION:
                assessed = True
                quality = self._assess_quality()

                if quality.avg_confidence < self.CONFIDENCE_THRESHOLD or \
                   quality.low_confidence_ratio > self.LOW_CONFIDENCE_RATIO_THRESHOLD:

                    if self.model_size != "large":
                        old_model = self.model_size
                        self.model_size = "large"
                        self.model_upgraded.emit(
                            old_model,
                            "large",
                            f"Low confidence ({quality.avg_confidence:.0%}), upgrading for better accuracy"
                        )

                        self.segments = []
                        model = self._load_model("large", device, compute_type)
                        segments_generator, info = model.transcribe(
                            self.audio_path,
                            beam_size=5,
                            word_timestamps=True,
                            vad_filter=True,
                            language=self.language
                        )
                        self.text_chunk.emit("\n\n[Restarting with large model...]\n\n")
                        assessed = True
                        continue
                    else:
                        self.quality_warning.emit(
                            f"Quality issues detected (confidence: {quality.avg_confidence:.0%}). "
                            "Results may be incomplete."
                        )

        # Step 6: Speaker diarization
        self.text_chunk.emit("\n\n[Identifying speakers...]\n")
        self.progress.emit(95, 0)
        self._run_diarization()

        # Step 7: Generate output files
        vtt_path, txt_path = self._save_outputs()

        # Emit completion with audio path for playback
        self.completed.emit(vtt_path, txt_path, self.audio_path)

    def _prepare_audio(self) -> str:
        """Extract audio from video if needed."""
        info = get_file_info(self.filepath)

        if info.get('has_video') and info.get('has_audio'):
            self._temp_audio = True
            return extract_audio(self.filepath)

        return self.filepath

    def _load_model(self, model_size: str, device: str, compute_type: str):
        """Load Whisper model with specified settings."""
        from faster_whisper import WhisperModel

        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

        return model

    def _run_diarization(self):
        """Run speaker diarization and assign speakers to segments."""
        try:
            from pyannote.audio import Pipeline
            import torch
        except ImportError:
            self.text_chunk.emit("[Speaker ID unavailable in bundled app]\n")
            return

        try:
            # Check for HF token
            hf_token = os.environ.get('HF_TOKEN')
            if not hf_token:
                self.text_chunk.emit("[Skipping speaker ID: HF_TOKEN not set]\n")
                return

            # Load diarization pipeline
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )

            # Use MPS on Apple Silicon if available
            if torch.backends.mps.is_available():
                pipeline.to(torch.device("mps"))
            elif torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))

            # Run diarization
            diarization = pipeline(self.audio_path)

            # Build speaker timeline
            speaker_timeline = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_timeline.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                })

            # Create human-readable speaker labels
            speaker_map = {}
            speaker_count = 0

            # Assign speakers to transcript segments
            for segment in self.segments:
                seg_mid = (segment.start + segment.end) / 2

                for turn in speaker_timeline:
                    if turn['start'] <= seg_mid <= turn['end']:
                        raw_speaker = turn['speaker']
                        if raw_speaker not in speaker_map:
                            speaker_count += 1
                            speaker_map[raw_speaker] = f"Speaker {speaker_count}"
                        segment.speaker = speaker_map[raw_speaker]
                        break

            num_speakers = len(speaker_map)
            self.text_chunk.emit(f"[Identified {num_speakers} speaker(s)]\n")

        except Exception as e:
            self.text_chunk.emit(f"[Speaker ID unavailable: {str(e)[:50]}]\n")

    def _assess_quality(self) -> QualityMetrics:
        """Assess transcription quality."""
        if not self.segments:
            return QualityMetrics(0.8, 0.0, 0.0)

        confidences = [s.confidence for s in self.segments]
        avg_conf = sum(confidences) / len(confidences)

        low_conf_count = sum(1 for c in confidences if c < self.CONFIDENCE_THRESHOLD)
        low_conf_ratio = low_conf_count / len(confidences)

        texts = [s.text.lower() for s in self.segments]
        repetition_score = self._calculate_repetition(texts)

        return QualityMetrics(
            avg_confidence=avg_conf,
            low_confidence_ratio=low_conf_ratio,
            repetition_score=repetition_score
        )

    def _calculate_repetition(self, texts: list[str]) -> float:
        """Calculate repetition score."""
        if len(texts) < 3:
            return 0.0

        repetitions = 0
        for i in range(1, len(texts)):
            if texts[i] == texts[i-1] and len(texts[i]) > 10:
                repetitions += 1

        return repetitions / len(texts)

    def _save_outputs(self) -> tuple[str, str]:
        """Save transcription to VTT and TXT files with speaker labels."""
        vtt_path, txt_path = generate_output_paths(self.filepath)

        # Write VTT with speakers
        with open(vtt_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")

            for i, seg in enumerate(self.segments, 1):
                start = self._format_timestamp(seg.start)
                end = self._format_timestamp(seg.end)
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                if seg.speaker:
                    f.write(f"<v {seg.speaker}>{seg.text}</v>\n\n")
                else:
                    f.write(f"{seg.text}\n\n")

        # Write TXT with speaker labels
        with open(txt_path, 'w', encoding='utf-8') as f:
            current_speaker = None
            for seg in self.segments:
                if seg.speaker and seg.speaker != current_speaker:
                    current_speaker = seg.speaker
                    f.write(f"\n{current_speaker}:\n")
                f.write(f"{seg.text} ")

        return vtt_path, txt_path

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as VTT timestamp."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def _cleanup(self):
        """Clean up temporary files."""
        if self._temp_audio and not self._keep_audio and self.audio_path:
            if os.path.exists(self.audio_path):
                try:
                    os.remove(self.audio_path)
                except OSError:
                    pass

    def cancel(self):
        """Cancel the transcription."""
        self._cancelled = True
        self._keep_audio = False
        self._cleanup()
