"""
Transcription pipeline, with no Qt anywhere.

Everything that decides *what happens* lives here; everything that decides *how
it is displayed* stays in the Qt wrapper (src/core/transcriber.py). The runner
reports through an observer object rather than signals, so it can be driven
from a plain thread, a test, or PodcastNotesWT's stage machine.

Two deliberate constraints:

* Settings are never read here. hf_token and enable_speaker_id are constructor
  arguments. PodcastNotesWT needs speaker ID on with a pinned model regardless
  of what the Transcribe-mode checkbox says, and a runner that consulted
  QSettings could not give it that.
* run() raises. It does not swallow exceptions into an error callback; the
  caller decides what an error means.

Callbacks fire on whatever thread called run(). Anything touching a UI has to
marshal them itself.
"""

import os
import platform
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from src.core.diarization import (
    DiarizationError,
    TokenValidationError,
    assign_speakers_to_segments,
    run_diarization,
)
from src.utils.file_utils import extract_audio, generate_output_paths, get_file_info
from src.utils.logger import get_logger, log_exception

# Memory requirements per model (approximate, in GB)
# Peak resident memory per model, in GB, for the stack this app actually runs:
# faster-whisper at int8 on CPU. The numbers here used to be the figures for
# fp32 openai-whisper, which is a different implementation using roughly two
# and a half times as much, so the check warned people off runs that were
# never going to be a problem.
#
# Measured for medium on 2026-09-11: 1.75 GB with the weights loaded, 2.02 GB
# peak during inference. The others are scaled from it by parameter count
# (39M, 74M, 244M, 769M, 1550M), because int8 is about a byte a parameter plus
# a fairly constant working set.
MODEL_MEMORY_REQUIREMENTS = {
    "tiny": 0.5,
    "base": 0.7,
    "small": 1.1,
    "medium": 2.1,
    "large": 3.8,
}

# What speaker identification adds on top. Separate because it is optional and
# because it is a second model: leaving it in the figures above would warn
# somebody transcribing a solo lecture about memory for a stage they turned
# off.
DIARIZATION_MEMORY_GB = 1.5

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


def check_memory_available(
    model_size: str,
    file_duration_minutes: float,
    speaker_id: bool = False,
) -> tuple[bool, str]:
    """Whether there is room to run, and what to say if it is close.

    Returns (is_ok, warning_message). An empty message means nothing to say.

    A false alarm here is not harmless: it tells somebody to close their apps
    or pick a worse model before a job that would have finished. That happened
    on a real 42-minute trip, which warned at 4.8 GB available against 5.4 GB
    "needed" and then completed, diarization included.
    """
    import psutil

    try:
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)

        required_gb = MODEL_MEMORY_REQUIREMENTS.get(model_size, 2.1)
        if speaker_id:
            required_gb += DIARIZATION_MEMORY_GB
        # Audio processing, roughly 0.1GB per 10 minutes.
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
class WordTiming:
    """One word with its own timing, from Whisper's word_timestamps pass."""
    start: float
    end: float
    word: str
    probability: float


@dataclass
class TranscriptionSegment:
    """A segment of transcribed text with timing, confidence, and speaker."""
    start: float
    end: float
    text: str
    confidence: float
    speaker: Optional[str] = None
    # Whisper computes these whenever word_timestamps=True. They used to be
    # read once for an average confidence and dropped; the JSON export needs
    # them so a later rewrite of the text can keep accurate timings.
    words: list[WordTiming] = field(default_factory=list)


@dataclass
class QualityMetrics:
    """Quality metrics for transcription assessment."""
    avg_confidence: float
    low_confidence_ratio: float
    repetition_score: float


@dataclass
class TranscriptionOutputs:
    """What a completed run produced."""
    vtt_path: str
    txt_path: str
    json_path: str
    audio_path: str
    segments: list
    speaker_id_used: bool


def detect_diarization_device() -> str:
    """Which device pyannote will pick, which is not the one Whisper uses.

    detect_optimal_settings() reports "cpu" on Apple Silicon because
    faster-whisper runs int8 on CPU there, while run_diarization independently
    moves the pipeline to MPS. Mirrors the selection in
    src/core/diarization.py so the two cannot drift.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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


class TranscriptionObserver:
    """Everything the runner reports. Every method is a no-op by default.

    Subclass and override only what you care about. This is deliberately a
    class rather than eight constructor callbacks: the Qt wrapper overrides
    most of them, a test overrides two, and neither has to pass a row of
    `lambda *a: None` for the rest.
    """

    def hardware(self, description: str) -> None:
        """Hardware chosen for this run, e.g. "Apple Silicon (optimized)"."""

    def status(self, message: str) -> None:
        """Free-text progress line, already bracketed for display."""

    def progress(self, percent: float, eta_seconds: int) -> None:
        """Transcription progress, 0 to the ceiling. eta_seconds may be 0."""

    def diarization_progress(self, percent: float, stage: str) -> None:
        """Speaker-ID progress, ceiling to 100, with a human stage name."""

    def segment(self, start: float, end: float, text: str, speaker: str) -> None:
        """One transcribed segment, as soon as it is available."""

    def language_detected(self, language: str, confidence: float) -> None:
        """What Whisper decided the audio is in."""

    def quality_warning(self, message: str) -> None:
        """Something is off but the run continues."""

    def audio_ready(self, path: str) -> None:
        """Decoded audio is on disk and playable."""


class TranscriptionRunner:
    """Runs one file through Whisper and, optionally, pyannote."""

    # Quality thresholds
    CONFIDENCE_THRESHOLD = 0.6
    LOW_CONFIDENCE_RATIO_THRESHOLD = 0.25
    ASSESSMENT_DURATION = 120  # seconds

    # Abort if this long elapses between two segments.
    SEGMENT_TIMEOUT = 300  # 5 minutes

    # When speaker ID is going to run, transcription owns the bar up to a
    # ceiling and diarization owns the rest. The ceiling cannot be a constant:
    # both phases scale linearly with audio duration, so their ratio is a
    # property of the model and the hardware. Measured on Apple Silicon,
    # transcription runs at ~0.34x realtime and diarization at ~0.04x, but
    # without MPS diarization falls back to CPU and takes a far larger share.
    # These factors only apportion the progress bar and never affect output.
    DIARIZATION_REALTIME_FACTOR = {"mps": 0.045, "cuda": 0.045, "cpu": 0.15}
    MIN_TRANSCRIBE_CEILING = 55.0
    MAX_TRANSCRIBE_CEILING = 92.0
    DEFAULT_TRANSCRIBE_CEILING = 80.0  # until the first throughput measurement lands

    # Transcribe this much audio before trusting a throughput measurement.
    CEILING_ESTIMATE_AFTER_AUDIO_S = 20.0

    def __init__(
        self,
        filepath: str,
        *,
        model: str = "medium",
        language: Optional[str] = None,
        hf_token: str = "",
        enable_speaker_id: bool = False,
        observer: Optional[TranscriptionObserver] = None,
    ):
        self.filepath = filepath
        self.model_size = model
        self.language = LANGUAGE_CODES.get(language, language) if language else None
        self.hf_token = hf_token
        self.enable_speaker_id = enable_speaker_id
        self.observer = observer or TranscriptionObserver()

        self.segments: list[TranscriptionSegment] = []
        self.audio_path: Optional[str] = None

        # cancel() is called from another thread, so this is an Event rather
        # than a bool.
        self._cancel = threading.Event()

        self._logger = get_logger()
        self._temp_dir: Optional[str] = None
        self._will_diarize = False
        self._progress_ceiling = self.DEFAULT_TRANSCRIBE_CEILING
        self._diarization = None
        self._speaker_map: dict[str, str] = {}
        # Initialized here rather than at the top of run(), so that any caller
        # reaching _save_outputs cannot hit an AttributeError.
        self._speaker_id_used = False
        self._last_segment_time = time.monotonic()

    # --- public API -----------------------------------------------------------

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self):
        """Ask the run to stop. Safe to call from another thread."""
        self._cancel.set()

    @property
    def last_segment_time(self) -> float:
        """Wall clock of the most recent segment, for an external stall watcher."""
        return self._last_segment_time

    def run(self) -> Optional[TranscriptionOutputs]:
        """Transcribe the file. Returns None if cancelled. Raises on failure."""
        try:
            self._logger.info(f"Starting transcription: {self.filepath}")
            outputs = self._transcribe()
            if outputs is not None:
                self._logger.info("Transcription completed successfully")
            return outputs
        finally:
            # Success, failure and cancellation all release the snapshot.
            self._cleanup_temp_audio()

    # --- pipeline -------------------------------------------------------------

    def _transcribe(self) -> Optional[TranscriptionOutputs]:
        from faster_whisper import WhisperModel

        obs = self.observer

        # Decide up front whether speaker ID will run, so the progress bar can
        # reserve room for it rather than discovering it at 95%. How much room
        # is refined once from measured throughput, below.
        self._will_diarize = bool(self.enable_speaker_id and self.hf_token)
        self._progress_ceiling = (
            self.DEFAULT_TRANSCRIBE_CEILING if self._will_diarize else 100.0
        )
        ceiling_measured = not self._will_diarize
        diarization_device = detect_diarization_device() if self._will_diarize else "cpu"

        # Step 1: Hardware detection
        device, compute_type, hw_desc = detect_optimal_settings()
        obs.hardware(f"Using {hw_desc}")
        obs.status(f"[Hardware: {hw_desc}]")
        self._logger.info(f"Hardware: {hw_desc}, device={device}, compute={compute_type}")

        # Step 2: Prepare audio
        obs.status("[Preparing audio...]")
        self.audio_path = self._prepare_audio()
        obs.audio_ready(self.audio_path)

        # Step 2.5: Memory check
        info = get_file_info(self.audio_path)
        duration_minutes = info.get('duration', 0) / 60
        mem_ok, mem_warning = check_memory_available(
            self.model_size, duration_minutes, speaker_id=self.enable_speaker_id
        )
        if not mem_ok:
            obs.quality_warning(mem_warning)
            self._logger.warning(f"Memory warning: {mem_warning}")
        elif mem_warning:
            obs.status(f"[{mem_warning}]")

        # Step 3: Load model (check cancellation before slow operation)
        if self.cancelled:
            return None
        obs.status(f"[Loading {self.model_size} model ({compute_type})...]")
        self._logger.info(f"Loading model: {self.model_size}")
        model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
        if self.cancelled:
            return None

        # Step 4: Get duration
        info = get_file_info(self.audio_path)
        total_duration = info.get('duration', 0)

        # Step 5: Transcribe
        lang_info = f"language: {self.language}" if self.language else "auto-detecting language"
        obs.status(f"[Starting transcription ({lang_info})...]")

        segments_gen, trans_info = model.transcribe(
            self.audio_path,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            language=self.language
        )

        obs.language_detected(trans_info.language, trans_info.language_probability)

        # Process segments
        start_time = time.monotonic()
        assessed = False

        # Start the stall clock here, not in the constructor. Everything above
        # this point -- model load, a first-run model download, audio
        # extraction -- can easily take longer than SEGMENT_TIMEOUT, and the
        # clock used to be running throughout, so a slow start killed the run
        # with "appears stuck" before a single segment had been attempted.
        #
        # monotonic, not time(): on macOS time.monotonic() does not advance
        # while the machine is asleep, and wall clock does. A laptop that naps
        # for fifteen minutes mid-transcription would otherwise look exactly
        # like a wedged one. That is not hypothetical; it killed a 42-minute
        # run during testing, and the log showed 863 seconds of Maintenance
        # Sleep against an 875 second "stall".
        self._last_segment_time = time.monotonic()

        for segment in segments_gen:
            if self.cancelled:
                return None

            # Stall check. Note what this can and cannot do: it only runs when
            # the generator yields, so it reports a gap between two segments
            # after the fact. It cannot fire while faster-whisper is genuinely
            # wedged inside a blocking call, because then we never get here.
            # The Qt wrapper watches last_segment_time for that case.
            now = time.monotonic()
            gap = now - self._last_segment_time
            if gap > self.SEGMENT_TIMEOUT:
                self._logger.error(f"Watchdog: {gap:.0f}s gap between segments")
                raise RuntimeError(
                    f"Transcription stalled (no progress for {gap / 60:.0f} minutes). "
                    "The file may be corrupted or incompatible."
                )
            self._last_segment_time = now

            # Calculate confidence
            words = [
                WordTiming(
                    start=w.start,
                    end=w.end,
                    word=w.word,
                    probability=w.probability,
                )
                for w in (segment.words or [])
            ]
            avg_conf = (
                sum(w.probability for w in words) / len(words) if words else 0.8
            )

            trans_seg = TranscriptionSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
                confidence=avg_conf,
                speaker=None,
                words=words,
            )
            self.segments.append(trans_seg)
            obs.segment(segment.start, segment.end, segment.text.strip(), "")

            # Once enough audio has gone through to trust the throughput
            # figure, size the speaker-ID share of the bar and freeze it. Doing
            # this once rather than continuously is what keeps the bar
            # monotonic; it lands within the first few percent, so the step is
            # under a percentage point.
            if not ceiling_measured and segment.end >= self.CEILING_ESTIMATE_AFTER_AUDIO_S:
                ceiling_measured = True
                self._progress_ceiling = self._estimate_progress_ceiling(
                    audio_duration=total_duration,
                    elapsed=time.monotonic() - start_time,
                    audio_done=segment.end,
                    device=diarization_device,
                )
                self._logger.info(
                    f"Progress split: transcription 0-{self._progress_ceiling:.0f}%, "
                    f"speaker ID {self._progress_ceiling:.0f}-100% "
                    f"(diarization device={diarization_device})"
                )

            # Progress
            if total_duration > 0:
                percent = (segment.end / total_duration) * self._progress_ceiling
                elapsed = time.monotonic() - start_time
                if segment.end > 0:
                    rate = elapsed / segment.end
                    remaining = (total_duration - segment.end) * rate
                    obs.progress(percent, int(remaining))
                else:
                    obs.progress(percent, 0)

            # Quality assessment at 2 minutes. Warns; does not act. There used
            # to be an automatic medium-to-large upgrade here, but it never
            # worked: it rebound segments_gen inside `for segment in
            # segments_gen`, and Python bound that iterator once at loop entry.
            if not assessed and segment.end >= self.ASSESSMENT_DURATION:
                assessed = True
                quality = self._assess_quality()

                if (quality.avg_confidence < self.CONFIDENCE_THRESHOLD or
                        quality.low_confidence_ratio > self.LOW_CONFIDENCE_RATIO_THRESHOLD):
                    obs.quality_warning(
                        f"Quality issues detected ({quality.avg_confidence:.0%} confidence). "
                        f"Try the large model for this file."
                    )

        # Step 6: Speaker diarization
        if self.cancelled:
            return None
        if self._will_diarize:
            obs.diarization_progress(self._progress_ceiling, "Loading speaker model")
        self._run_diarization()

        # Step 7: Save outputs
        if self.cancelled:
            return None
        vtt_path, txt_path, json_path = self._save_outputs()
        return TranscriptionOutputs(
            vtt_path=vtt_path,
            txt_path=txt_path,
            json_path=json_path,
            audio_path=self.audio_path,
            segments=self.segments,
            speaker_id_used=self._speaker_id_used,
        )

    # --- audio ----------------------------------------------------------------

    def _prepare_audio(self) -> str:
        """Extract audio from video if needed; otherwise snapshot the
        source to a stable temp path so the pipeline survives the source
        being moved or deleted mid-run (folder watchers, archival scripts)."""
        info = get_file_info(self.filepath)
        if info.get('has_video') and info.get('has_audio'):
            return extract_audio(self.filepath)
        return self._snapshot_audio(self.filepath)

    def _snapshot_audio(self, src: str) -> str:
        import shutil
        import tempfile

        tmp_dir = tempfile.mkdtemp(prefix='wt_source_')
        dst = os.path.join(tmp_dir, os.path.basename(src))
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
        # Remembered so _cleanup_temp_audio can remove it. A hardlink costs
        # nothing, but a cross-volume source (external drive, network share)
        # falls through to a full copy that used to be left behind forever.
        self._temp_dir = tmp_dir
        return dst

    def _cleanup_temp_audio(self):
        """Remove the snapshot directory, if we made one. Never raises."""
        if not self._temp_dir:
            return
        import shutil

        try:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._logger.info(f"Removed temp audio snapshot: {self._temp_dir}")
        finally:
            self._temp_dir = None

    # --- progress -------------------------------------------------------------

    @classmethod
    def _estimate_progress_ceiling(
        cls, audio_duration: float, elapsed: float, audio_done: float, device: str
    ) -> float:
        """Where transcription should stop so speaker ID gets a fair share of the bar.

        Extrapolates total transcription time from throughput so far, predicts
        diarization time from the device's realtime factor, and splits the bar
        in proportion. Returns a percentage, clamped so neither phase can be
        squeezed into nothing by a wild measurement.
        """
        if audio_duration <= 0 or audio_done <= 0 or elapsed <= 0:
            return cls.DEFAULT_TRANSCRIBE_CEILING

        predicted_transcribe = (elapsed / audio_done) * audio_duration
        factor = cls.DIARIZATION_REALTIME_FACTOR.get(
            device, cls.DIARIZATION_REALTIME_FACTOR["cpu"]
        )
        predicted_diarize = factor * audio_duration

        total = predicted_transcribe + predicted_diarize
        if total <= 0:
            return cls.DEFAULT_TRANSCRIBE_CEILING

        ceiling = 100.0 * predicted_transcribe / total
        return min(max(ceiling, cls.MIN_TRANSCRIBE_CEILING), cls.MAX_TRANSCRIBE_CEILING)

    # --- diarization ----------------------------------------------------------

    def _run_diarization(self):
        """Run speaker diarization and assign speakers to segments."""
        obs = self.observer

        if not self.enable_speaker_id:
            obs.status("[Speaker ID: Disabled]")
            self._speaker_id_used = False
            return
        if not self.hf_token:
            obs.status("[Speaker ID: No token configured - see Settings]")
            self._speaker_id_used = False
            return

        # Map the hook's 0.0-1.0 onto the slice of the bar we reserved.
        floor = self._progress_ceiling
        span = 100.0 - floor

        def progress_cb(fraction: float, stage_label: str):
            obs.diarization_progress(floor + fraction * span, stage_label)

        try:
            obs.status("[Speaker ID: Starting...]")
            result = run_diarization(
                self.audio_path,
                self.hf_token,
                status_callback=obs.status,
                progress_callback=progress_cb,
            )

            if not result.turns:
                obs.status("[Speaker ID: No speakers detected]")
                self._speaker_id_used = False
                return

            self._diarization = result
            # Align against the non-overlapping view where pyannote gave us
            # one. With the overlapping annotation, a segment spoken over by
            # two people is attributed to whichever turn happens to be first in
            # the list. For a group recording where people talk across each
            # other, that is most of the interesting material.
            alignment_turns = result.exclusive_turns or result.turns
            speaker_map = assign_speakers_to_segments(self.segments, alignment_turns)
            self._speaker_map = speaker_map
            self._speaker_id_used = True
            assigned = sum(1 for s in self.segments if s.speaker)
            obs.status(
                f"[Speaker ID complete: {len(speaker_map)} speaker(s), "
                f"{assigned}/{len(self.segments)} segments tagged]"
            )

        except TokenValidationError as e:
            for line in str(e).splitlines() or [""]:
                obs.status(f"[Speaker ID: Token error - {line}]")
            self._speaker_id_used = False

        except DiarizationError as e:
            for line in str(e).splitlines() or [""]:
                obs.status(f"[Speaker ID failed: {line}]")
            self._speaker_id_used = False

        except Exception as e:
            import traceback

            obs.status(f"[Speaker ID error: {type(e).__name__}: {e}]")
            for line in traceback.format_exc().rstrip().splitlines():
                obs.status(f"[  {line}]")
            self._speaker_id_used = False

    # --- quality --------------------------------------------------------------

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

    # --- output ---------------------------------------------------------------

    def _save_outputs(self) -> tuple[str, str, str]:
        """Save transcription to VTT, TXT and JSON with speaker labels."""
        vtt_path, txt_path, json_path = generate_output_paths(self.filepath)

        self._logger.info(
            f"Saving outputs: speaker_id_used={self._speaker_id_used}, "
            f"segments={len(self.segments)}"
        )

        # VTT - only include speaker tags if speaker ID was successfully used
        with open(vtt_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            for i, seg in enumerate(self.segments, 1):
                start = format_vtt_time(seg.start)
                end = format_vtt_time(seg.end)
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

        # JSON - the machine-readable one. Written on every run, including when
        # speaker ID is off, so a consumer can rely on it existing.
        self._save_json(json_path)

        return vtt_path, txt_path, json_path

    def _save_json(self, json_path: str):
        """Write the segment JSON. Never fails the run.

        A transcription that produced a good VTT and TXT should not be reported
        as failed because the extra machine-readable file could not be written.
        """
        from src.core.json_export import build_payload, write_json

        try:
            info = get_file_info(self.audio_path or self.filepath)
            payload = build_payload(
                source_path=self.filepath,
                segments=self.segments,
                duration=info.get('duration', 0.0),
                model=self.model_size,
                language=self.language,
                device=detect_diarization_device() if self._speaker_id_used else "cpu",
                speaker_id_used=self._speaker_id_used,
                speaker_map=self._speaker_map,
                diarization=self._diarization,
            )
            write_json(json_path, payload)
            self._logger.info(f"Wrote segment JSON: {json_path}")
        except Exception as e:
            log_exception(e, "json export")
            self.observer.status(f"[JSON export failed: {type(e).__name__}: {e}]")


def format_vtt_time(seconds: float) -> str:
    """Format seconds as a WebVTT timestamp."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
