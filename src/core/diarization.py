"""
Speaker diarization using pyannote.audio.
Requires HuggingFace token with read access to gated repos.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass(frozen=True)
class ModelSpec:
    """A HuggingFace repo that speaker diarization needs at runtime."""
    repo: str
    gated: bool

    @property
    def name(self) -> str:
        return self.repo.split("/")[-1]

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}"


# The pipeline we load. Everything else in REQUIRED_MODELS is a repo this
# one pulls in when it initializes.
DIARIZATION_PIPELINE = "pyannote/speaker-diarization-3.1"

# Single source of truth: the pre-download pass, token validation, the
# offline download patch and the Settings dialog all derive from this.
REQUIRED_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(DIARIZATION_PIPELINE, gated=True),
    ModelSpec("pyannote/segmentation-3.0", gated=True),
    ModelSpec("pyannote/speaker-diarization-community-1", gated=True),
    ModelSpec("pyannote/wespeaker-voxceleb-resnet34-LM", gated=False),
)

# Gated repos raise 403 GatedRepoError on any network hit, even when the
# files are already cached, so they need separate handling in a few places.
GATED_MODELS: tuple[ModelSpec, ...] = tuple(m for m in REQUIRED_MODELS if m.gated)
GATED_REPOS: frozenset[str] = frozenset(m.repo for m in GATED_MODELS)


@dataclass
class SpeakerTurn:
    """A segment of audio attributed to a speaker."""
    start: float
    end: float
    speaker: str


# Fraction of the diarization phase each pyannote step accounts for, in the
# order pyannote runs them. Only "segmentation" and "embeddings" report
# incremental completed/total; the other two fire once when they finish, so
# they get a token weight rather than a share proportional to their runtime.
# Clustering runs unhooked between "embeddings" and "discrete_diarization",
# which is why the bar legitimately pauses there.
DIARIZATION_STAGE_WEIGHTS: dict[str, float] = {
    "segmentation": 0.45,
    "speaker_counting": 0.05,
    "embeddings": 0.45,
    "discrete_diarization": 0.05,
}

DIARIZATION_STAGE_LABELS: dict[str, str] = {
    "segmentation": "Finding speech",
    "speaker_counting": "Counting speakers",
    "embeddings": "Analyzing voices",
    "discrete_diarization": "Assigning speakers",
}


class DiarizationProgressHook:
    """Reports diarization progress to a callback as (fraction, label).

    Implements pyannote's hook contract, which is just a callable taking
    ``(step_name, step_artifact, file=None, total=None, completed=None)``.

    Deliberately *not* a subclass of pyannote's own ``ProgressHook``: that one
    is a ``rich`` terminal renderer that starts a live progress display on
    stdout, which is meaningless in a windowed .app whose stdout may not be a
    tty, and would pull ``rich`` into the bundle for no benefit.

    Emission is throttled to ``min_interval_s`` because the embeddings step
    fires once per batch, thousands of times on a long recording. Step
    changes and step completions always emit, so a caller never misses a
    transition or a finished stage.

    ``clock`` is injectable so tests can exercise the throttle without
    sleeping.
    """

    def __init__(
        self,
        progress_callback: Callable[[float, str], None],
        min_interval_s: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._callback = progress_callback
        self._min_interval = min_interval_s
        self._clock = clock
        self._completed_steps: set[str] = set()
        self._current_step: Optional[str] = None
        self._last_emit_at = float("-inf")
        self._last_fraction = 0.0

    # Context-manager protocol, so this can be used the same way pyannote's
    # own hooks are: `with DiarizationProgressHook(cb) as h: pipeline(f, hook=h)`
    def __enter__(self) -> "DiarizationProgressHook":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def __call__(
        self,
        step_name: str,
        step_artifact=None,
        file=None,
        total: Optional[int] = None,
        completed: Optional[int] = None,
    ):
        step_changed = step_name != self._current_step
        self._current_step = step_name

        if completed is None or not total or total <= 0:
            # A one-shot tick: pyannote calls these once, when the step is done.
            step_fraction = 1.0
        else:
            # Inference reports completed = c + batch_size, which overshoots
            # total on the final batch. Belt and braces: the `finished` branch
            # below is what actually stops an overshoot reaching the bar, since
            # it routes the step through _completed_steps at its full weight.
            step_fraction = min(max(completed / total, 0.0), 1.0)

        finished = step_fraction >= 1.0
        if finished:
            self._completed_steps.add(step_name)

        if not (step_changed or finished or self._throttle_expired()):
            return

        # pyannote revisits a step (segmentation ticks, then completes), and an
        # unknown future step contributes no weight. Never let the bar retreat.
        fraction = max(self._overall_fraction(step_name, step_fraction), self._last_fraction)

        self._last_emit_at = self._clock()
        self._last_fraction = fraction
        self._callback(fraction, self._label(step_name, total, completed))

    def _throttle_expired(self) -> bool:
        return (self._clock() - self._last_emit_at) >= self._min_interval

    def _overall_fraction(self, step_name: str, step_fraction: float) -> float:
        """Weighted progress across all stages, in [0.0, 1.0].

        An unrecognised step_name contributes nothing, so a future pyannote
        adding a stage cannot push the total past 1.0.
        """
        total = 0.0
        for name, weight in DIARIZATION_STAGE_WEIGHTS.items():
            if name in self._completed_steps:
                total += weight
            elif name == step_name:
                total += weight * step_fraction
        return min(total, 1.0)

    def _label(self, step_name: str, total: Optional[int], completed: Optional[int]) -> str:
        label = DIARIZATION_STAGE_LABELS.get(step_name)
        if label is None:
            # Unknown step from a future pyannote: surface it rather than hide it.
            label = step_name.replace("_", " ").capitalize()
        if completed is not None and total:
            return f"{label} {min(completed, total)}/{total}"
        return label


class DiarizationError(Exception):
    """Error during diarization."""
    pass


class TokenValidationError(Exception):
    """HuggingFace token validation error."""
    pass


def _decode_audio_to_tensor(path: str, sample_rate: int = 16000):
    """Decode audio to a (1, N) float32 torch tensor via the bundled ffmpeg.

    Bypasses pyannote's torchcodec-based loader, which cannot dlopen FFmpeg's
    shared libraries inside the PyInstaller bundle (we ship a static ffmpeg
    binary, not the libav* dylibs torchcodec expects).
    """
    import subprocess
    import numpy as np
    import torch
    from src.utils.file_utils import get_bundled_binary

    ffmpeg = get_bundled_binary('ffmpeg')
    cmd = [
        ffmpeg, '-nostdin', '-loglevel', 'error',
        '-i', path,
        '-vn', '-ac', '1', '-ar', str(sample_rate),
        '-f', 'f32le', '-',
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode('utf-8', errors='replace').strip()
        raise DiarizationError(f"ffmpeg decode failed: {stderr}")
    if not result.stdout:
        raise DiarizationError("ffmpeg produced no audio data")

    samples = np.frombuffer(result.stdout, dtype=np.float32)
    return torch.from_numpy(samples.copy()).unsqueeze(0)


def _download_with_retry(repo_id: str, token: str, log: Callable[[str], None], max_retries: int = 3):
    """
    Download a model with exponential backoff retry.
    Returns the local directory path on success.
    Raises RuntimeError on failure after all retries.
    """
    from huggingface_hub import snapshot_download
    import time

    delays = [1, 2, 4, 8]  # Exponential backoff

    for attempt in range(max_retries):
        try:
            local_dir = snapshot_download(
                repo_id=repo_id,
                token=token,
                local_files_only=False,
            )
            return local_dir
        except Exception as e:
            error_str = str(e)
            if attempt < max_retries - 1:
                delay = delays[min(attempt, len(delays) - 1)]
                log(f"[Speaker ID: Download failed, retrying in {delay}s... ({error_str[:40]})]")
                time.sleep(delay)
            else:
                raise RuntimeError(f"Download failed after {max_retries} attempts: {error_str}")


def _verify_model_cache(repo_id: str, token: str) -> tuple[bool, str]:
    """
    Verify a cached model has required files and isn't corrupted.
    Returns (is_valid, message).
    """
    from huggingface_hub import try_to_load_from_cache, HfFileSystem
    import os

    try:
        # Check if config.yaml is cached (all pyannote models have this)
        cached_config = try_to_load_from_cache(repo_id, "config.yaml")
        if cached_config is None:
            return False, "Model not cached"

        # Verify file exists and has content
        if not os.path.exists(cached_config):
            return False, "Cache path doesn't exist"

        size = os.path.getsize(cached_config)
        if size < 100:  # config.yaml should be at least 100 bytes
            return False, "Config file appears corrupted (too small)"

        return True, "Cache verified"
    except Exception as e:
        return False, f"Cache check failed: {str(e)[:40]}"


def _ensure_models_downloaded(token: str, log: Callable[[str], None]):
    """
    Pre-download all model files required for pyannote speaker diarization.
    Uses snapshot_download to get full repo structure that Pipeline expects.
    Implements retry with exponential backoff and cache verification.
    """
    from huggingface_hub import snapshot_download, list_repo_files
    from huggingface_hub.utils import GatedRepoError

    # First, check access to gated models
    log("[Speaker ID: Verifying model access...]")
    missing_licenses = []

    for model in GATED_MODELS:
        try:
            list_repo_files(model.repo, token=token)
        except GatedRepoError:
            missing_licenses.append(model)
        except Exception as e:
            error_str = str(e).lower()
            if "403" in error_str or "401" in error_str or "gated" in error_str:
                missing_licenses.append(model)

    if missing_licenses:
        model_list = "\n".join([f"  - {m.name}: {m.url}" for m in missing_licenses])
        raise RuntimeError(
            f"Accept the license for these models:\n{model_list}\n\n"
            f"Click 'Agree and access repository' on each page."
        )

    # Download full snapshots with retry and cache verification
    for model in REQUIRED_MODELS:
        repo_id = model.repo

        # First, verify existing cache
        cache_ok, cache_msg = _verify_model_cache(repo_id, token)
        if cache_ok:
            log(f"[Speaker ID: {model.name} - cache verified]")
            continue

        log(f"[Speaker ID: Downloading {model.name} (with retry)...]")

        try:
            local_dir = _download_with_retry(repo_id, token, log, max_retries=3)
            log(f"[Speaker ID: {model.name} downloaded]")

            # Verify the download
            verify_ok, verify_msg = _verify_model_cache(repo_id, token)
            if not verify_ok:
                log(f"[Speaker ID: Warning - {verify_msg}]")
        except Exception as e:
            error_msg = str(e)
            for line in error_msg.splitlines() or [""]:
                log(f"[Speaker ID: Download error: {line}]")
            raise RuntimeError(f"Failed to download {model.name}: {error_msg}")

    log("[Speaker ID: All models downloaded]")


def validate_hf_token(token: str) -> tuple[bool, str]:
    """
    Validate HuggingFace token can access ALL required pyannote models.
    Checks each model and reports which licenses need to be accepted.

    Returns:
        tuple: (is_valid, message)
    """
    if not token or not token.strip():
        return False, "Token is empty"

    token = token.strip()

    if not token.startswith("hf_"):
        return False, "Token should start with 'hf_'"

    try:
        from huggingface_hub import HfApi, list_repo_files
        from huggingface_hub.utils import GatedRepoError

        api = HfApi(token=token)

        # Check token is valid by getting user info
        try:
            user_info = api.whoami()
            username = user_info.get("name", "Unknown")
        except Exception as e:
            error_str = str(e).lower()
            if "401" in error_str or "unauthorized" in error_str:
                return False, "Invalid token - please check and re-enter"
            return False, f"Token error: {str(e)[:50]}"

        # Check access to ALL required models
        missing_licenses = []
        access_denied = []

        for model in GATED_MODELS:
            try:
                list_repo_files(model.repo, token=token)
            except GatedRepoError:
                missing_licenses.append(model)
            except Exception as e:
                error_str = str(e).lower()
                if "403" in error_str or "401" in error_str or "gated" in error_str:
                    missing_licenses.append(model)
                else:
                    access_denied.append(model.name)

        if missing_licenses:
            links = "\n".join([f"  {m.name}: {m.url}" for m in missing_licenses])
            return False, (
                f"Accept the license for these models:\n{links}\n\n"
                "Click 'Agree and access repository' on each page."
            )

        if access_denied:
            return False, (
                f"Cannot access: {', '.join(access_denied)}\n"
                "Ensure your token has 'Read' permission for gated repos."
            )

        return True, f"Token valid for '{username}' - all model licenses accepted"

    except ImportError:
        return False, "huggingface_hub not installed"
    except Exception as e:
        return False, f"Validation error: {str(e)[:50]}"


def run_diarization(
    audio_path: str,
    hf_token: str,
    status_callback: Optional[Callable[[str], None]] = None
) -> list[SpeakerTurn]:
    """
    Run speaker diarization using pyannote.audio.

    Args:
        audio_path: Path to audio file
        hf_token: HuggingFace token (required)
        status_callback: Optional callback for status messages

    Returns:
        list of SpeakerTurn

    Raises:
        DiarizationError: If diarization fails
        TokenValidationError: If token is invalid
    """
    def log(msg: str):
        if status_callback:
            status_callback(msg)

    if not hf_token:
        raise TokenValidationError("HuggingFace token required for speaker identification")

    # Validate token first
    is_valid, message = validate_hf_token(hf_token)
    if not is_valid:
        raise TokenValidationError(message)

    log("[Speaker ID: Token validated]")

    # Ensure cache directory exists and set environment variables
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_TOKEN"] = hf_token  # Some libraries look for this
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token  # Alternative env var

    try:
        from pyannote.audio import Pipeline
        import torch
    except ImportError as e:
        raise DiarizationError(f"pyannote.audio not available: {e}")

    # Pre-download all required model files with progress
    log("[Speaker ID: Checking model files...]")
    try:
        _ensure_models_downloaded(hf_token, log)
    except Exception as e:
        raise DiarizationError(f"Failed to download models: {e}")

    log("[Speaker ID: Loading pipeline...]")

    try:
        # Monkey-patch hf_hub_download to force local_files_only=True ONLY
        # for gated repos (which cause 403 GatedRepoError even when cached).
        # Non-gated repos are allowed to download normally.
        # Setting HF_HUB_OFFLINE env var or constant doesn't reliably work
        # in bundled PyInstaller apps.
        import sys
        import huggingface_hub
        import huggingface_hub.file_download
        _original_hf_download = huggingface_hub.file_download.hf_hub_download

        def _offline_hf_download(*args, **kwargs):
            repo_id = args[0] if len(args) > 0 else kwargs.get('repo_id', '')
            if repo_id in GATED_REPOS:
                kwargs['local_files_only'] = True
                kwargs.pop('force_download', None)
            return _original_hf_download(*args, **kwargs)

        # Patch in the source module and top-level module
        huggingface_hub.file_download.hf_hub_download = _offline_hf_download
        huggingface_hub.hf_hub_download = _offline_hf_download
        # Patch in any module that already imported hf_hub_download.
        # Use vars() to avoid triggering __getattr__ on lazy modules
        # (e.g. speechbrain.integrations.k2_fsa raises ImportError on access).
        for mod_name, mod in list(sys.modules.items()):
            if mod is None:
                continue
            try:
                mod_dict = vars(mod)
            except TypeError:
                continue
            if mod_dict.get('hf_hub_download') is _original_hf_download:
                mod_dict['hf_hub_download'] = _offline_hf_download

        pipeline = Pipeline.from_pretrained(
            DIARIZATION_PIPELINE,
            token=hf_token
        )
        log("[Speaker ID: Pipeline loaded successfully]")
    except FileNotFoundError as e:
        error_str = str(e)
        log(f"[Speaker ID: FileNotFoundError details:]")
        log(f"[{str(e)}]")
        raise DiarizationError(
            f"Model file missing: {error_str}\n"
            f"Try clearing cache: rm -rf ~/.cache/huggingface/hub/models--pyannote*"
        )
    except Exception as e:
        import traceback
        error_str = str(e)
        error_type = type(e).__name__
        log(f"[Speaker ID: {error_type}: {error_str}]")
        for line in traceback.format_exc().rstrip().splitlines():
            log(f"[  {line}]")
        raise DiarizationError(f"Pipeline load failed ({error_type}): {error_str}")
    finally:
        # Restore original hf_hub_download. Use vars() to avoid triggering
        # __getattr__ on lazy modules (e.g. speechbrain lazy k2 integration).
        huggingface_hub.file_download.hf_hub_download = _original_hf_download
        huggingface_hub.hf_hub_download = _original_hf_download
        for mod_name, mod in list(sys.modules.items()):
            if mod is None:
                continue
            try:
                mod_dict = vars(mod)
            except TypeError:
                continue
            if mod_dict.get('hf_hub_download') is _offline_hf_download:
                mod_dict['hf_hub_download'] = _original_hf_download

    # Use GPU if available
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        log("[Speaker ID: Using Apple Silicon GPU]")
        pipeline.to(torch.device("mps"))
    elif torch.cuda.is_available():
        device = "cuda"
        log("[Speaker ID: Using CUDA GPU]")
        pipeline.to(torch.device("cuda"))
    else:
        log("[Speaker ID: Using CPU]")

    log("[Speaker ID: Analyzing speakers...]")

    try:
        from pathlib import Path
        log(f"[Speaker ID: Decoding audio via bundled ffmpeg...]")
        waveform = _decode_audio_to_tensor(audio_path, sample_rate=16000)
        log(f"[Speaker ID: Decoded {waveform.shape[1] / 16000:.1f}s of audio]")
        log(f"[Speaker ID: Running diarization pipeline...]")
        diarization = pipeline({
            "waveform": waveform,
            "sample_rate": 16000,
            "uri": Path(audio_path).stem,
        })
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log(f"[Speaker ID: Diarization error details:]")
        for line in tb.rstrip().splitlines():
            log(f"[  {line}]")
        raise DiarizationError(f"Diarization failed: {e}")

    # pyannote 4.x returns DiarizeOutput; extract the Annotation
    annotation = diarization
    if hasattr(diarization, 'speaker_diarization'):
        annotation = diarization.speaker_diarization

    turns = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append(SpeakerTurn(
            start=turn.start,
            end=turn.end,
            speaker=speaker
        ))

    num_speakers = len(set(t.speaker for t in turns))
    log(f"[Speaker ID: Found {num_speakers} speaker(s)]")

    return turns


def assign_speakers_to_segments(segments: list, turns: list[SpeakerTurn]) -> dict[str, str]:
    """
    Assign speaker labels to transcription segments based on diarization.

    Returns a mapping from raw speaker IDs to readable labels.
    """
    if not turns:
        # No diarization - assign all to Speaker 1
        for seg in segments:
            seg.speaker = "Speaker 1"
        return {}

    speaker_map = {}
    speaker_count = 0

    for seg in segments:
        seg_mid = (seg.start + seg.end) / 2

        # Find matching turn
        matched = False
        for turn in turns:
            if turn.start <= seg_mid <= turn.end:
                raw = turn.speaker
                if raw not in speaker_map:
                    speaker_count += 1
                    speaker_map[raw] = f"Speaker {speaker_count}"
                seg.speaker = speaker_map[raw]
                matched = True
                break

        if not matched:
            # Fallback: find closest turn
            if turns:
                closest = min(turns, key=lambda t: abs((t.start + t.end) / 2 - seg_mid))
                raw = closest.speaker
                if raw not in speaker_map:
                    speaker_count += 1
                    speaker_map[raw] = f"Speaker {speaker_count}"
                seg.speaker = speaker_map[raw]
            else:
                seg.speaker = "Speaker 1"

    return speaker_map
