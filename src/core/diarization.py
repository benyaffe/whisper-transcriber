"""
Speaker diarization using pyannote.audio.
Requires HuggingFace token with read access to gated repos.
"""

import os
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class SpeakerTurn:
    """A segment of audio attributed to a speaker."""
    start: float
    end: float
    speaker: str


class DiarizationError(Exception):
    """Error during diarization."""
    pass


class TokenValidationError(Exception):
    """HuggingFace token validation error."""
    pass


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
                raise RuntimeError(f"Download failed after {max_retries} attempts: {error_str[:60]}")


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

    # Models required by pyannote/speaker-diarization-3.1
    REQUIRED_MODELS = [
        {
            "repo": "pyannote/speaker-diarization-3.1",
            "name": "speaker-diarization-3.1",
            "url": "https://huggingface.co/pyannote/speaker-diarization-3.1",
            "gated": True
        },
        {
            "repo": "pyannote/segmentation-3.0",
            "name": "segmentation-3.0",
            "url": "https://huggingface.co/pyannote/segmentation-3.0",
            "gated": True
        },
        {
            "repo": "pyannote/speaker-diarization-community-1",
            "name": "speaker-diarization-community-1",
            "url": "https://huggingface.co/pyannote/speaker-diarization-community-1",
            "gated": True
        },
        {
            "repo": "pyannote/wespeaker-voxceleb-resnet34-LM",
            "name": "wespeaker-voxceleb-resnet34-LM",
            "url": "https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM",
            "gated": False
        },
    ]

    # First, check access to gated models
    log("[Speaker ID: Verifying model access...]")
    missing_licenses = []

    for model in REQUIRED_MODELS:
        if not model["gated"]:
            continue
        try:
            list_repo_files(model["repo"], token=token)
        except GatedRepoError:
            missing_licenses.append(model)
        except Exception as e:
            error_str = str(e).lower()
            if "403" in error_str or "401" in error_str or "gated" in error_str:
                missing_licenses.append(model)

    if missing_licenses:
        model_list = "\n".join([f"  - {m['name']}: {m['url']}" for m in missing_licenses])
        raise RuntimeError(
            f"Accept the license for these models:\n{model_list}\n\n"
            f"Click 'Agree and access repository' on each page."
        )

    # Download full snapshots with retry and cache verification
    for model in REQUIRED_MODELS:
        repo_id = model["repo"]

        # First, verify existing cache
        cache_ok, cache_msg = _verify_model_cache(repo_id, token)
        if cache_ok:
            log(f"[Speaker ID: {model['name']} - cache verified]")
            continue

        log(f"[Speaker ID: Downloading {model['name']} (with retry)...]")

        try:
            local_dir = _download_with_retry(repo_id, token, log, max_retries=3)
            log(f"[Speaker ID: {model['name']} downloaded]")

            # Verify the download
            verify_ok, verify_msg = _verify_model_cache(repo_id, token)
            if not verify_ok:
                log(f"[Speaker ID: Warning - {verify_msg}]")
        except Exception as e:
            error_msg = str(e)
            log(f"[Speaker ID: Download error: {error_msg[:80]}]")
            raise RuntimeError(f"Failed to download {model['name']}: {error_msg[:80]}")

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

    # Only these two models are gated and require license acceptance
    # wespeaker-voxceleb-resnet34-LM is NOT gated
    REQUIRED_MODELS = [
        ("pyannote/speaker-diarization-3.1", "https://huggingface.co/pyannote/speaker-diarization-3.1"),
        ("pyannote/segmentation-3.0", "https://huggingface.co/pyannote/segmentation-3.0"),
        ("pyannote/speaker-diarization-community-1", "https://huggingface.co/pyannote/speaker-diarization-community-1"),
    ]

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

        for repo_id, url in REQUIRED_MODELS:
            try:
                list_repo_files(repo_id, token=token)
            except GatedRepoError:
                missing_licenses.append((repo_id.split("/")[1], url))
            except Exception as e:
                error_str = str(e).lower()
                if "403" in error_str or "401" in error_str or "gated" in error_str:
                    missing_licenses.append((repo_id.split("/")[1], url))
                else:
                    access_denied.append(repo_id.split("/")[1])

        if missing_licenses:
            links = "\n".join([f"  {name}: {url}" for name, url in missing_licenses])
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
        raise DiarizationError(f"Failed to download models: {str(e)[:80]}")

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

        _GATED_REPOS = {
            "pyannote/speaker-diarization-3.1",
            "pyannote/segmentation-3.0",
            "pyannote/speaker-diarization-community-1",
        }

        def _offline_hf_download(*args, **kwargs):
            repo_id = args[0] if len(args) > 0 else kwargs.get('repo_id', '')
            if repo_id in _GATED_REPOS:
                kwargs['local_files_only'] = True
                kwargs.pop('force_download', None)
            return _original_hf_download(*args, **kwargs)

        # Patch in the source module and top-level module
        huggingface_hub.file_download.hf_hub_download = _offline_hf_download
        huggingface_hub.hf_hub_download = _offline_hf_download
        # Patch in any module that already imported hf_hub_download
        for mod_name, mod in list(sys.modules.items()):
            if (mod is not None
                and hasattr(mod, 'hf_hub_download')
                and getattr(mod, 'hf_hub_download') is _original_hf_download):
                setattr(mod, 'hf_hub_download', _offline_hf_download)

        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token
            )
        except TypeError:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
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
        error_str = str(e)
        error_type = type(e).__name__
        log(f"[Speaker ID: {error_type}: {error_str[:100]}]")
        raise DiarizationError(f"Pipeline load failed ({error_type}): {error_str[:120]}")
    finally:
        # Restore original hf_hub_download and get_plda
        huggingface_hub.file_download.hf_hub_download = _original_hf_download
        huggingface_hub.hf_hub_download = _original_hf_download
        for mod_name, mod in list(sys.modules.items()):
            if (mod is not None
                and hasattr(mod, 'hf_hub_download')
                and getattr(mod, 'hf_hub_download') is _offline_hf_download):
                setattr(mod, 'hf_hub_download', _original_hf_download)

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
        log(f"[Speaker ID: Running on {audio_path}]")
        diarization = pipeline(audio_path)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log(f"[Speaker ID: Diarization error details:]")
        # Log each line of traceback for visibility
        for line in tb.strip().split('\n')[-6:]:
            log(f"[  {line.strip()[:120]}]")
        raise DiarizationError(f"Diarization failed: {str(e)[:100]}")

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
