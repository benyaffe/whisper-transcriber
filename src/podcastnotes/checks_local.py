"""
The checks that need nothing but this machine.

ffmpeg, the speech models, and the HuggingFace token. None of these involve a
sign-in, so they work from the first launch and they are what the list shows
while the account-based checks are still being set up.
"""

import os

from src.podcastnotes.readiness import Check, failed, ok

HF_TOKEN_URL = "https://huggingface.co/settings/tokens"


def check_ffmpeg():
    """ffmpeg is bundled, so this should never fail. It has, though.

    It silently lost its execute bit during development, and every downstream
    failure that produced was baffling: audio "corrupt", durations zero,
    joins refusing. Two seconds of checking beats an hour of that.
    """
    from src.utils.file_utils import check_ffmpeg_health, get_bundled_binary

    path = get_bundled_binary("ffmpeg")
    healthy, message = check_ffmpeg_health()
    if healthy:
        # check_ffmpeg_health returns the first line of ffmpeg's banner,
        # truncated to 50 characters, which lands mid-word. Nobody reading
        # this list needs the copyright notice.
        return ok(_version_from_banner(message))

    if os.path.isabs(path) and os.path.exists(path) and not os.access(path, os.X_OK):
        return failed(
            "The bundled audio tool is not marked executable.",
            remedy="Reinstalling the app fixes this.",
        )
    return failed(
        message,
        remedy="Reinstall the app; the audio tool ships inside it.",
    )


def _version_from_banner(message: str) -> str:
    """"FFmpeg OK: ffmpeg version 9.0.1 Copyright (c) 2000-20" becomes
    "Ready (version 9.0.1)"."""
    import re

    match = re.search(r"ffmpeg version (\S+)", message)
    return f"Ready (version {match.group(1)})" if match else "Ready"


def check_models():
    """The speech models, roughly 3GB, downloaded on first use.

    Reports what is missing rather than just that something is, because the
    download is long enough that people want to know how much is left.
    """
    from huggingface_hub import try_to_load_from_cache

    from src.core.diarization import REQUIRED_MODELS

    missing = []
    for model in REQUIRED_MODELS:
        try:
            if try_to_load_from_cache(model.repo, "config.yaml") is None:
                missing.append(model.name)
        except Exception:
            missing.append(model.name)

    whisper_cached = _whisper_is_cached()
    if not whisper_cached:
        missing.insert(0, "the transcription model")

    if not missing:
        return ok("All downloaded")

    return failed(
        f"{len(missing)} still to download: {', '.join(missing[:3])}"
        + ("..." if len(missing) > 3 else ""),
        remedy="These download by themselves the first time you run a trip. "
               "It takes a while and needs the network.",
    )


def _whisper_is_cached() -> bool:
    """faster-whisper caches under the HuggingFace hub directory too."""
    from pathlib import Path

    hub = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub"
    try:
        return any(hub.glob("models--Systran--faster-whisper-medium"))
    except Exception:
        return False


SKIP_LABEL = "Skip this"


def check_huggingface():
    """The token that unlocks the speaker-identification models.

    Guided by default, because naming who said what is most of the value and
    somebody who does not know what HuggingFace is will otherwise simply stop
    here. Genuinely optional, though, so every failure offers a way past it:
    without speaker names the transcript still gets written, it just says
    "Speaker 1" instead of a person.

    Validates against HuggingFace rather than checking that a token exists. A
    revoked token, or one whose owner never accepted the three model licences,
    is indistinguishable from a good one until something calls with it.
    """
    from src.core.config import get_hf_token, is_speaker_id_enabled
    from src.core.diarization import token_status

    if not is_speaker_id_enabled():
        # A deliberate choice, so it is not a problem and must not hold up the
        # rest of the list.
        return ok("Off, so transcripts will not name speakers")

    status = token_status(get_hf_token())

    if status.valid:
        return ok(f"Signed in as '{status.username}', all licences accepted")

    if status.problem == "empty":
        return failed(
            "Not set up yet.",
            remedy="Takes about two minutes: a free account, a token, and "
                   "three licences to accept. Or skip it and go without "
                   "speaker names.",
            action="Set up",
            skip_action=SKIP_LABEL,
        )

    if status.problem == "licences":
        remaining = ", ".join(m.name for m in status.missing_licences)
        return failed(
            status.detail,
            remedy=f"Still to accept: {remaining}. Each one is a single "
                   f"button on its page.",
            action="Continue setup",
            skip_action=SKIP_LABEL,
            url=status.missing_licences[0].url,
        )

    return failed(
        status.detail,
        remedy="Check the token, or skip it and go without speaker names.",
        url=HF_TOKEN_URL,
        action="Set up",
        skip_action=SKIP_LABEL,
    )


LOCAL_CHECKS = [
    Check(
        key="ffmpeg",
        title="Audio tools",
        purpose="Reading and joining your recordings",
        run=check_ffmpeg,
        fixable=False,
    ),
    Check(
        key="models",
        title="Speech models",
        purpose="Turning speech into text",
        run=check_models,
        fixable=False,
    ),
    Check(
        key="huggingface",
        title="HuggingFace",
        purpose="Telling voices apart",
        run=check_huggingface,
    ),
]
