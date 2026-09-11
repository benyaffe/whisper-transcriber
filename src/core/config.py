"""
Persisted app configuration: the HuggingFace token and the speaker-ID toggle.

Lives in core rather than ui because src/core/transcriber.py needs the same
values, and importing them from the settings *dialog* made core depend on ui.
That was the one edge closing an import cycle
(main_window -> transcriber -> settings_dialog -> diarization), which only
avoided deadlocking because two of its legs happened to be function-local.

keyring and QSettings are imported inside the functions, matching the
convention used for every heavy import in this codebase, so importing this
module costs nothing and pulls in no Qt.
"""

# Keychain coordinates. The service name is also what macOS shows the user in
# Keychain Access, so changing it orphans every existing stored token.
#
# The app is called PodcastNotesWT now. This deliberately still says
# WhisperTranscriber, and so do the QSettings names below. They are storage
# keys, not labels: nobody sees them, and renaming them would silently sign
# everybody out and reset their preferences to buy nothing at all.
KEYRING_SERVICE = "WhisperTranscriber"
KEYRING_HF_TOKEN = "hf_token"

SETTINGS_SPEAKER_ID_ENABLED = "speaker_id_enabled"

# Deliberately hardcoded, and deliberately NOT the application identity set in
# main.py. QSettings("WhisperTranscriber", "WhisperTranscriber") resolves to
# ~/Library/Preferences/com.whispertranscriber.WhisperTranscriber.plist, which
# is where every existing install has its settings.
#
# main.py now sets the application name to "PodcastNotesWT", so a
# default-constructed QSettings() would resolve to
# com.whispertranscriber.PodcastNotesWT.plist: a different file, and everyone's
# preferences silently back to their defaults. Do not "tidy" this to
# QSettings(). The reason has survived one rename already and the earlier
# version of this comment described the pre-rename name.
_SETTINGS_ORG = "WhisperTranscriber"
_SETTINGS_APP = "WhisperTranscriber"


def _settings():
    from PyQt6.QtCore import QSettings

    return QSettings(_SETTINGS_ORG, _SETTINGS_APP)


def get_hf_token() -> str:
    """Retrieve stored HuggingFace token."""
    import keyring

    try:
        token = keyring.get_password(KEYRING_SERVICE, KEYRING_HF_TOKEN)
        return token or ""
    except Exception:
        return ""


def save_hf_token(token: str):
    """Save HuggingFace token to keychain."""
    import keyring

    try:
        if token:
            keyring.set_password(KEYRING_SERVICE, KEYRING_HF_TOKEN, token)
        else:
            try:
                keyring.delete_password(KEYRING_SERVICE, KEYRING_HF_TOKEN)
            except keyring.errors.PasswordDeleteError:
                pass
    except Exception as e:
        raise RuntimeError(f"Failed to save token: {e}")


def is_speaker_id_enabled() -> bool:
    """Whether to name the speakers. On unless somebody turned it off.

    The default has to be True, and it used to be False. Two screens disagreed
    about it, and a cold-start rehearsal found what that does to somebody new:

      - The setup checklist reads this value. With it False, the HuggingFace
        row reported "Off, so transcripts will not name speakers" and went
        green, so the whole list said the app was ready.
      - The intake screen's "Name the speakers" box is ticked by default and
        does not read this value at all, so the same person then hit "Naming
        speakers needs a HuggingFace token" and could not start.

    A checklist that says everything is fine and then a screen that refuses to
    start is the exact failure that checklist exists to prevent. Only fresh
    installs are affected, because anybody who has set this once has a stored
    value and never sees the default.
    """
    return _settings().value(SETTINGS_SPEAKER_ID_ENABLED, True, type=bool)


def set_speaker_id_enabled(enabled: bool):
    """Set speaker identification enabled state."""
    _settings().setValue(SETTINGS_SPEAKER_ID_ENABLED, enabled)
