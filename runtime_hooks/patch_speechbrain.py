"""
Runtime hook to patch speechbrain's torchaudio backend check.

The list_audio_backends() function was removed in torchaudio 2.2+,
but speechbrain still tries to call it. This patch provides a fallback.
"""

import sys


def patch_torchaudio():
    """Add missing list_audio_backends function to torchaudio if needed."""
    try:
        import torchaudio

        if not hasattr(torchaudio, 'list_audio_backends'):
            # Provide a stub that returns available backends
            # In torchaudio 2.2+, backends are always available via ffmpeg
            def list_audio_backends():
                """Stub for removed torchaudio.list_audio_backends()"""
                return ['ffmpeg']

            torchaudio.list_audio_backends = list_audio_backends

    except ImportError:
        pass


# Apply patch immediately when this hook runs
patch_torchaudio()
