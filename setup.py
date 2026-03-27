"""
py2app setup script for Whisper Transcriber.

Build with:
    python setup.py py2app

For development testing:
    python setup.py py2app -A
"""

import os
import sys
import shutil
from setuptools import setup

# Fix recursion limit for large packages
sys.setrecursionlimit(5000)

APP = ['main.py']

# Find FFmpeg binaries to bundle (prefer static builds in resources/)
def get_ffmpeg_paths():
    """Locate ffmpeg and ffprobe binaries."""
    paths = []
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for binary in ['ffmpeg', 'ffprobe']:
        # First check for static builds in resources/ffmpeg/
        static_path = os.path.join(script_dir, 'resources', 'ffmpeg', binary)
        if os.path.exists(static_path):
            paths.append(static_path)
            continue

        # Fall back to system binaries
        for loc in ['/opt/homebrew/bin', '/usr/local/bin', '/usr/bin']:
            path = os.path.join(loc, binary)
            if os.path.exists(path):
                paths.append(path)
                break

    return paths

ffmpeg_binaries = get_ffmpeg_paths()
print(f"FFmpeg binaries to bundle: {ffmpeg_binaries}")

# Bundle FFmpeg in Resources/bin/
DATA_FILES = []
if ffmpeg_binaries:
    DATA_FILES.append(('bin', ffmpeg_binaries))

OPTIONS = {
    'argv_emulation': True,
    'iconfile': 'resources/icon.icns',
    'plist': {
        'CFBundleName': 'Whisper Transcriber',
        'CFBundleDisplayName': 'Whisper Transcriber',
        'CFBundleIdentifier': 'com.whispertranscriber.app',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.15',
        # Allow drag-drop of files onto app icon
        'CFBundleDocumentTypes': [
            {
                'CFBundleTypeName': 'Media File',
                'CFBundleTypeRole': 'Viewer',
                'LSHandlerRank': 'Alternate',
                'LSItemContentTypes': [
                    'public.movie',
                    'public.audio',
                    'public.mpeg-4',
                    'public.mp3',
                    'com.apple.quicktime-movie',
                    'public.avi',
                ],
            }
        ],
    },
    'packages': [
        'faster_whisper',
        'yt_dlp',
        'PyQt6',
        'ctranslate2',
        'huggingface_hub',
        'tokenizers',
        'av',
        'numpy',
        'onnxruntime',
    ],
    'includes': [
        'src',
        'src.ui',
        'src.core',
        'src.utils',
    ],
    'excludes': [
        'tkinter',
        'matplotlib',
        'scipy',
        'PIL',
        'pytest',
        'IPython',
        'jupyter',
        # Exclude heavy ML packages - diarization runs from source only
        'pyannote',
        'pyannote.audio',
        'torch',
        'torchaudio',
        'lightning',
        'pytorch_lightning',
    ],
    'frameworks': [],
}

setup(
    name='Whisper Transcriber',
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
