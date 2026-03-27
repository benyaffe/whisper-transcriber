"""
py2app setup script for Whisper Transcriber.

Build with:
    python setup.py py2app

For development testing:
    python setup.py py2app -A
"""

import os
import shutil
from setuptools import setup

APP = ['main.py']

# Find FFmpeg binaries to bundle
def get_ffmpeg_paths():
    """Locate ffmpeg and ffprobe binaries."""
    paths = []
    for binary in ['ffmpeg', 'ffprobe']:
        # Check common locations
        for loc in ['/opt/homebrew/bin', '/usr/local/bin', '/usr/bin']:
            path = os.path.join(loc, binary)
            if os.path.exists(path):
                paths.append(path)
                break
    return paths

ffmpeg_binaries = get_ffmpeg_paths()

# Bundle FFmpeg in Resources/bin/
DATA_FILES = []
if ffmpeg_binaries:
    DATA_FILES.append(('bin', ffmpeg_binaries))

OPTIONS = {
    'argv_emulation': True,
    'iconfile': None,  # Add icon path here: 'resources/icon.icns'
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
