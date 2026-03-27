"""
py2app setup script for Whisper Transcriber.

Build with:
    python setup.py py2app

For development testing:
    python setup.py py2app -A
"""

from setuptools import setup

APP = ['main.py']
DATA_FILES = []

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
    ],
}

setup(
    name='Whisper Transcriber',
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
