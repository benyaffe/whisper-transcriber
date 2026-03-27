# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Whisper Transcriber
"""

import os
import sys

block_cipher = None

# Get paths
project_dir = os.path.dirname(os.path.abspath(SPEC))
resources_dir = os.path.join(project_dir, 'resources')
ffmpeg_dir = os.path.join(resources_dir, 'ffmpeg')

# Data files to bundle
datas = [
    (os.path.join(ffmpeg_dir, 'ffmpeg'), 'bin'),
    (os.path.join(ffmpeg_dir, 'ffprobe'), 'bin'),
]

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'src',
    'src.ui',
    'src.ui.main_window',
    'src.core',
    'src.core.transcriber',
    'src.core.downloader',
    'src.utils',
    'src.utils.file_utils',
    'src.utils.error_handler',
    # PyQt6
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.QtMultimedia',
    # Whisper/ML
    'faster_whisper',
    'ctranslate2',
    'huggingface_hub',
    'tokenizers',
    # Torch
    'torch',
    'torchaudio',
    # Pyannote
    'pyannote.audio',
    'pyannote.audio.pipelines',
    'pyannote.audio.pipelines.speaker_diarization',
    'pyannote.core',
    'pyannote.pipeline',
    'pyannote.database',
    'pyannote.metrics',
    # Lightning
    'pytorch_lightning',
    'lightning',
    'lightning.fabric',
    'lightning.pytorch',
    # Other deps
    'yt_dlp',
    'av',
    'numpy',
    'onnxruntime',
    'scipy',
    'sklearn',
    'sklearn.cluster',
]

a = Analysis(
    ['main.py'],
    pathex=[project_dir],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'IPython',
        'jupyter',
        'pytest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Whisper Transcriber',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Whisper Transcriber',
)

app = BUNDLE(
    coll,
    name='Whisper Transcriber.app',
    icon=os.path.join(resources_dir, 'icon.icns'),
    bundle_identifier='com.whispertranscriber.app',
    info_plist={
        'CFBundleName': 'Whisper Transcriber',
        'CFBundleDisplayName': 'Whisper Transcriber',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.15',
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
)
