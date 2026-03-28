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
venv_dir = os.path.join(project_dir, 'venv')

# Find faster_whisper assets (contains silero VAD model)
import glob
faster_whisper_assets = glob.glob(os.path.join(venv_dir, 'lib', 'python*', 'site-packages', 'faster_whisper', 'assets'))
fw_assets_dir = faster_whisper_assets[0] if faster_whisper_assets else None

# Find pyannote telemetry config
pyannote_telemetry = glob.glob(os.path.join(venv_dir, 'lib', 'python*', 'site-packages', 'pyannote', 'audio', 'telemetry'))
pyannote_telemetry_dir = pyannote_telemetry[0] if pyannote_telemetry else None

# Use PyInstaller's collect functions to properly bundle packages
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

# Collect speechbrain properly as a package (not just data files)
speechbrain_datas, speechbrain_binaries, speechbrain_hiddenimports = collect_all('speechbrain')

# Collect pyannote packages
pyannote_datas, pyannote_binaries, pyannote_hiddenimports = collect_all('pyannote')

# Collect torchcodec (required by pyannote.audio 4.x for audio decoding)
torchcodec_datas, torchcodec_binaries, torchcodec_hiddenimports = collect_all('torchcodec')

# Collect asteroid_filterbanks
asteroid_datas = collect_data_files('asteroid_filterbanks')

# Data files to bundle
datas = [
    (os.path.join(ffmpeg_dir, 'ffmpeg'), 'bin'),
    (os.path.join(ffmpeg_dir, 'ffprobe'), 'bin'),
]

# Add faster_whisper assets (silero VAD model)
if fw_assets_dir and os.path.exists(fw_assets_dir):
    datas.append((fw_assets_dir, 'faster_whisper/assets'))

# Add pyannote telemetry config
if pyannote_telemetry_dir and os.path.exists(pyannote_telemetry_dir):
    datas.append((pyannote_telemetry_dir, 'pyannote/audio/telemetry'))

# Merge collected data files
datas += speechbrain_datas + pyannote_datas + asteroid_datas + torchcodec_datas

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'src',
    'src.ui',
    'src.ui.main_window',
    'src.ui.settings_dialog',
    'src.core',
    'src.core.transcriber',
    'src.core.downloader',
    'src.core.diarization',
    'src.core.checkpoint',
    'src.utils',
    'src.utils.file_utils',
    'src.utils.error_handler',
    'src.utils.logger',
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
    # Pyannote (optional, for HF token users)
    'pyannote.audio',
    'pyannote.audio.pipelines',
    'pyannote.audio.pipelines.speaker_diarization',
    'pyannote.core',
    'pyannote.pipeline',
    'pyannote.database',
    'pyannote.metrics',
    # SpeechBrain (default diarization)
    'speechbrain',
    'speechbrain.inference',
    'speechbrain.inference.speaker',
    'speechbrain.utils',
    'speechbrain.utils.fetching',
    'hyperpyyaml',
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
    'scipy.io',
    'scipy.io.wavfile',
    'soundfile',
    'sklearn',
    'sklearn.cluster',
    'sklearn.metrics',
    # Keyring for secure token storage
    'keyring',
    'keyring.backends',
    'keyring.backends.macOS',
    # psutil for memory monitoring
    'psutil',
]

# Merge collected hidden imports
hiddenimports += speechbrain_hiddenimports + pyannote_hiddenimports + torchcodec_hiddenimports

# Collect binaries
binaries = speechbrain_binaries + pyannote_binaries + torchcodec_binaries

a = Analysis(
    ['main.py'],
    pathex=[project_dir],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(project_dir, 'runtime_hooks', 'patch_speechbrain.py')],
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
        # Prevent respawning on quit
        'NSSupportsAutomaticTermination': True,
        'NSSupportsSuddenTermination': True,
        'LSUIElement': False,  # Not a background app
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
