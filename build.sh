#!/bin/bash
# Build script for Whisper Transcriber
# Creates a self-contained .app bundle and DMG using PyInstaller

set -e

echo "=== Whisper Transcriber Build ==="
echo ""

cd "$(dirname "$0")"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist *.dmg dmg_staging

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Download static ffmpeg if not present
if [ ! -f "resources/ffmpeg/ffmpeg" ]; then
    echo "Downloading static ffmpeg binaries..."
    python scripts/download_ffmpeg.py
fi

# Build with PyInstaller
echo "Building app bundle with PyInstaller (this may take several minutes)..."
pyinstaller WhisperTranscriber.spec --noconfirm

# Verify FFmpeg was bundled (PyInstaller puts it in Frameworks/bin/)
if [ -f "dist/Whisper Transcriber.app/Contents/Frameworks/bin/ffmpeg" ]; then
    echo "✓ FFmpeg bundled successfully"
else
    echo "⚠ Warning: FFmpeg may not be bundled correctly"
fi

# Create styled DMG
echo "Creating DMG..."
DMG_NAME="WhisperTranscriber-1.0.1.dmg"
rm -f "$DMG_NAME"

create-dmg \
    --volname "Whisper Transcriber" \
    --volicon "resources/icon.icns" \
    --background "resources/dmg_background.png" \
    --window-pos 200 120 \
    --window-size 660 400 \
    --icon-size 100 \
    --icon "Whisper Transcriber.app" 150 220 \
    --hide-extension "Whisper Transcriber.app" \
    --app-drop-link 510 220 \
    "$DMG_NAME" \
    "dist/Whisper Transcriber.app"

echo ""
echo "=== Build Complete ==="
echo "App: dist/Whisper Transcriber.app"
echo "DMG: $DMG_NAME"
ls -lh "$DMG_NAME"
