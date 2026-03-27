#!/bin/bash
# Build script for Whisper Transcriber
# Creates a self-contained .app bundle and DMG with Applications shortcut

set -e

echo "=== Whisper Transcriber Build ==="
echo ""

# Ensure we're in the project directory
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

# Build the app
echo "Building app bundle (this may take a few minutes)..."
python setup.py py2app

# Verify FFmpeg was bundled
if [ -f "dist/Whisper Transcriber.app/Contents/Resources/bin/ffmpeg" ]; then
    echo "✓ FFmpeg bundled successfully"
else
    echo "⚠ Warning: FFmpeg not bundled. Recipients will need FFmpeg installed."
fi

# Create styled DMG with Applications shortcut and background
echo "Creating DMG..."
DMG_NAME="WhisperTranscriber-1.0.0.dmg"

# Remove old DMG if exists
rm -f "$DMG_NAME"

# Use create-dmg for professional installer look
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
echo ""
echo "To install: Open the DMG and drag the app to Applications."
