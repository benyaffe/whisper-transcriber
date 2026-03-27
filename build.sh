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

# Build the app
echo "Building app bundle (this may take a few minutes)..."
python setup.py py2app

# Verify FFmpeg was bundled
if [ -f "dist/Whisper Transcriber.app/Contents/Resources/bin/ffmpeg" ]; then
    echo "✓ FFmpeg bundled successfully"
else
    echo "⚠ Warning: FFmpeg not bundled. Recipients will need FFmpeg installed."
fi

# Create DMG with Applications shortcut
echo "Creating DMG..."
DMG_NAME="WhisperTranscriber-1.0.0.dmg"

# Create staging folder with app and Applications alias
mkdir -p dmg_staging
cp -R "dist/Whisper Transcriber.app" dmg_staging/
ln -s /Applications dmg_staging/Applications

# Create the DMG
hdiutil create \
    -volname "Whisper Transcriber" \
    -srcfolder dmg_staging \
    -ov \
    -format UDZO \
    "$DMG_NAME"

# Clean up staging folder
rm -rf dmg_staging

echo ""
echo "=== Build Complete ==="
echo "App: dist/Whisper Transcriber.app"
echo "DMG: $DMG_NAME"
echo ""
echo "To install: Open the DMG and drag the app to Applications."
