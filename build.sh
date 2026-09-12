#!/bin/bash
# Build script for PodcastNotesWT
# Creates a self-contained .app bundle and DMG using PyInstaller

set -e

echo "=== PodcastNotesWT Build ==="
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
pyinstaller PodcastNotesWT.spec --noconfirm

# Verify FFmpeg was bundled (PyInstaller puts it in Frameworks/bin/)
if [ -f "dist/PodcastNotesWT.app/Contents/Frameworks/bin/ffmpeg" ]; then
    echo "✓ FFmpeg bundled successfully"
else
    echo "⚠ Warning: FFmpeg may not be bundled correctly"
fi

# Sign the app before it goes into the DMG, so the DMG carries a signed app.
# A no-op with a clear message when there is no Developer ID certificate, and
# an unsigned build is still a working build.
scripts/sign_and_notarize.sh "dist/PodcastNotesWT.app"

# Create styled DMG
echo "Creating DMG..."
VERSION=$(tr -d '[:space:]' < VERSION)
DMG_NAME="PodcastNotesWT-${VERSION}.dmg"
rm -f "$DMG_NAME"

create-dmg \
    --volname "PodcastNotesWT" \
    --volicon "resources/icon.icns" \
    --background "resources/dmg_background.png" \
    --window-pos 200 120 \
    --window-size 660 400 \
    --icon-size 100 \
    --icon "PodcastNotesWT.app" 150 220 \
    --hide-extension "PodcastNotesWT.app" \
    --app-drop-link 510 220 \
    "$DMG_NAME" \
    "dist/PodcastNotesWT.app"

# And again for the DMG itself, which is the thing that actually gets sent.
scripts/sign_and_notarize.sh "dist/PodcastNotesWT.app" "$DMG_NAME"

echo ""
echo "=== Build Complete ==="
echo "App: dist/PodcastNotesWT.app"
echo "DMG: $DMG_NAME"
ls -lh "$DMG_NAME"
