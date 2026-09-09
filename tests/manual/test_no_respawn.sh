#!/bin/bash
# Test that Whisper Transcriber doesn't respawn after quitting

APP_NAME="Whisper Transcriber"
APP_PATH="dist/Whisper Transcriber.app"

echo "=== Testing No-Respawn Behavior ==="
echo ""

# Kill any existing instances
echo "1. Killing any existing instances..."
pkill -f "Whisper Transcriber" 2>/dev/null
sleep 1

# Verify no instances running
if pgrep -f "Whisper Transcriber" > /dev/null; then
    echo "   FAIL: Could not kill existing instances"
    exit 1
fi
echo "   OK: No instances running"
echo ""

# Launch the app
echo "2. Launching app..."
open "$APP_PATH"
sleep 3

# Verify it started
if ! pgrep -f "Whisper Transcriber" > /dev/null; then
    echo "   FAIL: App did not start"
    exit 1
fi
PID_BEFORE=$(pgrep -f "Whisper Transcriber" | head -1)
echo "   OK: App running (PID: $PID_BEFORE)"
echo ""

# Quit the app using AppleScript (simulates Cmd+Q)
echo "3. Quitting app via AppleScript (Cmd+Q simulation)..."
osascript -e "tell application \"$APP_NAME\" to quit"
sleep 3

# Check if it respawned
echo "4. Checking for respawn..."
if pgrep -f "Whisper Transcriber" > /dev/null; then
    PID_AFTER=$(pgrep -f "Whisper Transcriber" | head -1)
    if [ "$PID_BEFORE" != "$PID_AFTER" ]; then
        echo "   FAIL: App respawned with new PID: $PID_AFTER"
        pkill -f "Whisper Transcriber"
        exit 1
    else
        echo "   FAIL: App still running (didn't quit properly)"
        pkill -f "Whisper Transcriber"
        exit 1
    fi
fi

echo "   OK: App did not respawn"
echo ""
echo "=== TEST PASSED ==="
