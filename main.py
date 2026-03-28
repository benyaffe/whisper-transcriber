#!/usr/bin/env python3
"""
Whisper Transcription GUI
Transcribe audio/video files to VTT/TXT using faster-whisper
"""

import sys
import os
import fcntl
import tempfile
import signal
import atexit

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from src.ui.main_window import MainWindow


# Single instance lock
_lock_file = None
_is_quitting = False


def acquire_single_instance_lock():
    """Prevent multiple instances of the app from running."""
    global _lock_file
    lock_path = os.path.join(tempfile.gettempdir(), 'whisper_transcriber.lock')

    try:
        _lock_file = open(lock_path, 'w')
        fcntl.flock(_lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_file.write(str(os.getpid()))
        _lock_file.flush()
        return True
    except (IOError, OSError):
        # Another instance is running
        return False


def release_lock():
    """Release the single instance lock."""
    global _lock_file
    if _lock_file:
        try:
            fcntl.flock(_lock_file.fileno(), fcntl.LOCK_UN)
            _lock_file.close()
            _lock_file = None
        except Exception:
            pass


def cleanup_and_exit(*args):
    """Clean up and exit without respawning."""
    global _is_quitting
    _is_quitting = True
    release_lock()
    # Force exit without triggering any respawn mechanisms
    os._exit(0)


class WhisperApp(QApplication):
    """Custom QApplication that raises window on activation (Cmd+Tab)."""

    def __init__(self, argv):
        super().__init__(argv)
        self.main_window = None
        self._quitting = False

        # Connect aboutToQuit to ensure clean shutdown
        self.aboutToQuit.connect(self._on_about_to_quit)

    def _on_about_to_quit(self):
        """Handle app quit - release locks, cleanup."""
        global _is_quitting
        self._quitting = True
        _is_quitting = True
        release_lock()

    def event(self, event):
        global _is_quitting
        # Don't process events if we're quitting
        if _is_quitting:
            return True
        # Raise window when app is activated (e.g., Cmd+Tab)
        if event.type() == event.Type.ApplicationActivate:
            if self.main_window and not self._quitting:
                self.main_window.raise_()
                self.main_window.activateWindow()
        return super().event(event)


def main():
    global _is_quitting

    # Prevent multiple instances
    if not acquire_single_instance_lock():
        print("Whisper Transcriber is already running.")
        sys.exit(0)

    # Register cleanup handlers to prevent respawn
    atexit.register(release_lock)
    signal.signal(signal.SIGTERM, cleanup_and_exit)
    signal.signal(signal.SIGINT, cleanup_and_exit)

    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = WhisperApp(sys.argv)
    app.setApplicationName("Whisper Transcriber")
    app.setOrganizationName("WhisperTranscriber")

    # Ensure app quits when window is closed (prevent respawn)
    app.setQuitOnLastWindowClosed(True)

    # Handle files dropped onto app icon (macOS)
    files_to_process = []
    for arg in sys.argv[1:]:
        if os.path.isfile(arg):
            files_to_process.append(arg)

    window = MainWindow()
    app.main_window = window  # For Cmd+Tab window raising
    window.show()

    # If files were passed via command line (drag onto icon), queue them
    if files_to_process:
        window.queue_files(files_to_process)

    ret = app.exec()

    # Mark as quitting and cleanup
    _is_quitting = True
    release_lock()

    # Use os._exit to prevent any respawn mechanisms
    os._exit(ret)


if __name__ == "__main__":
    main()
