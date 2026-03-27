#!/usr/bin/env python3
"""
Whisper Transcription GUI
Transcribe audio/video files to VTT/TXT using faster-whisper
"""

import sys
import os

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from src.ui.main_window import MainWindow


class WhisperApp(QApplication):
    """Custom QApplication that raises window on activation (Cmd+Tab)."""

    def __init__(self, argv):
        super().__init__(argv)
        self.main_window = None

    def event(self, event):
        # Raise window when app is activated (e.g., Cmd+Tab)
        if event.type() == event.Type.ApplicationActivate:
            if self.main_window:
                self.main_window.raise_()
                self.main_window.activateWindow()
        return super().event(event)


def main():
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = WhisperApp(sys.argv)
    app.setApplicationName("Whisper Transcriber")
    app.setOrganizationName("WhisperTranscriber")

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

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
