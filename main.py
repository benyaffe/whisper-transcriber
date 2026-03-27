#!/usr/bin/env python3
"""
Whisper Transcription GUI
Transcribe audio/video files to VTT/TXT using faster-whisper
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from ui.main_window import MainWindow


def main():
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Whisper Transcriber")
    app.setOrganizationName("WhisperTranscriber")

    # Handle files dropped onto app icon (macOS)
    files_to_process = []
    for arg in sys.argv[1:]:
        if os.path.isfile(arg):
            files_to_process.append(arg)

    window = MainWindow()
    window.show()

    # If files were passed via command line (drag onto icon), queue them
    if files_to_process:
        window.queue_files(files_to_process)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
