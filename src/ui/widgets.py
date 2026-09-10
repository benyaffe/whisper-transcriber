"""
Widgets shared between the app's screens.

These used to live in main_window.py, but the trip screens need them too and
importing them from the window they are placed in would be circular.
"""

import os

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import QFileDialog, QFrame, QLabel, QTextBrowser, QVBoxLayout

from src.utils.file_utils import get_supported_extensions


class DropZone(QFrame):
    """Drag-and-drop zone with click-to-browse."""

    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(100)
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        self._set_default_style()

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label = QLabel("Drop audio/video files here\nor click to browse")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: #666; font-size: 14px;")
        layout.addWidget(self.label)

        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def _set_default_style(self):
        self.setStyleSheet(
            "DropZone { border: 2px dashed #888; border-radius: 10px; background-color: #f5f5f5; }"
        )

    def _set_hover_style(self):
        self.setStyleSheet(
            "DropZone { border: 2px solid #4a90d9; border-radius: 10px; background-color: #d0e4fc; }"
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            extensions = get_supported_extensions()
            filter_str = f"Media Files ({' '.join('*' + ext for ext in extensions)})"
            files, _ = QFileDialog.getOpenFileNames(self, "Select Audio/Video Files", "", filter_str)
            if files:
                self.files_dropped.emit(files)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._set_hover_style()

    def dragLeaveEvent(self, event):
        self._set_default_style()

    def dropEvent(self, event: QDropEvent):
        self._set_default_style()
        files = [url.toLocalFile() for url in event.mimeData().urls() if url.toLocalFile()]
        if files:
            self.files_dropped.emit(files)


class ClickablePreview(QTextBrowser):
    """Text preview with clickable timestamps."""

    timestamp_clicked = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setOpenLinks(False)
        self.anchorClicked.connect(self._handle_anchor)

        # Set document stylesheet for timestamp styling (this actually works in QTextBrowser)
        # Note: Use Menlo as primary - it's guaranteed on macOS
        self.document().setDefaultStyleSheet("""
            body {
                font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 13px;
            }
            a.ts {
                font-family: Menlo, Monaco, Courier;
                color: #2962ff;
                text-decoration: none;
            }
            .status {
                color: #666;
                font-style: italic;
            }
            .warning {
                color: #c90;
                font-style: italic;
            }
        """)

    def _handle_anchor(self, url: QUrl):
        """Handle timestamp link clicks."""
        if url.scheme() == "timestamp":
            try:
                # Use path to avoid QUrl host parsing issues with dots
                seconds = float(url.path().lstrip('/'))
                self.timestamp_clicked.emit(seconds)
            except ValueError:
                pass

