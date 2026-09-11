"""
Widgets shared between the app's screens.

These used to live in main_window.py, but the trip screens need them too and
importing them from the window they are placed in would be circular.
"""

import os

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import QFileDialog, QFrame, QLabel, QTextBrowser, QVBoxLayout

from src.ui.theme import restyle, role
from src.utils.file_utils import get_supported_extensions


class DropZone(QFrame):
    """Drag-and-drop zone with click-to-browse."""

    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(92)
        self.setFrameShape(QFrame.Shape.NoFrame)
        # Styled through the theme rather than inline, because an inline
        # stylesheet on a widget silently wins over the application one and
        # this widget would then be the only thing that never follows a
        # theme change.
        role(self, "dropzone")
        self._set_default_style()

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(2)

        self.label = QLabel("Drop recordings here")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        role(self.label, "h2")
        layout.addWidget(self.label)

        self.sublabel = QLabel("or click to choose files")
        self.sublabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        role(self.sublabel, "muted")
        layout.addWidget(self.sublabel)

        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_compact(self, compact: bool):
        """Shrink once there is something in the list.

        A full-height target is the right thing to offer an empty screen and
        the wrong thing to keep once it has been used, where it just pushes
        the actual content down.

        Returns early when nothing has changed. It is called from the intake
        screen's refresh, which runs on every keystroke, and a style unpolish
        and repolish plus a layout invalidation per character is not free.
        """
        if getattr(self, "_compact", None) == compact:
            return
        self._compact = compact
        self.sublabel.setVisible(not compact)
        self.label.setText("Add more" if compact else "Drop recordings here")
        role(self.label, "muted" if compact else "h2")
        self.setMinimumHeight(44 if compact else 92)
        self.setMaximumHeight(44 if compact else 16777215)
        restyle(self.label)

    def _set_default_style(self):
        self.setProperty("hover", False)
        restyle(self)

    def _set_hover_style(self):
        self.setProperty("hover", True)
        restyle(self)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            extensions = get_supported_extensions()
            filter_str = f"Media Files ({' '.join('*' + ext for ext in extensions)})"
            files, _ = QFileDialog.getOpenFileNames(self, "Choose recordings", "", filter_str)
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

        # A QTextBrowser lays its content out with Qt's rich-text engine, so
        # the application stylesheet does not reach inside it and it needs its
        # own. Generated from the same tokens as everything else: the colours
        # used to be written out here by hand, which made this the one widget
        # that ignored any change to the theme.
        from src.ui.theme import document_stylesheet

        self.document().setDefaultStyleSheet(document_stylesheet())

    def _handle_anchor(self, url: QUrl):
        """Handle timestamp link clicks."""
        if url.scheme() == "timestamp":
            try:
                # Use path to avoid QUrl host parsing issues with dots
                seconds = float(url.path().lstrip('/'))
                self.timestamp_clicked.emit(seconds)
            except ValueError:
                pass

