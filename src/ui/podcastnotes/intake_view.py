"""
The screen a trip starts on: what it is about, and what was recorded.

The description is not decoration. It is what later stages use to look up the
people, organisations and acronyms that a transcript of a real meeting is full
of, so the form asks for it before anything runs rather than after.
"""

import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

from src.podcastnotes.project import check_duration
from src.utils.file_utils import get_file_info, is_url, validate_input_file
from src.ui.widgets import DropZone

PLACEHOLDER_DESCRIPTION = (
    "Who was there, where you went, what it was about. Names and organisations "
    "here are what let the app fix them up in the transcript later."
)


class RecordingList(QListWidget):
    """The recordings, in the order they will be joined.

    Drag to reorder: which meeting came first is a decision only the operator
    can make, and alphabetical order is not it.
    """

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setMinimumHeight(160)
        self.model().rowsMoved.connect(lambda *_: self.changed.emit())

    def add(self, source: str):
        if source in self.sources():
            return False
        item = QListWidgetItem(self._label(source))
        item.setData(Qt.ItemDataRole.UserRole, source)
        item.setToolTip(source)
        self.addItem(item)
        self.changed.emit()
        return True

    def remove_selected(self):
        for item in self.selectedItems():
            self.takeItem(self.row(item))
        self.changed.emit()

    def sources(self) -> list[str]:
        return [
            self.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.count())
        ]

    @staticmethod
    def _label(source: str) -> str:
        return source if is_url(source) else os.path.basename(source)


class IntakeView(QWidget):
    """Name the trip, add the recordings, start."""

    start_requested = pyqtSignal(str, str, list, bool)  # name, description, sources, speakers
    settings_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()
        self._refresh_start_button()

    # --- construction ---------------------------------------------------------

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        layout.addWidget(QLabel("<b>What is this?</b>"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Trip or session name, e.g. Ashford hospital tour")
        self.name_input.textChanged.connect(self._refresh_start_button)
        layout.addWidget(self.name_input)

        self.description_input = QPlainTextEdit()
        self.description_input.setPlaceholderText(PLACEHOLDER_DESCRIPTION)
        self.description_input.setMaximumHeight(90)
        layout.addWidget(self.description_input)

        layout.addWidget(QLabel("<b>Recordings</b>, in the order they happened"))
        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self.add_sources)
        layout.addWidget(self.drop_zone)

        url_row = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("...or paste a YouTube, Vimeo or other URL")
        self.url_input.returnPressed.connect(self._add_url)
        url_row.addWidget(self.url_input)
        add_url = QPushButton("Add URL")
        add_url.clicked.connect(self._add_url)
        url_row.addWidget(add_url)
        layout.addLayout(url_row)

        self.recordings = RecordingList()
        self.recordings.changed.connect(self._refresh_start_button)
        layout.addWidget(self.recordings, 1)

        controls = QHBoxLayout()
        remove = QPushButton("Remove")
        remove.clicked.connect(self.recordings.remove_selected)
        controls.addWidget(remove)
        controls.addStretch()

        self.speakers_checkbox = QCheckBox("Multiple speakers")
        self.speakers_checkbox.setChecked(True)
        self.speakers_checkbox.toggled.connect(self._refresh_start_button)
        self.speakers_checkbox.setToolTip(
            "Work out who said what. Turn off for a single-speaker recording; "
            "it is about a tenth of the running time."
        )
        controls.addWidget(self.speakers_checkbox)
        layout.addLayout(controls)

        self.hint = QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: #666;")
        layout.addWidget(self.hint)

        self.start_button = QPushButton("Start")
        self.start_button.setDefault(True)
        self.start_button.clicked.connect(self._start)
        layout.addWidget(self.start_button)

    # --- adding recordings ----------------------------------------------------

    def has_input(self) -> bool:
        """Whether somebody has started describing a trip.

        The setup checklist uses this to decide whether it may take the screen:
        interrupting an empty form is helpful, interrupting a half-written one
        is not.
        """
        return bool(
            self.name_input.text().strip()
            or self.description_input.toPlainText().strip()
            or self.recordings.sources()
        )

    def add_sources(self, paths: list):
        """Add local files, rejecting anything that is not usable audio."""
        rejected = []
        for path in paths:
            ok, message = validate_input_file(path)
            if ok:
                self.recordings.add(path)
            else:
                rejected.append(message)

        if rejected:
            QMessageBox.warning(
                self, "Some files skipped", "\n".join(f"• {m}" for m in rejected[:5])
            )

    def _add_url(self):
        url = self.url_input.text().strip()
        if not url:
            return
        if not is_url(url):
            QMessageBox.warning(
                self, "Not a URL", "That does not look like a link. It should start with http."
            )
            return
        self.recordings.add(url)
        self.url_input.clear()

    # --- validation -----------------------------------------------------------

    def _duration(self) -> tuple[float, bool]:
        """Total length of the recordings, and whether that is only a floor.

        A URL's length is unknown until it has been downloaded, so a trip
        containing one can turn out longer than it looks here.
        """
        total = 0.0
        estimated = False
        for source in self.recordings.sources():
            if is_url(source):
                estimated = True
                continue
            total += float(get_file_info(source).get("duration") or 0.0)
        return total, estimated

    def _problem(self) -> str:
        """Why the trip cannot start yet, or empty if it can."""
        if not self.name_input.text().strip():
            return "Give the trip a name."
        if not self.recordings.sources():
            return "Add at least one recording."
        if self.speakers_checkbox.isChecked() and not self._hf_token():
            return (
                "Working out who is speaking needs a HuggingFace token, which is not "
                "set up yet. Add one in Settings, or untick Multiple speakers."
            )
        # Only speaker identification has the memory problem, so a trip that is
        # not doing it has no reason to be capped.
        if self.speakers_checkbox.isChecked():
            allowed, message = check_duration(*self._duration())
            if not allowed:
                return message
        return ""

    def _advisory(self) -> str:
        """Something worth saying that is not a reason to stop."""
        if not self.recordings.sources() or not self.speakers_checkbox.isChecked():
            return ""
        allowed, message = check_duration(*self._duration())
        return message if allowed else ""

    @staticmethod
    def _hf_token() -> str:
        from src.core.config import get_hf_token

        return get_hf_token()

    def _refresh_start_button(self):
        # The button stays enabled with no name or no recordings, because a
        # disabled button with no explanation is the least helpful thing a form
        # can do. Pressing it says what is wrong.
        if not self.recordings.sources():
            self.hint.setText("")
            return
        self.hint.setText(self._problem() or self._advisory())

    def _report_problem(self, problem: str):
        """Tell the operator why nothing happened.

        Split out from _start so that what a trip requires can be tested
        without a modal dialog blocking the test runner forever.
        """
        box = QMessageBox(QMessageBox.Icon.Information, "Not ready yet", problem, parent=self)
        if self._needs_token(problem):
            open_settings = box.addButton("Open Settings", QMessageBox.ButtonRole.AcceptRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            if box.clickedButton() is open_settings:
                self.settings_requested.emit()
        else:
            box.exec()

    @staticmethod
    def _needs_token(problem: str) -> bool:
        return "Settings" in problem

    def _start(self):
        problem = self._problem()
        if problem:
            self._report_problem(problem)
            return

        self.start_requested.emit(
            self.name_input.text().strip(),
            self.description_input.toPlainText().strip(),
            self.recordings.sources(),
            self.speakers_checkbox.isChecked(),
        )
