"""
The screen a trip starts on: what it was, and what was recorded.

Three pieces of copy here were actively misleading and are worth naming, since
the layout is built around fixing them.

"In the order they happened" asked for something the person should not have to
supply. Order is recoverable from the recordings themselves, and asking for it
put the cost of a solvable problem onto them.

The description looked like a request for full context, which made it feel
like homework and like the quality of the write-up depended on how much you
typed. It is a seed: a few names and the app searches Glean from there.

"Multiple speakers" named a mechanism rather than an outcome. What somebody
wants to know is whether the transcript will say who spoke.

Ordering is not asked for at all, and is not draggable either. It is worked
out after transcription from what was said. See RecordingList for why the
obvious shortcut does not work.
"""

import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton,
    QVBoxLayout, QWidget,
)

from src.podcastnotes.project import check_duration
from src.ui.theme import role
from src.ui.widgets import DropZone
from src.utils.file_utils import get_file_info, is_url, validate_input_file

PLACEHOLDER_DESCRIPTION = (
    "Ashford hospital tour with Sarah and the Ridgeline team"
)

DESCRIPTION_HINT = (
    "A sentence is plenty. Names and places here are search terms: the app "
    "looks them up in Glean and finds the rest of the context itself."
)

SPEAKERS_HINT = (
    "The transcript says who spoke, instead of Speaker 1 and Speaker 2. "
    "Adds roughly a third to the running time."
)


class RecordingList(QListWidget):
    """The recordings in this trip, in no particular order.

    Deliberately not reorderable. Which meeting came first is worked out after
    transcription, by reading what was said, because that is the only source
    that actually knows.

    File timestamps look like an easier answer and are not. On the Ashford
    recordings every file reports a creation time within four seconds of the
    others, because that is when they were copied off the device rather than
    when they were recorded. Sorting by that would have produced a confident
    wrong order, which is worse than no order at all.
    """

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.changed.connect(self._fit_to_contents)

    # Two recordings should not occupy the same space as ten. Beyond this many
    # the list scrolls instead, so a trip with thirty files does not push the
    # Start button off the bottom of the screen.
    MAX_VISIBLE_ROWS = 7

    # Used only when the list is empty and there is no row to measure.
    FALLBACK_ROW_HEIGHT = 32

    def _fit_to_contents(self):
        """Make room for the rows there actually are.

        The height used to be `rows * 30 + 12` against a real row of 32 px, so
        it under-shot at every size and a scrollbar appeared from the first
        item. Two recordings looked exactly like a cramped scrolling box, which
        read as the list not growing at all.

        Measured rather than assumed: the row height comes from the theme's
        item padding and the body font size, so a change to either would put a
        constant here back out of step.
        """
        rows = min(max(self.count(), 1), self.MAX_VISIBLE_ROWS)
        row_height = self.sizeHintForRow(0) if self.count() else self.FALLBACK_ROW_HEIGHT
        chrome = 2 * self.frameWidth() + 2 * self.spacing() + 8
        self.setFixedHeight(rows * row_height + chrome)

    def add(self, source: str):
        if source in self.sources():
            return False
        item = QListWidgetItem(self._label(source))
        item.setData(Qt.ItemDataRole.UserRole, source)
        item.setToolTip(source)
        self.addItem(item)
        self.changed.emit()
        self._fit_to_contents()
        return True

    def remove_selected(self):
        for item in self.selectedItems():
            self.takeItem(self.row(item))
        self.changed.emit()
        self._fit_to_contents()

    def sources(self) -> list[str]:
        return [
            self.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.count())
        ]

    @staticmethod
    def _label(source: str) -> str:
        return source if is_url(source) else os.path.basename(source)


class Section(QFrame):
    """A titled card. Gives each question its own space to sit in."""

    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        role(self, "card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(18, 16, 18, 16)
        self.body.setSpacing(8)

        heading = role(QLabel(title), "h2")
        self.body.addWidget(heading)
        if subtitle:
            hint = role(QLabel(subtitle), "muted")
            hint.setWordWrap(True)
            self.body.addWidget(hint)


class IntakeView(QWidget):
    """Name the trip, add the recordings, start."""

    start_requested = pyqtSignal(str, str, list, bool)
    settings_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # Read from the Keychain on first use, not on every keystroke.
        self._token_cache = None
        self._build()
        self._refresh()

    # --- construction ---------------------------------------------------------

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 24, 28, 20)
        outer.setSpacing(16)

        outer.addWidget(role(QLabel("New trip"), "h1"))

        outer.addWidget(self._about_section())
        outer.addWidget(self._recordings_section())
        # The slack goes here, between the last card and the footer, so the
        # cards stay the size of what is in them.
        outer.addStretch()
        outer.addLayout(self._footer())

    def _about_section(self) -> QWidget:
        section = Section(
            "What was it?",
            DESCRIPTION_HINT,
        )
        self.name_input = QLineEdit()
        role(self.name_input, "title")
        self.name_input.setPlaceholderText(PLACEHOLDER_DESCRIPTION)
        self.name_input.textChanged.connect(self._refresh)
        section.body.addWidget(self.name_input)

        self.description_input = QPlainTextEdit()
        self.description_input.setPlaceholderText(
            "Anything else worth knowing. Optional."
        )
        self.description_input.setFixedHeight(64)
        section.body.addWidget(self.description_input)
        return section

    def _recordings_section(self) -> QWidget:
        section = Section("Recordings")

        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self.add_sources)
        section.body.addWidget(self.drop_zone)

        self.recordings = RecordingList()
        self.recordings.changed.connect(self._refresh)
        section.body.addWidget(self.recordings)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText(
            "Paste a link to a recording on YouTube, Vimeo or Drive"
        )
        self.url_input.returnPressed.connect(self._add_url)
        row.addWidget(self.url_input, 1)

        add_url = QPushButton("Add link")
        add_url.clicked.connect(self._add_url)
        row.addWidget(add_url)

        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.recordings.remove_selected)
        row.addWidget(self.remove_button)
        section.body.addLayout(row)
        return section

    def _footer(self) -> QHBoxLayout:
        footer = QHBoxLayout()
        footer.setSpacing(14)

        speakers = QVBoxLayout()
        speakers.setSpacing(1)
        self.speakers_checkbox = QCheckBox("Name the speakers")
        self.speakers_checkbox.setChecked(True)
        self.speakers_checkbox.toggled.connect(self._refresh)
        speakers.addWidget(self.speakers_checkbox)
        speakers.addWidget(role(QLabel(SPEAKERS_HINT), "faint"))
        footer.addLayout(speakers)

        footer.addStretch()

        right = QVBoxLayout()
        right.setSpacing(4)
        self.hint = role(QLabel(""), "muted")
        self.hint.setWordWrap(True)
        self.hint.setAlignment(Qt.AlignmentFlag.AlignRight)
        right.addWidget(self.hint)
        footer.addLayout(right)

        self.start_button = QPushButton("Start")
        role(self.start_button, "primary")
        self.start_button.setDefault(True)
        self.start_button.clicked.connect(self._start)
        footer.addWidget(self.start_button, 0, Qt.AlignmentFlag.AlignBottom)
        return footer

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
                self, "Not a link",
                "That does not look like a link. It should start with http.",
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
                "Naming speakers needs a HuggingFace token, which is not set up "
                "yet. Add one in Accounts, or untick Name the speakers."
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

    def _hint_text(self) -> str:
        """What to say under the form, computed once.

        `self._problem() or self._advisory()` reads well and costs double: on a
        valid form the problem check runs the duration scan, finds nothing to
        complain about, and then the advisory runs exactly the same scan again.
        With that scan spawning an ffprobe per recording, the difference is two
        subprocesses per file rather than one, on every keystroke.
        """
        if not self.name_input.text().strip():
            return "Give the trip a name."
        if not self.recordings.sources():
            return "Add at least one recording."
        if self.speakers_checkbox.isChecked() and not self._hf_token():
            return (
                "Naming speakers needs a HuggingFace token, which is not set up "
                "yet. Add one in Accounts, or untick Name the speakers."
            )
        if not self.speakers_checkbox.isChecked():
            return ""
        # One scan, and both answers come out of it.
        allowed, message = check_duration(*self._duration())
        return message

    def _hf_token(self) -> str:
        """The stored token, read once rather than once per keystroke.

        `get_hf_token` is a Keychain round trip. On the validation path that put
        one macOS IPC call between every character typed and the character
        appearing. Nothing can change it while this screen has focus except the
        Accounts dialog, which calls `forget_hf_token` on the way out.
        """
        if self._token_cache is None:
            from src.core.config import get_hf_token

            self._token_cache = get_hf_token()
        return self._token_cache

    def forget_hf_token(self):
        """Drop the cached token, for when somebody has just changed it."""
        self._token_cache = None

    def clear(self):
        """Empty the form for a new trip.

        Nothing did this, so "Start another trip" landed on a form still
        holding the last trip's name, description and recordings. Pressing
        Start from there makes a second trip of the same files, which is
        neither what anybody meant nor obviously wrong until it has run.

        The speaker checkbox is deliberately left as it was: whether somebody
        wants speaker names is a preference that holds across trips, unlike the
        recordings, which never do.
        """
        self.name_input.clear()
        self.description_input.clear()
        self.url_input.clear()
        self.recordings.clear()
        self.recordings.changed.emit()
        self._refresh()

    def _refresh(self):
        # The button stays enabled with no name or no recordings, because a
        # disabled button with no explanation is the least helpful thing a form
        # can do. Pressing it says what is wrong.
        has_any = bool(self.recordings.sources())
        self.remove_button.setEnabled(has_any)
        self.remove_button.setVisible(has_any)
        self.recordings.setVisible(has_any)
        self.drop_zone.set_compact(has_any)
        if not has_any:
            self.hint.setText("")
            return
        self.hint.setText(self._hint_text())

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
        return "Accounts" in problem

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
