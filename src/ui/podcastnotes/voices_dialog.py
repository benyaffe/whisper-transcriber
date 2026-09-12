"""What the app remembers about people's voices, and how to make it stop.

Naming a speaker teaches the app that voice, so the same colleague across
several trips is suggested rather than asked about again. That is the feature
working. It is also the failure mode: a name given in haste, or given to the
wrong row, is then suggested confidently on every trip afterwards, and until
now there was no way to see that had happened, let alone undo it.

Two rules. Nothing here changes a document that has already been written: this
is only about what gets suggested next time. And forgetting is per person and
irreversible, so it asks, and it says what it is about to lose.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QVBoxLayout,
)

from src.ui.theme import role


class VoicesDialog(QDialog):
    """The remembered voices, and a way to forget one."""

    def __init__(self, library_path: str = "", parent=None):
        super().__init__(parent)
        self._library_path = library_path
        self.setWindowTitle("Remembered voices")
        self.resize(520, 420)
        self._build()
        self.refresh()

    def _build(self):
        layout = QVBoxLayout(self)

        blurb = QLabel(
            "Naming a speaker teaches the app that voice, so the same person is "
            "suggested on the next trip instead of being asked about again. "
            "Forgetting one only changes what gets suggested; it does not touch "
            "any write-up you have already made."
        )
        blurb.setWordWrap(True)
        role(blurb, "muted")
        layout.addWidget(blurb)

        self.list = QListWidget()
        # A long row scrolls sideways by default, which puts a horizontal
        # scrollbar under a list of names and hides the end of the one row
        # somebody is trying to read. Elided instead, with the whole line on
        # hover, since the tail is the trips and that is the part that says
        # whether this is the wrong voice.
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setTextElideMode(Qt.TextElideMode.ElideRight)
        layout.addWidget(self.list, 1)

        self.empty = QLabel("Nothing yet. Voices are remembered as you name them.")
        self.empty.setWordWrap(True)
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        role(self.empty, "muted")
        layout.addWidget(self.empty)

        row = QHBoxLayout()
        self.forget = QPushButton("Forget this voice")
        self.forget.clicked.connect(self._forget_selected)
        row.addWidget(self.forget)
        row.addStretch(1)
        self.done_button = QPushButton("Done")
        role(self.done_button, "primary")
        self.done_button.clicked.connect(self.accept)
        row.addWidget(self.done_button)
        layout.addLayout(row)

        self.list.currentItemChanged.connect(self._selection_changed)

    def refresh(self):
        from src.podcastnotes.speaker_library import Library

        self.list.clear()
        with Library(self._library_path) as library:
            entries = library.entries()

        for entry in entries:
            item = QListWidgetItem(_describe(entry))
            item.setToolTip(_describe(entry))
            item.setData(Qt.ItemDataRole.UserRole, entry["name"])
            self.list.addItem(item)

        self.list.setVisible(bool(entries))
        self.empty.setVisible(not entries)
        self._selection_changed()

    def _selection_changed(self, *args):
        self.forget.setEnabled(self.list.currentItem() is not None)

    def _forget_selected(self):
        """Ask first, and name what is about to go.

        Irreversible, and the samples are the only copy: the embeddings are not
        recoverable from the recordings without running the whole diarization
        again.
        """
        item = self.list.currentItem()
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)

        answer = QMessageBox.question(
            self, "Forget this voice?",
            f"{name} will stop being suggested on future trips. Anything "
            f"already written keeps the name.\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Ok:
            return

        from src.podcastnotes.speaker_library import Library

        with Library(self._library_path) as library:
            library.forget(name)
        self.refresh()


def _describe(entry: dict) -> str:
    """One line: who, how well known, and from where.

    The count on its own does not answer the question somebody opens this to
    ask, which is whether a given row is the wrong one. The trips do.
    """
    samples = entry["samples"]
    line = f"{entry['name']}, {samples} recording{'s' if samples != 1 else ''}"
    if entry["trips"]:
        line += f", from {_and_list(entry['trips'])}"
    return line


def _and_list(items: list) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"
