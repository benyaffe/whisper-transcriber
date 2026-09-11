"""
The screen the write-up happens on, and the one place it asks anything.

Three panes behind one widget: waiting, naming the speakers, and the finished
documents. They are panes rather than separate screens because the write-up is
one continuous thing from the user's point of view, and bouncing between
top-level screens for a job that takes a few minutes reads as the app losing
its place.

**Naming a speaker is a decision, so nothing here pre-confirms one.** The
suggestion arrives filled in, from Claude and from the voice library, and the
person either accepts it or types over it. A speaker they leave blank stays
unnamed rather than quietly becoming whoever was suggested, which is why the
button says what it does and the count of unnamed speakers is on screen next
to it.

Each speaker row carries the clip `snippet.py` extracts, because the reliable
way to tell two colleagues apart is to hear six seconds of each, and the
sample lines beside it, because the reliable way to tell which one is the
hardware engineer is to read what they talk about.
"""

import os

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit, QProgressBar, QPushButton,
    QScrollArea, QStackedWidget, QTextBrowser, QVBoxLayout, QWidget,
)

from src.ui.theme import role

# Long enough to recognise a colleague, short enough that reviewing four of
# them is not a chore. Matches what snippet.py already cuts.
CLIP_SECONDS = 6.0


class SpeakerRow(QFrame):
    """One voice: play it, read it, name it."""

    play_requested = pyqtSignal(float)

    def __init__(self, voice, suggestion=None, parent=None):
        super().__init__(parent)
        self.voice = voice
        role(self, "card")
        self._build(suggestion)

    def _build(self, suggestion):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        top = QHBoxLayout()
        label = QLabel(self.voice.speaker)
        role(label, "h2")
        top.addWidget(label)

        share = QLabel(f"{self.voice.total_speech_s:.0f} seconds, "
                       f"{self.voice.share:.0%} of the recording")
        role(share, "muted")
        top.addWidget(share)
        top.addStretch(1)

        self.play = QPushButton("Play")
        self.play.clicked.connect(self._play)
        top.addWidget(self.play)
        layout.addLayout(top)

        if self.voice.is_slight:
            warning = QLabel(
                "This voice barely speaks. It is often the transcriber inventing a "
                "person, so it may belong to somebody already named above."
            )
            warning.setWordWrap(True)
            role(warning, "muted")
            layout.addWidget(warning)

        for line in self.voice.lines[:3]:
            quote = QLabel(f"“{line}”")
            quote.setWordWrap(True)
            role(quote, "faint")
            layout.addWidget(quote)

        row = QHBoxLayout()
        row.addWidget(QLabel("Name"))
        self.name = QLineEdit()
        self.name.setPlaceholderText("Leave blank if you are not sure")
        if suggestion and suggestion.name:
            self.name.setText(suggestion.name)
        row.addWidget(self.name, 1)
        layout.addLayout(row)

        if suggestion and suggestion.evidence:
            why = QLabel(suggestion.evidence)
            why.setWordWrap(True)
            role(why, "faint")
            layout.addWidget(why)

    def _play(self):
        self.play_requested.emit(self.voice.first_heard)

    @property
    def chosen(self) -> str:
        return self.name.text().strip()


class WriteUpView(QWidget):
    """Waiting, naming, and the finished pair of documents."""

    speakers_confirmed = pyqtSignal(dict)
    publish_requested = pyqtSignal()
    new_trip_requested = pyqtSignal()
    retry_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._audio_path = ""
        self._rows = []
        self._player = None
        self._build()

    # --- construction ---------------------------------------------------------

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        self.title = QLabel("Writing this up")
        role(self.title, "h1")
        layout.addWidget(self.title)

        self.panes = QStackedWidget()
        self.panes.addWidget(self._waiting_pane())
        self.panes.addWidget(self._speakers_pane())
        self.panes.addWidget(self._done_pane())
        layout.addWidget(self.panes, 1)

    def _waiting_pane(self) -> QWidget:
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.addStretch(1)

        self.stage = QLabel("")
        role(self.stage, "h2")
        self.stage.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.stage)

        self.bar = QProgressBar()
        # Indeterminate on purpose. These steps take minutes and have no
        # measurable fraction, and a bar that guesses is worse than one that
        # only says "still going".
        self.bar.setRange(0, 0)
        layout.addWidget(self.bar)

        self.detail = QLabel("")
        self.detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail.setWordWrap(True)
        role(self.detail, "muted")
        layout.addWidget(self.detail)

        self.problem = QLabel("")
        self.problem.setWordWrap(True)
        self.problem.setAlignment(Qt.AlignmentFlag.AlignCenter)
        role(self.problem, "danger")
        self.problem.hide()
        layout.addWidget(self.problem)

        self.retry = QPushButton("Try again")
        self.retry.clicked.connect(self.retry_requested)
        self.retry.hide()
        layout.addWidget(self.retry, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addStretch(1)
        return pane

    def _speakers_pane(self) -> QWidget:
        pane = QWidget()
        layout = QVBoxLayout(pane)

        blurb = QLabel(
            "Listen to each voice and give it a name. Suggestions come from the "
            "recording and from people you have named before, and nothing is "
            "applied until you confirm."
        )
        blurb.setWordWrap(True)
        role(blurb, "muted")
        layout.addWidget(blurb)

        self._speaker_area = QScrollArea()
        self._speaker_area.setWidgetResizable(True)
        self._speaker_holder = QWidget()
        self._speaker_layout = QVBoxLayout(self._speaker_holder)
        self._speaker_layout.addStretch(1)
        self._speaker_area.setWidget(self._speaker_holder)
        layout.addWidget(self._speaker_area, 1)

        row = QHBoxLayout()
        self.unnamed = QLabel("")
        role(self.unnamed, "muted")
        row.addWidget(self.unnamed)
        row.addStretch(1)
        self.confirm = QPushButton("Use these names")
        role(self.confirm, "primary")
        self.confirm.clicked.connect(self._confirm)
        row.addWidget(self.confirm)
        layout.addLayout(row)
        return pane

    def _done_pane(self) -> QWidget:
        pane = QWidget()
        layout = QVBoxLayout(pane)

        self.changes = QLabel("")
        self.changes.setWordWrap(True)
        role(self.changes, "muted")
        layout.addWidget(self.changes)

        self.warnings = QLabel("")
        self.warnings.setWordWrap(True)
        role(self.warnings, "danger")
        self.warnings.hide()
        layout.addWidget(self.warnings)

        # The documents themselves, because this is the only chance to read
        # them before they go into a Doc with somebody's name on it, and
        # because a screen that reports "49 corrections applied" and shows
        # none of them is asking to be trusted rather than checked.
        switch = QHBoxLayout()
        self.show_summary = QPushButton("Summary and actions")
        self.show_summary.setCheckable(True)
        self.show_summary.setChecked(True)
        self.show_summary.clicked.connect(lambda: self._show_document("summary"))
        switch.addWidget(self.show_summary)
        self.show_transcript = QPushButton("Transcript")
        self.show_transcript.setCheckable(True)
        self.show_transcript.clicked.connect(lambda: self._show_document("transcript"))
        switch.addWidget(self.show_transcript)
        switch.addStretch(1)
        layout.addLayout(switch)

        self.document = QTextBrowser()
        self.document.setOpenExternalLinks(True)
        layout.addWidget(self.document, 1)

        row = QHBoxLayout()
        row.addStretch(1)
        self.another = QPushButton("Start another trip")
        role(self.another, "quiet")
        self.another.clicked.connect(self.new_trip_requested)
        row.addWidget(self.another)
        self.publish = QPushButton("Copy and open a new Google Doc")
        role(self.publish, "primary")
        self.publish.clicked.connect(self.publish_requested)
        row.addWidget(self.publish)
        layout.addLayout(row)
        return pane

    # --- what the window drives -----------------------------------------------

    def set_audio(self, path: str):
        self._audio_path = path or ""

    def working(self, stage: str, detail: str = ""):
        """Show a long step running, and clear any previous failure."""
        self.title.setText("Writing this up")
        self.stage.setText(stage)
        self.detail.setText(detail)
        self.problem.hide()
        self.retry.hide()
        self.bar.show()
        self.panes.setCurrentIndex(0)

    def note(self, detail: str):
        self.detail.setText(detail)

    def on_failed(self, message: str):
        """A failure with a next move, not a stack trace."""
        self.problem.setText(message)
        self.problem.show()
        self.retry.show()
        self.bar.hide()
        self.stage.setText("The write-up stopped")
        self.panes.setCurrentIndex(0)

    def ask_speakers(self, voices: list, suggestions: list = None):
        by_label = {s.speaker: s for s in (suggestions or [])}
        for row in self._rows:
            row.setParent(None)
        self._rows = []

        for voice in voices:
            row = SpeakerRow(voice, by_label.get(voice.speaker))
            row.play_requested.connect(self._play_from)
            row.name.textChanged.connect(self._count_unnamed)
            self._speaker_layout.insertWidget(self._speaker_layout.count() - 1, row)
            self._rows.append(row)

        self._count_unnamed()
        self.panes.setCurrentIndex(1)

    def show_documents(self, documents, notes=None):
        self._documents = documents
        # The heading has to stop saying "Writing this up" once it has been
        # written. Left as it was, the finished screen reads as though the job
        # is still running and the buttons are premature.
        self.title.setText("Ready to publish")
        self.changes.setText(_summarise(notes or []))
        trouble = []
        if documents.dropped_markers:
            trouble.append(
                "These uncertain terms were not carried into the transcript: "
                + ", ".join(documents.dropped_markers)
            )
        self.warnings.setText("\n".join(trouble))
        self.warnings.setVisible(bool(trouble))
        self._show_document("summary")
        self.panes.setCurrentIndex(2)

    def _show_document(self, which: str):
        documents = getattr(self, "_documents", None)
        if documents is None:
            return
        self.show_summary.setChecked(which == "summary")
        self.show_transcript.setChecked(which == "transcript")
        text = documents.summary if which == "summary" else documents.transcript
        self.document.setMarkdown(text or "")

    # --- internals ------------------------------------------------------------

    def _count_unnamed(self):
        missing = sum(1 for row in self._rows if not row.chosen)
        self.unnamed.setText(
            "" if not missing
            else f"{missing} voice{'s' if missing != 1 else ''} left unnamed, which is fine "
                 f"if you are not sure."
        )

    def _confirm(self):
        self.speakers_confirmed.emit(
            {row.voice.speaker: row.chosen for row in self._rows if row.chosen}
        )

    def _play_from(self, seconds: float):
        if not self._audio_path or not os.path.exists(self._audio_path):
            return
        if self._player is None:
            self._player = QMediaPlayer(self)
            self._output = QAudioOutput(self)
            self._player.setAudioOutput(self._output)
            self._player.setSource(QUrl.fromLocalFile(self._audio_path))
        self._player.setPosition(int(seconds * 1000))
        self._player.play()


def _summarise(notes: list) -> str:
    """The change log, counted rather than listed in full.

    A person about to publish wants to know the shape of what was changed. The
    full list runs to fifty lines on a real trip, which nobody reads, so the
    counts go on screen and the detail goes in the file beside the documents.
    """
    if not notes:
        return "No corrections were needed."
    corrected = sum(1 for n in notes if n.startswith("Corrected"))
    flagged = sum(1 for n in notes if n.startswith("Flagged"))
    held = sum(1 for n in notes if n.startswith("Held back"))
    parts = [f"{corrected} correction{'s' if corrected != 1 else ''} applied"]
    if flagged:
        parts.append(f"{flagged} marked as uncertain for you to check")
    if held:
        parts.append(f"{held} held back because they would have removed words")
    return ", ".join(parts) + "."
