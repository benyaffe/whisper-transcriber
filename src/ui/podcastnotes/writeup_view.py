"""
The screen the write-up happens on, and the one place it asks anything.

Four panes behind one widget: waiting, naming the speakers, answering the few
questions nothing could resolve, and the finished documents. They are panes
rather than separate screens because the write-up is one continuous thing from
the user's point of view, and bouncing between top-level screens for a job that
takes a few minutes reads as the app losing its place.

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
    QCheckBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QProgressBar, QPushButton,
    QScrollArea, QStackedWidget, QTextBrowser, QVBoxLayout, QWidget,
)

from src.ui.theme import role

# Long enough to recognise a colleague, short enough that reviewing four of
# them is not a chore. Matches what snippet.py already cuts.
CLIP_SECONDS = 6.0

# Named, because they are positions in a stack and inserting a pane shifts
# every one after it. Adding the questions pane silently moved the finished
# documents from 2 to 3, which a bare number gives no way to notice.
PANE_WAITING = 0
PANE_CONTEXT = 1
PANE_SPEAKERS = 2
PANE_QUESTIONS = 3
PANE_DONE = 4


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

        self.asking = suggestion is not None and not suggestion.worth_filling_in

        if self.asking:
            # Said plainly, because this row is the one that needs a person and
            # the three above it do not. A guess left in the box here reads
            # exactly like the confident ones and gets accepted with them.
            prompt = QLabel("Who is this? The recording does not make it clear.")
            prompt.setWordWrap(True)
            role(prompt, "danger")
            layout.addWidget(prompt)
        elif self.voice.is_slight:
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
        self.name.setPlaceholderText(
            "Type a name, or leave blank" if self.asking
            else "Leave blank if you are not sure"
        )
        if suggestion and suggestion.worth_filling_in:
            self.name.setText(suggestion.name)
        row.addWidget(self.name, 1)
        layout.addLayout(row)

        if suggestion and suggestion.evidence:
            why = QLabel(
                f"Best guess, not confident enough to fill in: {suggestion.evidence}"
                if self.asking else suggestion.evidence
            )
            why.setWordWrap(True)
            role(why, "faint")
            layout.addWidget(why)

    def _play(self):
        self.play_requested.emit(self.voice.first_heard)

    @property
    def chosen(self) -> str:
        return self.name.text().strip()


class WriteUpView(QWidget):
    """Waiting, naming, answering, and the finished pair of documents."""

    context_approved = pyqtSignal(list)   # the corrections to drop
    speakers_confirmed = pyqtSignal(dict)
    answers_given = pyqtSignal(dict)
    publish_requested = pyqtSignal()
    new_trip_requested = pyqtSignal()
    retry_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._audio_path = ""
        self._rows = []
        self._questions = []
        self._corrections = []
        self._player = None
        # Where the next seek should land. Held rather than applied, because
        # the media may not be loaded yet.
        self._wanted_at = None
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
        self.panes.addWidget(self._context_pane())
        self.panes.addWidget(self._speakers_pane())
        self.panes.addWidget(self._questions_pane())
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

    def _context_pane(self) -> QWidget:
        """What was found, and the chance to reject a correction before it lands.

        The corrections are the editable part because they are the part that
        changes the words. Everything else here is background that only
        influences how Claude reads the recording, whereas an entry in this
        list is applied literally and everywhere, so one wrong row becomes one
        wrong word throughout a published document.
        """
        pane = QWidget()
        layout = QVBoxLayout(pane)

        self.found = QLabel("")
        self.found.setWordWrap(True)
        role(self.found, "muted")
        layout.addWidget(self.found)

        area = QScrollArea()
        area.setWidgetResizable(True)
        self._correction_holder = QWidget()
        self._correction_layout = QVBoxLayout(self._correction_holder)
        self._correction_layout.addStretch(1)
        area.setWidget(self._correction_holder)
        layout.addWidget(area, 1)

        row = QHBoxLayout()
        self.rejected_count = QLabel("")
        role(self.rejected_count, "muted")
        row.addWidget(self.rejected_count)
        row.addStretch(1)
        self.approve = QPushButton("Use these corrections")
        role(self.approve, "primary")
        self.approve.clicked.connect(self._approve_context)
        row.addWidget(self.approve)
        layout.addLayout(row)
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

    def _questions_pane(self) -> QWidget:
        pane = QWidget()
        layout = QVBoxLayout(pane)

        self.round_blurb = QLabel("")
        self.round_blurb.setWordWrap(True)
        role(self.round_blurb, "muted")
        layout.addWidget(self.round_blurb)

        area = QScrollArea()
        area.setWidgetResizable(True)
        self._question_holder = QWidget()
        self._question_layout = QVBoxLayout(self._question_holder)
        self._question_layout.addStretch(1)
        area.setWidget(self._question_holder)
        layout.addWidget(area, 1)

        row = QHBoxLayout()
        row.addStretch(1)
        # One button. Skipping is leaving every box empty and pressing this,
        # which is what it already did: a separate "Skip these" implied the two
        # were different and invited somebody to hunt for the difference.
        self.send_answers = QPushButton("Use these answers")
        role(self.send_answers, "primary")
        self.send_answers.clicked.connect(self._send_answers)
        row.addWidget(self.send_answers)
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

        self.showing = QLabel("")
        role(self.showing, "muted")
        layout.addWidget(self.showing)

        self.document = QTextBrowser()
        self.document.setOpenExternalLinks(True)
        layout.addWidget(self.document, 1)

        self.published = QLabel("")
        self.published.setWordWrap(True)
        self.published.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        role(self.published, "success")
        self.published.hide()
        layout.addWidget(self.published)

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

    def working(self, stage: str, detail: str = "", phase: str = ""):
        """Show a long step running, and clear any previous failure.

        `phase` is the heading. It used to read "Writing this up" for every
        stage from the first Glean search to the finished pair, which is a
        quarter of an hour of a screen that never changes and a "this" nobody
        can point at.
        """
        self.title.setText(phase or "Writing this up")
        self.stage.setText(stage)
        self.detail.setText(detail)
        self.problem.hide()
        self.retry.hide()
        self.bar.show()
        self.panes.setCurrentIndex(PANE_WAITING)

    def note(self, detail: str):
        self.detail.setText(detail)

    def on_failed(self, message: str):
        """A failure with a next move, not a stack trace."""
        self.problem.setText(message)
        self.problem.show()
        self.retry.show()
        self.bar.hide()
        self.title.setText("The write-up stopped")
        self.stage.setText("")
        # The last thing it was doing has to go with it. Left in place, "Looking
        # up: Miles Nadeau" sits under a failure message and reads as though
        # that lookup were still running.
        self.detail.setText("")
        self.panes.setCurrentIndex(PANE_WAITING)

    def review_context(self, context_map):
        """Show what the background search found, before any of it is applied."""
        for widget in self._corrections:
            widget.setParent(None)
        self._corrections = []

        self.found.setText(
            f"Found {len(context_map.people)} people, "
            f"{len(context_map.terms)} terms and "
            f"{len(context_map.projects)} projects, over "
            f"{len(context_map.searches)} searches. "
            f"These are the corrections it wants to make to the transcript. "
            f"Untick anything that looks wrong."
        )
        for row in context_map.likely_errors:
            box = _CorrectionRow(row, self._correction_holder)
            box.tick.toggled.connect(self._count_rejected)
            self._correction_layout.insertWidget(
                self._correction_layout.count() - 1, box
            )
            self._corrections.append(box)

        self._count_rejected()
        self.panes.setCurrentIndex(PANE_CONTEXT)

    def _count_rejected(self):
        dropped = [b for b in self._corrections if not b.tick.isChecked()]
        self.rejected_count.setText(
            "" if not dropped
            else f"{len(dropped)} correction{'s' if len(dropped) != 1 else ''} will not be applied."
        )

    def _approve_context(self):
        self.context_approved.emit(
            [b.heard for b in self._corrections if not b.tick.isChecked()]
        )

    def ask_speakers(self, voices: list, suggestions: list = None):
        by_label = {s.speaker: s for s in (suggestions or [])}
        for row in self._rows:
            row.setParent(None)
        self._rows = []

        for voice in voices:
            row = SpeakerRow(voice, by_label.get(voice.speaker))
            row.play_requested.connect(self._play_one)
            row.name.textChanged.connect(self._count_unnamed)
            self._speaker_layout.insertWidget(self._speaker_layout.count() - 1, row)
            self._rows.append(row)

        self._count_unnamed()
        self.panes.setCurrentIndex(PANE_SPEAKERS)

    def ask_questions(self, round_):
        """One round of questions, each with a box and each skippable."""
        for widget in self._questions:
            widget.setParent(None)
        self._questions = []

        left = (f" There are {round_.remaining} more that could be asked after this."
                if round_.remaining else "")
        self.round_blurb.setText(
            f"A few things I could not work out from the recording. Answer what you "
            f"can and leave the rest blank.{left}"
        )

        for question in round_.questions:
            box = _QuestionBox(question, self._question_holder)
            box.box.textChanged.connect(self._count_answers)
            self._questions.append(box)
            self._question_layout.insertWidget(
                self._question_layout.count() - 1, box
            )
        self._count_answers()
        self.panes.setCurrentIndex(PANE_QUESTIONS)

    def _count_answers(self):
        """Say what the button will do, since leaving them blank is allowed."""
        answered = sum(1 for box in self._questions if box.answer)
        self.send_answers.setText(
            "Use these answers" if answered else "Skip these, none of them are mine"
        )

    def _send_answers(self):
        self.answers_given.emit({
            box.question.marker: box.answer
            for box in self._questions if box.answer
        })

    def show_documents(self, documents, notes=None):
        self._documents = documents
        # The heading has to stop saying "Writing this up" once it has been
        # written. Left as it was, the finished screen reads as though the job
        # is still running and the buttons are premature.
        self.title.setText("Ready to publish")
        # No correction counts. By the time somebody is reading the finished
        # pair, how many substitutions were made three steps ago is not news,
        # and it pushed the thing they came for further down the screen. What
        # is still worth interrupting for is a marker that went missing, which
        # is below.
        self.changes.setText("")
        self.changes.hide()
        trouble = []
        if documents.dropped_markers:
            trouble.append(
                "These uncertain terms were not carried into the transcript: "
                + ", ".join(documents.dropped_markers)
            )
        self.warnings.setText("\n".join(trouble))
        self.warnings.setVisible(bool(trouble))
        self.published.hide()
        self._show_document("summary")
        self.panes.setCurrentIndex(PANE_DONE)

    def show_published(self, result):
        """Say what publishing actually did, and what to do next.

        It used to do three things and mention none of them. The browser tab
        opening was the only sign anything had happened, which is
        indistinguishable from the app having done nothing but open a tab, and
        a plain paste into Google Docs puts the raw Markdown source on the page
        as literal text.

        The instruction to right-click and choose Paste from Markdown has
        existed in publish.py since it was written and was never once shown.
        """
        from src.podcastnotes.publish import PASTE_INSTRUCTIONS

        lines = [result.summary()]
        if result.copied:
            lines.append(PASTE_INSTRUCTIONS)
        for problem in result.problems:
            lines.append(problem)

        self.published.setText("\n\n".join(line.strip() for line in lines if line))
        role(self.published, "success" if result.copied else "danger")
        from src.ui.theme import restyle

        restyle(self.published)
        self.published.show()

    def _show_document(self, which: str):
        documents = getattr(self, "_documents", None)
        if documents is None:
            return
        self.show_summary.setChecked(which == "summary")
        self.show_transcript.setChecked(which == "transcript")
        text = documents.summary if which == "summary" else documents.transcript
        self.showing.setText(
            "The summary and action items, for somebody who was not there."
            if which == "summary"
            else "The full transcript, reflowed and with the speakers named."
        )
        self.document.setMarkdown(text or "")

    # --- internals ------------------------------------------------------------

    def _count_unnamed(self):
        """Keep the button honest about what pressing it will do.

        Leaving a voice unnamed is allowed and sometimes right, but it should
        be a thing somebody chose rather than a thing they did not notice. The
        button says which one it is.
        """
        missing = sum(1 for row in self._rows if not row.chosen)
        self.unnamed.setText(
            "" if not missing
            else f"{missing} voice{'s' if missing != 1 else ''} will stay unnamed in the "
                 f"documents."
        )
        self.confirm.setText(
            "Use these names" if not missing
            else f"Continue with {missing} unnamed"
        )

    def _confirm(self):
        self.speakers_confirmed.emit(
            {row.voice.speaker: row.chosen for row in self._rows if row.chosen}
        )

    def _play_one(self, seconds: float):
        """Only one voice at a time; a second Play replaces the first."""
        self.stop_playing()
        self._play_from(seconds)

    def _play_from(self, seconds: float):
        """Play a voice from its own moment, not from the top of the recording.

        The seek has to wait for the media. `setSource` loads asynchronously,
        and a `setPosition` issued before it finishes is discarded in silence:
        measured, the position stays at 0 with status LoadingMedia, and
        playback then starts at 0:00 of a forty-two minute file. Clicking Play
        on the third speaker and hearing the first one is why this was reported
        as the buttons doing nothing. The second click always worked, because
        by then the media was loaded.
        """
        if not self._audio_path or not os.path.exists(self._audio_path):
            return

        self._wanted_at = int(seconds * 1000)
        if self._player is None:
            self._player = QMediaPlayer(self)
            self._output = QAudioOutput(self)
            self._player.setAudioOutput(self._output)
            self._player.mediaStatusChanged.connect(self._media_status_changed)
            self._player.setSource(QUrl.fromLocalFile(self._audio_path))

        if self._player.isSeekable():
            self._seek_and_play()

    def _media_status_changed(self, status):
        """Apply a seek that arrived before the file was ready."""
        ready = (
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
        )
        if status in ready and self._wanted_at is not None:
            self._seek_and_play()

    def _seek_and_play(self):
        self._player.setPosition(self._wanted_at)
        self._wanted_at = None
        self._player.play()

    def stop_playing(self):
        """Silence, for leaving the screen or picking a different voice."""
        self._wanted_at = None
        if self._player is not None:
            self._player.stop()


class _CorrectionRow(QFrame):
    """One proposed substitution, with a tick that decides whether it happens."""

    def __init__(self, row, parent=None):
        super().__init__(parent)
        self.heard = (row.get("heard") or "").strip()
        role(self, "card")

        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        confidence = (row.get("confidence") or "").strip().lower()
        marker = "" if confidence == "high" else "  (will be marked uncertain)"
        self.label = (
            f"{self.heard}  →  {(row.get('probably') or '').strip()}{marker}"
        )

        # The text is a wrapping label beside the box rather than the box's own
        # caption. A QCheckBox caption does not wrap, so a correction covering
        # four heard variants ran off the edge and put a horizontal scrollbar
        # under a vertical list, leaving the change unreadable.
        top = QHBoxLayout()
        self.tick = QCheckBox()
        self.tick.setChecked(True)
        top.addWidget(self.tick, alignment=Qt.AlignmentFlag.AlignTop)
        text = QLabel(self.label)
        text.setWordWrap(True)
        text.setBuddy(self.tick)
        top.addWidget(text, 1)
        layout.addLayout(top)

        evidence = (row.get("evidence") or "").strip()
        if evidence:
            why = QLabel(evidence)
            why.setWordWrap(True)
            role(why, "faint")
            layout.addWidget(why)


class _QuestionBox(QFrame):
    """One question, its context, and somewhere to answer it."""

    def __init__(self, question, parent=None):
        super().__init__(parent)
        self.question = question
        role(self, "card")

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        ask = QLabel(question.ask)
        ask.setWordWrap(True)
        layout.addWidget(ask)

        if question.why_it_matters:
            why = QLabel(question.why_it_matters)
            why.setWordWrap(True)
            role(why, "faint")
            layout.addWidget(why)

        self.box = QLineEdit()
        self.box.setPlaceholderText("Leave blank if you do not know")
        layout.addWidget(self.box)

    @property
    def answer(self) -> str:
        return self.box.text().strip()


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
