"""
The five green ticks.

Shown whenever something is not working, not once at first run. A token
expiring, a laptop being replaced and an admin revoking access all look the
same from here, and they all get the same list.

Every check makes a real call, several over the network, so they run on a
worker thread and rows update as each one lands. A list that sits blank for
ten seconds and then fills in at once looks broken while it is working.
"""

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from src.podcastnotes.checks_account import ACCOUNT_CHECKS
from src.podcastnotes.checks_local import LOCAL_CHECKS
from src.podcastnotes.readiness import Readiness, Result, State

# Local checks first: they are fast, they need no accounts, and getting three
# ticks immediately tells somebody the app itself is fine and the rest is
# plumbing.
ALL_CHECKS = LOCAL_CHECKS + ACCOUNT_CHECKS

MARKS = {
    State.OK: "✓",
    State.FAILED: "✗",
    State.BLOCKED: "·",
    State.CHECKING: "…",
    State.UNKNOWN: "·",
}

COLOURS = {
    State.OK: "#1a7f37",
    State.FAILED: "#b3261e",
    State.BLOCKED: "#888888",
    State.CHECKING: "#666666",
    State.UNKNOWN: "#888888",
}


class ChecksWorker(QThread):
    """Runs the checks off the GUI thread, reporting each as it lands."""

    started_one = pyqtSignal(str)
    finished_one = pyqtSignal(str, object)
    all_done = pyqtSignal(dict)

    def __init__(self, readiness: Readiness, only=None, known=None, parent=None):
        super().__init__(parent)
        self._readiness = readiness
        self._only = only
        self._known = known or {}

    def run(self):
        results = self._readiness.run(
            only=self._only,
            known=self._known,
            on_start=self.started_one.emit,
            on_result=lambda key, result: self.finished_one.emit(key, result),
        )
        self.all_done.emit(results)


class SignInWorker(QThread):
    """The Google loopback flow, which blocks until the browser comes back."""

    done = pyqtSignal(bool, str)

    def run(self):
        from src.podcastnotes import auth

        try:
            auth.run_sign_in()
            self.done.emit(True, "")
        except Exception as e:
            self.done.emit(False, str(e))


class CheckRow(QFrame):
    """One thing that has to work, and what to do when it does not."""

    fix_requested = pyqtSignal(str)

    def __init__(self, check, parent=None):
        super().__init__(parent)
        self.check = check
        self._build()
        self.show_result(Result(State.UNKNOWN, detail="Not checked yet"))

    def _build(self):
        self.setFrameShape(QFrame.Shape.NoFrame)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(12)

        self.mark = QLabel()
        self.mark.setFixedWidth(20)
        self.mark.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.mark)

        text = QVBoxLayout()
        text.setSpacing(2)
        self.title = QLabel(f"<b>{self.check.title}</b>")
        text.addWidget(self.title)
        self.purpose = QLabel(self.check.purpose)
        self.purpose.setStyleSheet("color: #666;")
        text.addWidget(self.purpose)
        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        text.addWidget(self.detail)
        layout.addLayout(text, 1)

        buttons = QVBoxLayout()
        buttons.setSpacing(4)
        self.fix_button = QPushButton("Fix")
        self.fix_button.clicked.connect(lambda: self.fix_requested.emit(self.check.key))
        buttons.addWidget(self.fix_button)
        self.link_button = QPushButton("Open page")
        self.link_button.clicked.connect(self._open_link)
        buttons.addWidget(self.link_button)
        buttons.addStretch()
        layout.addLayout(buttons)

        self._url = ""

    def _open_link(self):
        if self._url:
            QDesktopServices.openUrl(QUrl(self._url))

    def show_checking(self):
        self.show_result(Result(State.CHECKING, detail="Checking..."))

    def show_result(self, result: Result):
        self.result = result
        self.mark.setText(
            f"<span style='color:{COLOURS[result.state]};font-size:16px'>"
            f"{MARKS[result.state]}</span>"
        )
        parts = [result.detail]
        if result.remedy and not result.ok:
            parts.append(f"<i>{result.remedy}</i>")
        self.detail.setText("<br>".join(p for p in parts if p))
        self.detail.setStyleSheet(
            "color: #b3261e;" if result.state is State.FAILED else "color: #444;"
        )

        self._url = result.url
        self.link_button.setVisible(bool(result.url))
        # Nothing to click on a check that is only waiting, or one nobody can
        # act on. A button that does nothing is worse than no button.
        self.fix_button.setVisible(
            result.state is State.FAILED and self.check.fixable
        )


class SetupView(QWidget):
    """The whole list, and the way through to the rest of the app."""

    ready = pyqtSignal()
    settings_requested = pyqtSignal()
    all_done = pyqtSignal(bool)

    def __init__(self, checks=None, parent=None):
        super().__init__(parent)
        self.readiness = Readiness(list(checks if checks is not None else ALL_CHECKS))
        self.results: dict[str, Result] = {}
        self.worker = None
        self.sign_in_worker = None
        self._build()

    # --- construction ---------------------------------------------------------

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        heading = QLabel("<h2>Setup</h2>")
        layout.addWidget(heading)
        self.summary = QLabel("Checking what is working...")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        holder = QWidget()
        rows = QVBoxLayout(holder)
        rows.setContentsMargins(0, 0, 0, 0)

        self.rows: dict[str, CheckRow] = {}
        for check in self.readiness.checks:
            row = CheckRow(check)
            row.fix_requested.connect(self._fix)
            self.rows[check.key] = row
            rows.addWidget(row)
        rows.addStretch()
        scroll.setWidget(holder)
        layout.addWidget(scroll, 1)

        controls = QHBoxLayout()
        self.recheck_button = QPushButton("Check again")
        self.recheck_button.clicked.connect(lambda: self.start_checks())
        controls.addWidget(self.recheck_button)
        controls.addStretch()
        self.continue_button = QPushButton("Continue")
        self.continue_button.setDefault(True)
        self.continue_button.setEnabled(False)
        self.continue_button.clicked.connect(self.ready.emit)
        controls.addWidget(self.continue_button)
        layout.addLayout(controls)

    # --- running the checks ----------------------------------------------------

    def start_checks(self, only=None):
        if self.worker is not None and self.worker.isRunning():
            return

        self.recheck_button.setEnabled(False)
        self.continue_button.setEnabled(False)
        self.summary.setText("Checking...")

        self.worker = ChecksWorker(
            self.readiness, only=only, known=dict(self.results), parent=self
        )
        self.worker.started_one.connect(self._on_check_started)
        self.worker.finished_one.connect(self._on_check_finished)
        self.worker.all_done.connect(self._on_all_done)
        self.worker.start()

    def _on_check_started(self, key: str):
        if key in self.rows:
            self.rows[key].show_checking()

    def _on_check_finished(self, key: str, result: Result):
        self.results[key] = result
        if key in self.rows:
            self.rows[key].show_result(result)

    def _on_all_done(self, results: dict):
        self.results = results
        for key, result in results.items():
            if key in self.rows:
                self.rows[key].show_result(result)

        self.recheck_button.setEnabled(True)
        everything_works = Readiness.all_ok(results)
        self.continue_button.setEnabled(everything_works)
        self.summary.setText(self._summary_text(results))
        self.all_done.emit(everything_works)

    def _summary_text(self, results: dict) -> str:
        if Readiness.all_ok(results):
            return "Everything is working."

        first = Readiness.first_problem(results)
        broken = sum(1 for r in results.values() if r.state is State.FAILED)
        if first is None:
            return "Still working out what is wrong."

        title = self.rows[first].check.title
        if broken == 1:
            return f"One thing needs sorting out: <b>{title}</b>."
        return (
            f"{broken} things need sorting out. Start with <b>{title}</b>; "
            f"the others may clear up once it works."
        )

    # --- fixing ----------------------------------------------------------------

    def _fix(self, key: str):
        if key == "google":
            self._sign_in_to_google()
        else:
            # Everything else is a value somebody pastes, and those all live in
            # one place rather than in seven bespoke dialogs.
            self.settings_requested.emit()

    def _sign_in_to_google(self):
        if self.sign_in_worker is not None and self.sign_in_worker.isRunning():
            return
        self.summary.setText("Waiting for you to sign in, in your browser...")
        self.sign_in_worker = SignInWorker(parent=self)
        self.sign_in_worker.done.connect(self._on_sign_in_done)
        self.sign_in_worker.start()

    def _on_sign_in_done(self, worked: bool, message: str):
        if worked:
            # Claude and Drive both hang off this, so re-check the lot.
            self.start_checks()
        else:
            self.summary.setText(f"Sign-in did not finish: {message}")
