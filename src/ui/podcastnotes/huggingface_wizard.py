"""
Walking somebody through HuggingFace, or letting them decide not to.

Naming who said what is most of the value in a trip write-up, so the setup is
offered rather than buried. It is also the fiddliest thing in the whole app:
an account, a token, and three separate licence pages. Somebody who does not
know what HuggingFace is will stop dead at "paste a token" and never come back.

So this walks through it a step at a time, checks after each one, and says
plainly what is left. Skipping is a first-class button rather than a cancel,
because going without speaker names is a reasonable choice and the transcript
is still written either way.
"""

from PyQt6.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QWidget,
)

from src.core.config import get_hf_token, save_hf_token, set_speaker_id_enabled
from src.core.diarization import GATED_MODELS
from src.ui.theme import role

SIGNUP_URL = "https://huggingface.co/join"

# The name is prefilled as well as the type. HuggingFace requires a name and
# the wizard never mentioned it, so people reached a form with a required
# field the instructions had not prepared them for.
TOKEN_NAME = "PodcastNotesWT"
TOKEN_URL = (
    "https://huggingface.co/settings/tokens/new"
    f"?tokenType=read&tokenName={TOKEN_NAME}"
)

WHAT_IT_BUYS = (
    "With this, the transcript says who spoke. Without it, every line reads "
    "\"Speaker 1\" and you match them up yourself."
)


class _CheckWorker(QThread):
    """Validating hits the network three times, so it does not go on the GUI."""

    done = pyqtSignal(object)

    def __init__(self, token, parent=None):
        super().__init__(parent)
        self._token = token

    def run(self):
        from src.core.diarization import token_status

        try:
            self.done.emit(token_status(self._token))
        except Exception as e:
            from src.core.diarization import TokenStatus

            self.done.emit(TokenStatus(problem="error", detail=str(e)[:120]))


class _Step(QFrame):
    """One numbered instruction with a button beside it."""

    def __init__(self, number: int, text: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)

        self.marker = QLabel(str(number))
        self.marker.setFixedWidth(22)
        self.marker.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.marker)

        self.label = QLabel(text)
        self.label.setWordWrap(True)
        layout.addWidget(self.label, 1)

        self.button = QPushButton()
        self.button.setVisible(False)
        layout.addWidget(self.button)

    def set_done(self, done: bool):
        self.marker.setText(
            "<span style='color:#1a7f37'>&#10003;</span>" if done else str(self._n())
        )

    def _n(self):
        return getattr(self, "_number", "")

    def set_number(self, number: int):
        self._number = number
        self.marker.setText(str(number))

    def link(self, label: str, url: str):
        self.button.setText(label)
        self.button.setVisible(True)
        self.button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(url)))


class HuggingFaceWizard(QDialog):
    """Three steps, checked after each, with skipping always available."""

    skipped = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set up speaker names")
        self.setMinimumWidth(560)
        self.worker = None
        self.status = None
        self._build()
        self._load()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        layout.addWidget(role(QLabel("Speaker names"), "h1"))

        why = QLabel(WHAT_IT_BUYS)
        why.setWordWrap(True)
        role(why, "muted")
        layout.addWidget(why)

        cost = QLabel(
            "It is free and takes about two minutes. The models are run on "
            "your own machine; nothing is uploaded."
        )
        cost.setWordWrap(True)
        role(cost, "faint")
        layout.addWidget(cost)

        layout.addSpacing(6)

        self.step_account = _Step(1, "Create a free HuggingFace account, if "
                                    "you do not have one.")
        self.step_account.set_number(1)
        self.step_account.link("Open", SIGNUP_URL)
        layout.addWidget(self.step_account)

        self.step_token = _Step(2, f"Make a token with Read access. The page asks "
                                   f"for a name; anything does, and the link below "
                                   f"fills in \u201c{TOKEN_NAME}\u201d. Paste it here.")
        self.step_token.set_number(2)
        self.step_token.link("Open", TOKEN_URL)
        layout.addWidget(self.step_token)

        # Unmasked, unlike the same field in Accounts, which has a Show button.
        # This one has nothing to hide: it is being pasted in for the first
        # time, off a page the person still has open, and a paste that came
        # through truncated is unverifiable behind dots.
        token_row = QHBoxLayout()
        self.token_input = QLineEdit()
        self.token_input.setPlaceholderText("hf_...")
        self.token_input.returnPressed.connect(self._check)
        token_row.addWidget(self.token_input, 1)

        # Beside the field it validates. It used to be in the bottom button
        # bar, below step 3, so the only way forward from step 2 was a control
        # five widgets further down that looked like it belonged to the dialog
        # rather than to the token.
        self.check_button = QPushButton("Check")
        self.check_button.clicked.connect(self._check)
        token_row.addWidget(self.check_button)
        layout.addLayout(token_row)

        self.step_licences = _Step(3, "Accept three model licences. Each is "
                                      "one button on its page.")
        self.step_licences.set_number(3)
        layout.addWidget(self.step_licences)

        self.licence_box = QWidget()
        self.licence_layout = QVBoxLayout(self.licence_box)
        self.licence_layout.setContentsMargins(22, 0, 0, 0)
        self.licence_layout.setSpacing(3)
        layout.addWidget(self.licence_box)

        # Shown from the start, not only after a token has checked out. Step 3
        # had no button of its own and the three links appeared only once a
        # valid token produced a "licences" problem, so before then the step
        # named three pages with no way to reach any of them, and the licences
        # looked like something the wizard would do for you.
        self._show_licences(GATED_MODELS)

        self.message = QLabel("")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)

        layout.addStretch()

        buttons = QHBoxLayout()
        self.skip_button = QPushButton("Skip, no speaker names")
        self.skip_button.clicked.connect(self._skip)
        buttons.addWidget(self.skip_button)
        buttons.addStretch()

        self.done_button = QPushButton("Done")
        role(self.done_button, "primary")
        self.done_button.setDefault(True)
        self.done_button.setEnabled(False)
        self.done_button.clicked.connect(self._finish)
        buttons.addWidget(self.done_button)
        layout.addLayout(buttons)

    def _load(self):
        self.token_input.setText(get_hf_token())
        if self.token_input.text():
            self._check()

    # --- checking -------------------------------------------------------------

    def _check(self):
        if self.worker is not None and self.worker.isRunning():
            return
        token = self.token_input.text().strip()
        if not token:
            self.message.setText("Paste the token first.")
            return

        self.check_button.setEnabled(False)
        self.message.setText("Checking with HuggingFace...")
        self.worker = _CheckWorker(token, parent=self)
        self.worker.done.connect(self._on_checked)
        self.worker.start()

    def _on_checked(self, status):
        self.status = status
        self.check_button.setEnabled(True)
        self._clear_licences()

        # Separately, because they are separate things. Both used to tick from
        # `signed_in`, so step 1 could never complete on its own and somebody
        # who already had an account saw it stay unticked until they had also
        # pasted a working token.
        self.step_account.set_done(status.signed_in)
        self.step_token.set_done(bool(self.token_input.text().strip()) and status.signed_in)

        if status.valid:
            self.step_licences.set_done(True)
            self.message.setText(
                f"<span style='color:#1a7f37'>Ready. Signed in as "
                f"'{status.username}'.</span>"
            )
            self.done_button.setEnabled(True)
            return

        self.done_button.setEnabled(False)
        self.step_licences.set_done(False)

        if status.problem == "licences":
            self._show_licences(status.missing_licences)
            self.message.setText(
                f"Signed in as '{status.username}'. {len(status.missing_licences)} "
                f"licence(s) to accept, then press Check again."
            )
        else:
            self.message.setText(
                f"<span style='color:#b3261e'>{status.detail}</span>"
            )

    def _show_licences(self, models):
        for model in models:
            row = QHBoxLayout()
            label = QLabel(f"\u2022  {model.name}")
            row.addWidget(label, 1)
            button = QPushButton("Accept")
            button.clicked.connect(
                lambda _=False, url=model.url: QDesktopServices.openUrl(QUrl(url))
            )
            row.addWidget(button)
            holder = QWidget()
            holder.setLayout(row)
            self.licence_layout.addWidget(holder)

    def _clear_licences(self):
        while self.licence_layout.count():
            item = self.licence_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    # --- finishing ------------------------------------------------------------

    def _finish(self):
        self._settle_worker()
        save_hf_token(self.token_input.text().strip())
        set_speaker_id_enabled(True)
        self.accept()

    def _skip(self):
        """Record the choice rather than just closing.

        Closing would leave the row red and the person would be asked again
        every launch, which is how a genuine choice turns into nagging.
        """
        self.skipped.emit()
        self.reject()   # which is what records the choice

    def _settle_worker(self):
        """Let a check in flight finish before the dialog goes away.

        Closing on top of a running QThread produces "QThread: Destroyed while
        thread is still running" and can take the process with it. The check is
        three network calls, so a short wait is the whole cost.
        """
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(3000)

    def reject(self):
        """Escape and the close box mean the same thing as Skip.

        They used to go straight to QDialog.reject, bypassing `_skip`, so the
        choice was never written down and the person was asked again on every
        launch. Which is the exact nagging `_skip` exists to prevent.
        """
        self._settle_worker()
        if not self.status or not self.status.valid:
            set_speaker_id_enabled(False)
        super().reject()
