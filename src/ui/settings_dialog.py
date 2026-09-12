"""
Settings dialog for PodcastNotesWT.
Handles speaker identification toggle and HuggingFace token configuration.
"""

from PyQt6.QtWidgets import (
    QComboBox, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGroupBox, QFormLayout, QMessageBox, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# These used to be defined here, which made src/core import src/ui.
# src/core/config.py is now their only home.
from src.ui.theme import role
from src.core.config import (
    get_hf_token,
    is_speaker_id_enabled,
    save_hf_token,
    set_speaker_id_enabled,
)


class _ProjectsWorker(QThread):
    """Lists the account's Cloud projects without freezing the dialog."""

    done = pyqtSignal(list, str)

    def __init__(self, credentials, parent=None):
        super().__init__(parent)
        self._credentials = credentials

    def run(self):
        from src.podcastnotes import auth

        try:
            self.done.emit(auth.list_projects(self._credentials), "")
        except auth.ProjectListUnavailable:
            # Common, and not an error worth showing. Google will not list
            # projects unless the Cloud Resource Manager API is switched on,
            # which it usually is not. Fall back to the project gcloud is
            # already pointed at, which is nearly always the right one.
            default = auth.default_project()
            if default:
                self.done.emit([{"id": default, "name": default}], "")
            else:
                self.done.emit([], "LIST_UNAVAILABLE")
        except Exception as e:
            self.done.emit([], str(e))


class SettingsDialog(QDialog):
    """Settings dialog for app configuration."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Accounts")
        self.setMinimumWidth(550)
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Speaker Identification group
        speaker_group = QGroupBox("Speaker Identification")
        speaker_layout = QVBoxLayout(speaker_group)

        # Enable toggle
        self.enable_checkbox = QCheckBox("Enable speaker identification")
        self.enable_checkbox.stateChanged.connect(self._on_toggle_changed)
        speaker_layout.addWidget(self.enable_checkbox)

        # Info label
        self.info_label = QLabel(
            "Identifies different speakers in your audio using pyannote.audio.\n"
            "Requires a free HuggingFace account and access token."
        )
        self.info_label.setWordWrap(True)
        role(self.info_label, "muted")
        speaker_layout.addWidget(self.info_label)

        # Token section (shown when enabled)
        self.token_widget = QGroupBox("HuggingFace Token")
        token_layout = QVBoxLayout(self.token_widget)

        # Requirements. The gated-model list is derived from the same
        # constant the downloader and token validator use, so the links
        # here cannot drift out of sync with what we actually check.
        from src.core.diarization import GATED_MODELS

        model_links = "".join(
            f"&nbsp;&nbsp;&nbsp;- <a href='{m.url}'>{m.repo}</a><br>"
            for m in GATED_MODELS
        )
        req_label = QLabel(
            "<b>Requirements:</b><br>"
            "1. Create a free account at <a href='https://huggingface.co/join'>huggingface.co</a><br>"
            f"2. Accept the license for <b>all {len(GATED_MODELS)}</b> required models:<br>"
            f"{model_links}"
            "3. Create an access token at <a href='https://huggingface.co/settings/tokens'>Settings &gt; Access Tokens</a><br>"
            "<br>"
            "<b>Token permissions:</b> Read access to gated repos (select 'Read' when creating)"
        )
        req_label.setOpenExternalLinks(True)
        req_label.setWordWrap(True)
        token_layout.addWidget(req_label)

        # Token input
        form_layout = QFormLayout()
        token_input_layout = QHBoxLayout()

        self.token_input = QLineEdit()
        self.token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_input.setPlaceholderText("hf_xxxxxxxxxxxxxxxxxxxxxxxx")
        self.token_input.textChanged.connect(self._on_token_changed)
        token_input_layout.addWidget(self.token_input)

        self.show_token_btn = QPushButton("Show")
        self.show_token_btn.setFixedWidth(60)
        self.show_token_btn.clicked.connect(self._toggle_token_visibility)
        token_input_layout.addWidget(self.show_token_btn)

        form_layout.addRow("Token:", token_input_layout)
        token_layout.addLayout(form_layout)

        # Validation status
        self.validation_label = QLabel("")
        self.validation_label.setWordWrap(True)
        token_layout.addWidget(self.validation_label)

        # Validate button
        validate_layout = QHBoxLayout()
        self.validate_btn = QPushButton("Validate Token")
        self.validate_btn.clicked.connect(self._validate_token)
        validate_layout.addWidget(self.validate_btn)
        validate_layout.addStretch()
        token_layout.addLayout(validate_layout)

        speaker_layout.addWidget(self.token_widget)
        layout.addWidget(speaker_group)

        layout.addWidget(self._build_google_group())
        layout.addWidget(self._build_glean_group())

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        self.save_btn = QPushButton("Save")
        role(self.save_btn, "primary")
        self.save_btn.setDefault(True)
        self.save_btn.clicked.connect(self._save_settings)
        button_layout.addWidget(self.save_btn)

        layout.addLayout(button_layout)

    # --- the accounts the AI stages need --------------------------------------

    def _build_google_group(self):
        """Which Google Cloud project to bill Claude to.

        Everyone is on a different one, because everyone is on a different
        cost center, so this cannot be a constant. It is a dropdown rather
        than a text field because a project id is not something most people
        can produce from memory, and typing one wrong produces an error that
        looks like a permissions problem.
        """
        group = QGroupBox("Google Cloud project")
        box = QVBoxLayout(group)

        hint = QLabel(
            "Claude runs on your team's own Google Cloud project, so the cost "
            "lands on your cost center. Sign in first, then find your projects."
        )
        hint.setWordWrap(True)
        role(hint, "muted")
        box.addWidget(hint)

        row = QHBoxLayout()
        self.project_combo = QComboBox()
        self.project_combo.setEditable(True)
        self.project_combo.setPlaceholderText("your-team-project-id")
        row.addWidget(self.project_combo, 1)
        self.find_projects_btn = QPushButton("Find my projects")
        self.find_projects_btn.clicked.connect(self._find_projects)
        row.addWidget(self.find_projects_btn)
        box.addLayout(row)

        self.project_status = QLabel("")
        self.project_status.setWordWrap(True)
        role(self.project_status, "muted")
        box.addWidget(self.project_status)
        return group

    def _build_glean_group(self):
        group = QGroupBox("Glean")
        box = QVBoxLayout(group)

        hint = QLabel(
            "Used to look up the people, organisations and acronyms a trip "
            "involves. Your own credential, so a trip only ever sees what you "
            "can see."
        )
        hint.setWordWrap(True)
        role(hint, "muted")
        box.addWidget(hint)

        form = QFormLayout()
        self.glean_instance_input = QLineEdit()
        self.glean_instance_input.setPlaceholderText(
            "the address you use for Glean, e.g. acme.glean.com"
        )
        form.addRow("Address:", self.glean_instance_input)

        self.glean_token_input = QLineEdit()
        self.glean_token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.glean_token_input.setPlaceholderText("your personal Glean token")
        form.addRow("Token:", self.glean_token_input)
        box.addLayout(form)
        return group

    def _find_projects(self):
        """List the signed-in account's projects, off the GUI thread."""
        from src.podcastnotes import auth

        credentials = auth.stored_credentials() if auth.is_configured() else None
        if credentials is None:
            self.project_status.setText(
                "Sign in with Google first, on the Setup screen."
            )
            return

        self.find_projects_btn.setEnabled(False)
        self.project_status.setText("Looking...")
        self._projects_worker = _ProjectsWorker(credentials, parent=self)
        self._projects_worker.done.connect(self._on_projects_found)
        self._projects_worker.start()

    def _on_projects_found(self, projects: list, error: str):
        self.find_projects_btn.setEnabled(True)
        if error == "LIST_UNAVAILABLE":
            self.project_status.setText(
                "Your account cannot list projects, which is normal. Type the "
                "project id instead; your team will know it."
            )
            return
        if error:
            self.project_status.setText(f"Could not list your projects: {error}")
            return
        if not projects:
            self.project_status.setText(
                "That account has no Google Cloud projects. Ask whoever runs "
                "your team's cloud account which one to use."
            )
            return

        current = self.project_combo.currentText().strip()
        self.project_combo.clear()
        for project in projects:
            label = (
                project["name"]
                if project["name"] == project["id"]
                else f"{project['name']} ({project['id']})"
            )
            self.project_combo.addItem(label, project["id"])
        if current:
            self._select_project(current)
        self.project_status.setText(f"Found {len(projects)}.")

    def _select_project(self, project_id: str):
        for i in range(self.project_combo.count()):
            if self.project_combo.itemData(i) == project_id:
                self.project_combo.setCurrentIndex(i)
                return
        self.project_combo.setCurrentText(project_id)

    def _chosen_project(self) -> str:
        """The id, whether it was picked from the list or typed.

        itemData holds the id and the visible text holds a friendly label, so
        reading currentText on a picked item would save the label as the id.
        """
        index = self.project_combo.currentIndex()
        if index >= 0 and self.project_combo.itemData(index):
            typed = self.project_combo.currentText().strip()
            if typed == self.project_combo.itemText(index):
                return self.project_combo.itemData(index)
        return self.project_combo.currentText().strip()

    def _toggle_token_visibility(self):
        if self.token_input.echoMode() == QLineEdit.EchoMode.Password:
            self.token_input.setEchoMode(QLineEdit.EchoMode.Normal)
            self.show_token_btn.setText("Hide")
        else:
            self.token_input.setEchoMode(QLineEdit.EchoMode.Password)
            self.show_token_btn.setText("Show")

    def _load_settings(self):
        self._load_account_settings()
        self.token_input.setText(get_hf_token())
        enabled = is_speaker_id_enabled()
        self.enable_checkbox.setChecked(enabled)
        self._update_token_visibility()
        self._token_validated = False

        # If enabled and token exists, assume it was previously validated
        if enabled and get_hf_token():
            self._token_validated = True
            self.validation_label.setText(
                "<span style='color: green;'>Token previously validated</span>"
            )

    def _on_toggle_changed(self, state):
        self._update_token_visibility()

        # If turning on without a validated token, prompt user
        if state == Qt.CheckState.Checked.value:
            token = self.token_input.text().strip()
            if not token:
                self.validation_label.setText(
                    "<span style='color: #c90;'>Please enter your HuggingFace token above</span>"
                )
                self._token_validated = False
            elif not self._token_validated:
                self.validation_label.setText(
                    "<span style='color: #c90;'>Please validate your token</span>"
                )

    def _on_token_changed(self, text):
        # Reset validation when token changes
        self._token_validated = False
        if text.strip():
            self.validation_label.setText(
                "<span style='color: #666;'>Click 'Validate Token' to verify</span>"
            )
        else:
            self.validation_label.setText("")

    def _update_token_visibility(self):
        enabled = self.enable_checkbox.isChecked()
        self.token_widget.setVisible(enabled)
        self.adjustSize()

    def _validate_token(self):
        token = self.token_input.text().strip()

        if not token:
            self.validation_label.setText(
                "<span style='color: red;'>Please enter a token</span>"
            )
            return

        self.validation_label.setText(
            "<span style='color: #666;'>Validating...</span>"
        )
        self.validate_btn.setEnabled(False)

        # Process events to show the "Validating..." message
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            from src.core.diarization import validate_hf_token
            is_valid, message = validate_hf_token(token)

            if is_valid:
                self.validation_label.setText(
                    f"<span style='color: green;'>{message}</span>"
                )
                self._token_validated = True
            else:
                self.validation_label.setText(
                    f"<span style='color: red;'>{message}</span>"
                )
                self._token_validated = False

        except Exception as e:
            self.validation_label.setText(
                f"<span style='color: red;'>Error: {str(e)[:50]}</span>"
            )
            self._token_validated = False

        self.validate_btn.setEnabled(True)

    def _save_settings(self):
        enabled = self.enable_checkbox.isChecked()
        token = self.token_input.text().strip()

        # If enabling, must have validated token
        if enabled:
            if not token:
                QMessageBox.warning(
                    self, "Token Required",
                    "Please enter a HuggingFace token to enable speaker identification."
                )
                return

            # Always validate before saving when enabling
            self._validate_token()
            if not self._token_validated:
                # Validation failed - don't close dialog
                return

        try:
            save_hf_token(token)
            set_speaker_id_enabled(enabled)
            self._save_account_settings()
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # --- the account fields ---------------------------------------------------

    def _load_account_settings(self):
        from src.podcastnotes import auth, glean

        self._select_project(auth.project_id())
        self.glean_instance_input.setText(glean.instance())
        self.glean_token_input.setText(glean.get_token())

    def _save_account_settings(self):
        """Saved without validating.

        The Setup screen is where things are proved to work, and it does that
        by making real calls. Validating here as well would mean two places
        that can disagree about whether a credential is good, and the one
        without the network would win by being first.
        """
        from src.podcastnotes import auth, glean

        # Neither value is tidied here. set_instance and save_token each own
        # their own normalising, and doing it twice means two rules that can
        # drift apart.
        auth.set_project_id(self._chosen_project())
        glean.set_instance(self.glean_instance_input.text())
        glean.save_token(self.glean_token_input.text())
