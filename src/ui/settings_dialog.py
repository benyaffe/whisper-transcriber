"""
Settings dialog for Whisper Transcriber.
Handles speaker identification toggle and HuggingFace token configuration.
"""

import keyring
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGroupBox, QFormLayout, QMessageBox, QCheckBox
)
from PyQt6.QtCore import Qt, QSettings


KEYRING_SERVICE = "WhisperTranscriber"
KEYRING_HF_TOKEN = "hf_token"
SETTINGS_SPEAKER_ID_ENABLED = "speaker_id_enabled"


def get_hf_token() -> str:
    """Retrieve stored HuggingFace token."""
    try:
        token = keyring.get_password(KEYRING_SERVICE, KEYRING_HF_TOKEN)
        return token or ""
    except Exception:
        return ""


def save_hf_token(token: str):
    """Save HuggingFace token to keychain."""
    try:
        if token:
            keyring.set_password(KEYRING_SERVICE, KEYRING_HF_TOKEN, token)
        else:
            try:
                keyring.delete_password(KEYRING_SERVICE, KEYRING_HF_TOKEN)
            except keyring.errors.PasswordDeleteError:
                pass
    except Exception as e:
        raise RuntimeError(f"Failed to save token: {e}")


def is_speaker_id_enabled() -> bool:
    """Check if speaker identification is enabled."""
    settings = QSettings("WhisperTranscriber", "WhisperTranscriber")
    return settings.value(SETTINGS_SPEAKER_ID_ENABLED, False, type=bool)


def set_speaker_id_enabled(enabled: bool):
    """Set speaker identification enabled state."""
    settings = QSettings("WhisperTranscriber", "WhisperTranscriber")
    settings.setValue(SETTINGS_SPEAKER_ID_ENABLED, enabled)


class SettingsDialog(QDialog):
    """Settings dialog for app configuration."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
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
        self.info_label.setStyleSheet("color: #666; margin-left: 20px;")
        speaker_layout.addWidget(self.info_label)

        # Token section (shown when enabled)
        self.token_widget = QGroupBox("HuggingFace Token")
        token_layout = QVBoxLayout(self.token_widget)

        # Requirements
        req_label = QLabel(
            "<b>Requirements:</b><br>"
            "1. Create a free account at <a href='https://huggingface.co/join'>huggingface.co</a><br>"
            "2. Accept the license for <b>all three</b> required models:<br>"
            "&nbsp;&nbsp;&nbsp;- <a href='https://huggingface.co/pyannote/speaker-diarization-3.1'>pyannote/speaker-diarization-3.1</a><br>"
            "&nbsp;&nbsp;&nbsp;- <a href='https://huggingface.co/pyannote/segmentation-3.0'>pyannote/segmentation-3.0</a><br>"
            "&nbsp;&nbsp;&nbsp;- <a href='https://huggingface.co/pyannote/speaker-diarization-community-1'>pyannote/speaker-diarization-community-1</a><br>"
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

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        self.save_btn = QPushButton("Save")
        self.save_btn.setDefault(True)
        self.save_btn.clicked.connect(self._save_settings)
        button_layout.addWidget(self.save_btn)

        layout.addLayout(button_layout)

    def _toggle_token_visibility(self):
        if self.token_input.echoMode() == QLineEdit.EchoMode.Password:
            self.token_input.setEchoMode(QLineEdit.EchoMode.Normal)
            self.show_token_btn.setText("Hide")
        else:
            self.token_input.setEchoMode(QLineEdit.EchoMode.Password)
            self.show_token_btn.setText("Show")

    def _load_settings(self):
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
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
