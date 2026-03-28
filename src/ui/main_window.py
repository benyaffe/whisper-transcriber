"""
Main application window with drag-drop zone, URL input, and transcription controls.
"""

import os
import time
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QProgressBar,
    QFileDialog, QFrame, QListWidget, QListWidgetItem,
    QSplitter, QComboBox, QSlider, QTextBrowser,
    QMenuBar, QMenu, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QUrl, QTimer
from PyQt6.QtGui import (
    QDragEnterEvent, QDropEvent, QTextCursor,
    QTextCharFormat, QColor, QBrush
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

from src.core.transcriber import TranscriptionWorker
from src.core.downloader import VideoDownloader
from src.utils.file_utils import get_supported_extensions, is_url, validate_input_file, check_ffmpeg_health


class DropZone(QFrame):
    """Drag-and-drop zone with click-to-browse."""

    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(100)
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        self._set_default_style()

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label = QLabel("Drop audio/video files here\nor click to browse")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: #666; font-size: 14px;")
        layout.addWidget(self.label)

        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def _set_default_style(self):
        self.setStyleSheet(
            "DropZone { border: 2px dashed #888; border-radius: 10px; background-color: #f5f5f5; }"
        )

    def _set_hover_style(self):
        self.setStyleSheet(
            "DropZone { border: 2px solid #4a90d9; border-radius: 10px; background-color: #d0e4fc; }"
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            extensions = get_supported_extensions()
            filter_str = f"Media Files ({' '.join('*' + ext for ext in extensions)})"
            files, _ = QFileDialog.getOpenFileNames(self, "Select Audio/Video Files", "", filter_str)
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


class QueueItem(QListWidgetItem):
    """Represents a file in the transcription queue."""

    ICONS = {
        "pending": "",
        "downloading": "",
        "transcribing": "",
        "completed": "",
        "error": "",
        "warning": ""
    }

    def __init__(self, filepath: str, is_url: bool = False):
        super().__init__()
        self.filepath = filepath
        self.is_url = is_url
        self.status = "pending"
        self.audio_path = None
        self.update_display()

    def update_display(self):
        name = self.filepath if self.is_url else os.path.basename(self.filepath)
        icon = self.ICONS.get(self.status, "")
        self.setText(f"{icon} {name}")


class ClickablePreview(QTextBrowser):
    """Text preview with clickable timestamps."""

    timestamp_clicked = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setOpenLinks(False)
        self.anchorClicked.connect(self._handle_anchor)

        # Set document stylesheet for timestamp styling (this actually works in QTextBrowser)
        # Note: Use Menlo as primary - it's guaranteed on macOS
        self.document().setDefaultStyleSheet("""
            body {
                font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 13px;
            }
            a.ts {
                font-family: Menlo, Monaco, Courier;
                color: #2962ff;
                text-decoration: none;
            }
            .status {
                color: #666;
                font-style: italic;
            }
            .warning {
                color: #c90;
                font-style: italic;
            }
        """)

    def _handle_anchor(self, url: QUrl):
        """Handle timestamp link clicks."""
        if url.scheme() == "timestamp":
            try:
                # Use path to avoid QUrl host parsing issues with dots
                seconds = float(url.path().lstrip('/'))
                self.timestamp_clicked.emit(seconds)
            except ValueError:
                pass


class MainWindow(QMainWindow):
    """Main application window."""

    # Monospace font for timestamps
    TIMESTAMP_FONT = "'SF Mono', Menlo, Monaco, 'Courier New', monospace"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper Transcriber")
        self.setMinimumSize(900, 750)

        # State
        self.transcription_worker = None
        self.downloader = None
        self.queue = []
        self.current_item = None
        self.is_transcribing = False

        # Progress throttling
        self._last_progress_update = 0
        self._progress_interval_ms = 500

        # Audio playback
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.current_audio_path = None
        self._slider_pressed = False
        self._audio_loaded = False

        self._setup_ui()
        self._setup_menu()
        self._connect_media_signals()

    def _connect_media_signals(self):
        self.media_player.positionChanged.connect(self._on_position_changed)
        self.media_player.durationChanged.connect(self._on_duration_changed)
        self.media_player.mediaStatusChanged.connect(self._on_media_status_changed)

    def _setup_menu(self):
        menubar = self.menuBar()

        # App menu (macOS shows this under app name)
        app_menu = menubar.addMenu("Whisper Transcriber")
        settings_action = app_menu.addAction("Settings...")
        settings_action.setShortcut("Cmd+,")
        settings_action.triggered.connect(self._open_settings)

    def _open_settings(self):
        from src.ui.settings_dialog import SettingsDialog
        dialog = SettingsDialog(self)
        if dialog.exec():
            # Reload speaker ID setting after dialog closes
            self._load_speaker_id_setting()

    def _load_speaker_id_setting(self):
        """Load speaker ID enabled state from settings."""
        from src.ui.settings_dialog import is_speaker_id_enabled
        self.speaker_id_checkbox.blockSignals(True)
        self.speaker_id_checkbox.setChecked(is_speaker_id_enabled())
        self.speaker_id_checkbox.blockSignals(False)

    def _on_speaker_id_toggled(self, state):
        """Handle speaker ID checkbox toggle."""
        from src.ui.settings_dialog import get_hf_token, set_speaker_id_enabled
        from PyQt6.QtWidgets import QMessageBox

        enabled = state == Qt.CheckState.Checked.value

        if enabled:
            # Check network connectivity first
            from src.utils.file_utils import check_network_connectivity
            net_ok, net_msg = check_network_connectivity()
            if not net_ok:
                QMessageBox.warning(
                    self, "Network Required",
                    f"{net_msg}\n\nPlease connect to the internet and try again."
                )
                self.speaker_id_checkbox.blockSignals(True)
                self.speaker_id_checkbox.setChecked(False)
                self.speaker_id_checkbox.blockSignals(False)
                return

            # Check if token exists
            token = get_hf_token()
            if not token:
                # No token - open settings dialog
                QMessageBox.information(
                    self, "Setup Required",
                    "Speaker identification requires a HuggingFace account and access token.\n\n"
                    "The Settings dialog will open to guide you through setup."
                )
                self.speaker_id_checkbox.blockSignals(True)
                self.speaker_id_checkbox.setChecked(False)
                self.speaker_id_checkbox.blockSignals(False)
                self._open_settings()
                return

            # Token exists - validate it (always validate, don't assume previous validation)
            from src.core.diarization import validate_hf_token

            # Show brief validation message
            self.speaker_id_checkbox.setEnabled(False)
            self.speaker_id_checkbox.setText("Validating...")
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()

            is_valid, message = validate_hf_token(token)

            self.speaker_id_checkbox.setText("Identify Speakers")
            self.speaker_id_checkbox.setEnabled(True)

            if not is_valid:
                QMessageBox.warning(
                    self, "Token Invalid",
                    f"Your HuggingFace token could not be validated:\n\n{message}\n\n"
                    "Please update your token in Settings."
                )
                self.speaker_id_checkbox.blockSignals(True)
                self.speaker_id_checkbox.setChecked(False)
                self.speaker_id_checkbox.blockSignals(False)
                self._open_settings()
                return

        # Save the setting
        set_speaker_id_enabled(enabled)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # Drop zone
        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self.queue_files)
        layout.addWidget(self.drop_zone)

        # URL input
        url_layout = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Paste YouTube, Instagram, TikTok URL...")
        self.url_input.returnPressed.connect(self.add_url)
        url_layout.addWidget(self.url_input)
        self.url_button = QPushButton("Add URL")
        self.url_button.clicked.connect(self.add_url)
        url_layout.addWidget(self.url_button)
        layout.addLayout(url_layout)

        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._create_queue_panel())
        splitter.addWidget(self._create_preview_panel())
        splitter.setSizes([250, 650])
        layout.addWidget(splitter, 1)

        # Status section
        layout.addLayout(self._create_status_section())

        # Error display
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #c00; padding: 8px; background: #fee;")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

    def _create_queue_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("<b>Queue</b>"))

        self.queue_list = QListWidget()
        self.queue_list.setMinimumWidth(200)
        self.queue_list.currentItemChanged.connect(self._on_queue_selection_changed)
        layout.addWidget(self.queue_list)

        # Controls
        controls = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_current)
        self.cancel_btn.setEnabled(False)
        controls.addWidget(self.cancel_btn)

        self.retry_btn = QPushButton("Retry")
        self.retry_btn.clicked.connect(self.retry_selected)
        self.retry_btn.setEnabled(False)
        controls.addWidget(self.retry_btn)

        self.clear_btn = QPushButton("Clear Done")
        self.clear_btn.setToolTip("Remove completed, failed, and cancelled items")
        self.clear_btn.clicked.connect(self.clear_completed)
        controls.addWidget(self.clear_btn)
        layout.addLayout(controls)

        # Language selection
        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("Language:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems([
            "Auto-detect", "English", "Spanish", "French", "German",
            "Italian", "Portuguese", "Dutch", "Russian", "Chinese",
            "Japanese", "Korean", "Arabic", "Hindi", "Turkish",
            "Polish", "Ukrainian", "Vietnamese", "Thai", "Indonesian",
            "Malay", "Swedish", "Norwegian", "Danish", "Finnish",
            "Greek", "Czech", "Romanian", "Hungarian", "Hebrew",
        ])
        lang_row.addWidget(self.language_combo)
        lang_row.addStretch()
        layout.addLayout(lang_row)

        # Model selection
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setCurrentText("medium")
        self.model_combo.setToolTip("Auto-upgrades to large if quality is low")
        model_row.addWidget(self.model_combo)
        model_row.addStretch()
        layout.addLayout(model_row)

        # Speaker ID toggle
        self.speaker_id_checkbox = QCheckBox("Identify Speakers")
        self.speaker_id_checkbox.setToolTip("Requires HuggingFace token - configure in Settings")
        self.speaker_id_checkbox.stateChanged.connect(self._on_speaker_id_toggled)
        self._load_speaker_id_setting()
        layout.addWidget(self.speaker_id_checkbox)

        return widget

    def _create_preview_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Search bar
        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Find:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search transcript...")
        self.search_input.textChanged.connect(self._highlight_search)
        self.search_input.returnPressed.connect(self._find_next)
        search_row.addWidget(self.search_input)

        self.search_count_label = QLabel("")
        self.search_count_label.setMinimumWidth(80)
        search_row.addWidget(self.search_count_label)

        self.find_prev_btn = QPushButton("")
        self.find_prev_btn.setFixedWidth(30)
        self.find_prev_btn.clicked.connect(self._find_prev)
        search_row.addWidget(self.find_prev_btn)

        self.find_next_btn = QPushButton("")
        self.find_next_btn.setFixedWidth(30)
        self.find_next_btn.clicked.connect(self._find_next)
        search_row.addWidget(self.find_next_btn)
        layout.addLayout(search_row)

        # Search alert
        self.search_alert = QLabel("")
        self.search_alert.setStyleSheet(
            "background: #ffe082; padding: 4px 8px; border-radius: 4px; font-weight: bold;"
        )
        self.search_alert.setVisible(False)
        layout.addWidget(self.search_alert)

        # Preview
        layout.addWidget(QLabel("<b>Live Preview (click timestamps to play)</b>"))
        self.preview_text = ClickablePreview()
        self.preview_text.timestamp_clicked.connect(self._play_from_timestamp)
        layout.addWidget(self.preview_text)

        # Audio controls
        player_row = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.setFixedWidth(50)
        self.play_btn.clicked.connect(self._toggle_playback)
        self.play_btn.setEnabled(False)
        player_row.addWidget(self.play_btn)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setEnabled(False)
        self.position_slider.sliderMoved.connect(self._seek_position)
        self.position_slider.sliderPressed.connect(lambda: setattr(self, '_slider_pressed', True))
        self.position_slider.sliderReleased.connect(self._on_slider_released)
        player_row.addWidget(self.position_slider)

        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setMinimumWidth(100)
        player_row.addWidget(self.time_label)
        layout.addLayout(player_row)

        return widget

    def _create_status_section(self) -> QVBoxLayout:
        layout = QVBoxLayout()

        self.file_info_label = QLabel("No file selected")
        self.file_info_label.setStyleSheet("color: #666;")
        layout.addWidget(self.file_info_label)

        progress_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_row.addWidget(self.progress_bar)

        self.eta_label = QLabel("")
        self.eta_label.setFixedWidth(140)
        self.eta_label.setStyleSheet("font-family: monospace;")
        progress_row.addWidget(self.eta_label)
        layout.addLayout(progress_row)

        self.model_label = QLabel("Model: -")
        self.model_label.setStyleSheet("color: #666;")
        layout.addWidget(self.model_label)

        return layout

    # --- Search ---

    def _highlight_search(self, search_text: str = None):
        if search_text is None:
            search_text = self.search_input.text()

        # Clear highlights
        cursor = self.preview_text.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        fmt = QTextCharFormat()
        fmt.setBackground(QBrush(Qt.GlobalColor.transparent))
        cursor.mergeCharFormat(fmt)

        if not search_text:
            self.search_count_label.setText("")
            return

        # Highlight matches
        doc = self.preview_text.document()
        highlight_fmt = QTextCharFormat()
        highlight_fmt.setBackground(QBrush(QColor("#ffeb3b")))

        cursor = QTextCursor(doc)
        count = 0
        while True:
            cursor = doc.find(search_text, cursor)
            if cursor.isNull():
                break
            cursor.mergeCharFormat(highlight_fmt)
            count += 1

        self.search_count_label.setText(f"{count} match{'es' if count != 1 else ''}")

    def _find_next(self):
        text = self.search_input.text()
        if not text:
            return
        cursor = self.preview_text.textCursor()
        found = self.preview_text.document().find(text, cursor)
        if found.isNull():
            found = self.preview_text.document().find(text, QTextCursor(self.preview_text.document()))
        if not found.isNull():
            self.preview_text.setTextCursor(found)
            self.preview_text.ensureCursorVisible()

    def _find_prev(self):
        from PyQt6.QtGui import QTextDocument
        text = self.search_input.text()
        if not text:
            return
        cursor = self.preview_text.textCursor()
        cursor.setPosition(cursor.selectionStart())
        found = self.preview_text.document().find(text, cursor, QTextDocument.FindFlag.FindBackward)
        if found.isNull():
            end_cursor = QTextCursor(self.preview_text.document())
            end_cursor.movePosition(QTextCursor.MoveOperation.End)
            found = self.preview_text.document().find(text, end_cursor, QTextDocument.FindFlag.FindBackward)
        if not found.isNull():
            self.preview_text.setTextCursor(found)
            self.preview_text.ensureCursorVisible()

    def _check_search_match(self, text: str):
        search = self.search_input.text()
        if search and search.lower() in text.lower():
            self.search_alert.setText(f"Found: \"{search}\"")
            self.search_alert.setVisible(True)
            QTimer.singleShot(3000, lambda: self.search_alert.setVisible(False))
            self._highlight_search()

    # --- Audio ---

    def _format_time(self, ms: int) -> str:
        secs = ms // 1000
        return f"{secs // 60}:{secs % 60:02d}"

    def _on_position_changed(self, position: int):
        if not self._slider_pressed:
            self.position_slider.setValue(position)
        self.time_label.setText(
            f"{self._format_time(position)} / {self._format_time(self.media_player.duration())}"
        )

    def _on_duration_changed(self, duration: int):
        self.position_slider.setRange(0, duration)

    def _on_media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            self._audio_loaded = True

    def _on_slider_released(self):
        self._slider_pressed = False
        self.media_player.setPosition(self.position_slider.value())

    def _toggle_playback(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.play_btn.setText("Play")
        else:
            self.media_player.play()
            self.play_btn.setText("Pause")

    def _seek_position(self, position: int):
        self.media_player.setPosition(position)

    def _play_from_timestamp(self, seconds: float):
        if self.current_audio_path:
            self.media_player.setPosition(int(seconds * 1000))
            self.media_player.play()
            self.play_btn.setText("Pause")

    def _load_audio(self, audio_path: str):
        self.current_audio_path = audio_path
        self._audio_loaded = False
        self.media_player.setSource(QUrl.fromLocalFile(audio_path))
        self.play_btn.setEnabled(True)
        self.position_slider.setEnabled(True)

    # --- Queue ---

    def queue_files(self, files: list):
        from PyQt6.QtWidgets import QMessageBox

        invalid_files = []
        valid_count = 0

        for filepath in files:
            # Validate each file before queueing
            is_valid, message = validate_input_file(filepath)
            if is_valid:
                item = QueueItem(filepath, is_url=False)
                self.queue.append(item)
                self.queue_list.addItem(item)
                valid_count += 1
            else:
                invalid_files.append(message)

        # Show validation errors if any
        if invalid_files:
            error_text = "\n".join(f"• {msg}" for msg in invalid_files[:5])
            if len(invalid_files) > 5:
                error_text += f"\n... and {len(invalid_files) - 5} more"
            QMessageBox.warning(
                self, "Some Files Skipped",
                f"The following files could not be added:\n\n{error_text}"
            )

        if valid_count > 0 and not self.current_item:
            self.process_next()

    def add_url(self):
        url = self.url_input.text().strip()
        if not url:
            return
        if not is_url(url):
            self._show_error("Invalid URL", "Please enter a valid URL starting with http:// or https://")
            return
        item = QueueItem(url, is_url=True)
        self.queue.append(item)
        self.queue_list.addItem(item)
        self.url_input.clear()
        if not self.current_item:
            self.process_next()

    def process_next(self):
        for item in self.queue:
            if item.status == "pending":
                self.current_item = item
                break
        else:
            self.current_item = None
            self.cancel_btn.setEnabled(False)
            self.is_transcribing = False
            return

        self.cancel_btn.setEnabled(True)
        self.preview_text.clear()
        self.error_label.setVisible(False)
        self.search_alert.setVisible(False)

        if self.current_item.is_url:
            self._download_and_transcribe(self.current_item)
        else:
            self._start_transcription(self.current_item)

    def _download_and_transcribe(self, item: QueueItem):
        item.status = "downloading"
        item.update_display()
        self.file_info_label.setText(f"Downloading: {item.filepath}")
        self.progress_bar.setValue(0)
        self.eta_label.setText("Downloading...")

        self.downloader = VideoDownloader(
            url=item.filepath,
            output_dir=os.path.expanduser("~/Downloads")
        )
        self.downloader.progress.connect(self._on_download_progress)
        self.downloader.completed.connect(self._on_download_complete)
        self.downloader.error.connect(self._on_download_error)
        self.downloader.start()

    def _on_download_progress(self, percent: float, status: str):
        now = int(time.time() * 1000)
        if percent < 100 and now - self._last_progress_update < self._progress_interval_ms:
            return
        self._last_progress_update = now
        self.progress_bar.setValue(int(percent))
        self.eta_label.setText(status[:20].ljust(20))

    def _on_download_complete(self, filepath: str):
        if self.current_item:
            self.current_item.filepath = filepath
            self.current_item.is_url = False
            self._start_transcription(self.current_item)

    def _on_download_error(self, error_msg: str):
        if self.current_item:
            self.current_item.status = "error"
            self.current_item.update_display()
        self._show_error("Download Failed", error_msg)
        self.retry_btn.setEnabled(True)
        self.process_next()

    def _start_transcription(self, item: QueueItem):
        from src.utils.file_utils import get_file_info
        from PyQt6.QtWidgets import QMessageBox

        # FFmpeg health check (only needed for video files)
        info = get_file_info(item.filepath)
        if info.get('has_video'):
            ffmpeg_ok, ffmpeg_msg = check_ffmpeg_health()
            if not ffmpeg_ok:
                QMessageBox.critical(
                    self, "FFmpeg Error",
                    f"{ffmpeg_msg}\n\nVideo files require FFmpeg for audio extraction."
                )
                item.status = "error"
                item.update_display()
                self.process_next()
                return

        item.status = "transcribing"
        item.update_display()
        self.is_transcribing = True
        self.file_info_label.setText(
            f"File: {os.path.basename(item.filepath)} | "
            f"Type: {info.get('format', 'Unknown')} | "
            f"Duration: {info.get('duration_str', 'Unknown')}"
        )

        self.progress_bar.setValue(0)
        self.eta_label.setText("Starting...")

        selected_lang = self.language_combo.currentText()
        language = None if selected_lang == "Auto-detect" else selected_lang.lower()
        model = self.model_combo.currentText()

        self.transcription_worker = TranscriptionWorker(item.filepath, initial_model=model, language=language)
        self.transcription_worker.status_message.connect(self._on_status_message)
        self.transcription_worker.progress.connect(self._on_transcription_progress)
        self.transcription_worker.segment_ready.connect(self._on_segment_ready)
        self.transcription_worker.language_detected.connect(self._on_language_detected)
        self.transcription_worker.model_upgraded.connect(self._on_model_upgraded)
        self.transcription_worker.quality_warning.connect(self._on_quality_warning)
        self.transcription_worker.hardware_info.connect(self._on_hardware_info)
        self.transcription_worker.audio_ready.connect(self._load_audio)
        self.transcription_worker.completed.connect(self._on_transcription_complete)
        self.transcription_worker.error.connect(self._on_transcription_error)
        self.transcription_worker.start()

    def _on_transcription_progress(self, percent: float, eta_seconds: int):
        now = int(time.time() * 1000)
        # Always allow speaker ID updates (eta_seconds < 0) through without throttle
        if eta_seconds >= 0 and percent < 100 and now - self._last_progress_update < self._progress_interval_ms:
            return
        self._last_progress_update = now

        self.progress_bar.setValue(int(percent))
        if eta_seconds > 0:
            mins, secs = divmod(eta_seconds, 60)
            self.eta_label.setText(f"ETA: {mins:2d}m {secs:02d}s")
        elif eta_seconds < 0:
            self.eta_label.setText("Speaker ID...  ")
        else:
            self.eta_label.setText("Processing...  ")

    def _is_at_bottom(self) -> bool:
        scrollbar = self.preview_text.verticalScrollBar()
        return scrollbar.value() >= scrollbar.maximum() - 20

    def _scroll_to_bottom(self):
        if self._is_at_bottom():
            scrollbar = self.preview_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def _on_status_message(self, message: str):
        """Handle status messages (gray italic text)."""
        was_at_bottom = self._is_at_bottom()
        # Escape HTML and render with status class
        escaped = message.replace('<', '&lt;').replace('>', '&gt;')
        self.preview_text.append(f"<span class='status'>{escaped}</span>")
        if was_at_bottom:
            self._scroll_to_bottom()

    def _on_segment_ready(self, start: float, end: float, text: str, speaker: str):
        """Handle transcribed segment with clickable timestamp."""
        was_at_bottom = self._is_at_bottom()

        mins = int(start // 60)
        secs = int(start % 60)
        ts_display = f"{mins}:{secs:02d}"

        # Use class 'ts' which is styled via document stylesheet as monospace
        html = f"<a href='timestamp:///{start}' class='ts'>[{ts_display}]</a> {text}"
        self.preview_text.append(html)

        if was_at_bottom:
            self._scroll_to_bottom()

        self._check_search_match(text)

    def _on_language_detected(self, language: str, confidence: float):
        self._on_status_message(f"[Detected: {language} ({confidence:.0%})]")

    def _on_hardware_info(self, info: str):
        self.model_label.setText(f"Model: {self.model_combo.currentText()} | {info}")

    def _on_model_upgraded(self, old_model: str, new_model: str, reason: str):
        self.model_label.setText(f"Model: {new_model} (upgraded)")
        self._on_status_message(f"[Upgraded to {new_model}: {reason}]")

    def _on_quality_warning(self, message: str):
        if self.current_item:
            self.current_item.status = "warning"
            self.current_item.update_display()
        self.preview_text.append(f"<i style='color: #c90;'>[Warning: {message}]</i>")

    def _on_transcription_complete(self, vtt_path: str, txt_path: str, audio_path: str):
        import subprocess

        if self.current_item:
            self.current_item.status = "completed"
            self.current_item.audio_path = audio_path
            self.current_item.update_display()

        self.progress_bar.setValue(100)
        self.eta_label.setText("Complete!")
        self.is_transcribing = False

        if audio_path and audio_path != self.current_audio_path:
            self._load_audio(audio_path)

        subprocess.run(["open", vtt_path], check=False)
        self.process_next()

    def _on_transcription_error(self, error_msg: str):
        if self.current_item:
            self.current_item.status = "error"
            self.current_item.update_display()
        self._show_error("Transcription Failed", error_msg)
        self.retry_btn.setEnabled(True)
        self.is_transcribing = False
        self.process_next()

    def _show_error(self, title: str, message: str):
        from src.utils.error_handler import get_error_suggestion
        suggestion = get_error_suggestion(message)
        full = f"{message}\n\n{suggestion}" if suggestion else message
        self.error_label.setText(f"<b>{title}:</b> {full}")
        self.error_label.setVisible(True)

    def cancel_current(self):
        if self.transcription_worker:
            self.transcription_worker.cancel()
            # Wait briefly for thread to acknowledge cancellation
            self.transcription_worker.wait(500)
            self.transcription_worker = None
        if self.downloader:
            self.downloader.cancel()
            self.downloader.wait(500)
            self.downloader = None
        if self.current_item:
            self.current_item.status = "error"
            self.current_item.update_display()
        self.is_transcribing = False
        self.cancel_btn.setEnabled(False)
        self.eta_label.setText("Cancelled")
        self.process_next()

    def retry_selected(self):
        selected = self.queue_list.currentItem()
        if selected and isinstance(selected, QueueItem) and selected.status in ("error", "warning", "completed"):
            selected.status = "pending"
            selected.update_display()
            self.retry_btn.setEnabled(False)
            if not self.current_item:
                self.process_next()

    def _on_queue_selection_changed(self, current, previous):
        """Enable retry button when a retryable item is selected."""
        if current and isinstance(current, QueueItem):
            can_retry = current.status in ("error", "warning", "completed") and not self.is_transcribing
            self.retry_btn.setEnabled(can_retry)
        else:
            self.retry_btn.setEnabled(False)

    def clear_completed(self):
        """Remove all finished items (completed, error, warning) from queue."""
        finished_statuses = ("completed", "error", "warning")
        for item in [i for i in self.queue if i.status in finished_statuses]:
            self.queue.remove(item)
            self.queue_list.takeItem(self.queue_list.row(item))
