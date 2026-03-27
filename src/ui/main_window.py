"""
Main application window with drag-drop zone, URL input, and transcription controls.
"""

import re
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QProgressBar,
    QTextEdit, QFileDialog, QMessageBox, QFrame,
    QListWidget, QListWidgetItem, QSplitter, QComboBox,
    QSlider, QTextBrowser, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QUrl, QTimer
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QTextCursor, QTextCharFormat, QColor, QBrush
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

from src.core.transcriber import TranscriptionWorker
from src.core.downloader import VideoDownloader
from src.utils.file_utils import get_supported_extensions, is_url


class DropZone(QFrame):
    """Drag-and-drop zone that also acts as a clickable file picker."""

    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(100)
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        self.setStyleSheet("""
            DropZone {
                border: 2px dashed #888;
                border-radius: 10px;
                background-color: #f5f5f5;
            }
            DropZone:hover {
                border-color: #4a90d9;
                background-color: #e8f0fc;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label = QLabel("Drop audio/video files here\nor click to browse")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: #666; font-size: 14px;")
        layout.addWidget(self.label)

        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.open_file_dialog()

    def open_file_dialog(self):
        extensions = get_supported_extensions()
        filter_str = f"Media Files ({' '.join('*' + ext for ext in extensions)})"
        files, _ = QFileDialog.getOpenFileNames(self, "Select Audio/Video Files", "", filter_str)
        if files:
            self.files_dropped.emit(files)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("DropZone { border: 2px solid #4a90d9; border-radius: 10px; background-color: #d0e4fc; }")

    def dragLeaveEvent(self, event):
        self.setStyleSheet("DropZone { border: 2px dashed #888; border-radius: 10px; background-color: #f5f5f5; }")

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet("DropZone { border: 2px dashed #888; border-radius: 10px; background-color: #f5f5f5; }")
        files = [url.toLocalFile() for url in event.mimeData().urls() if url.toLocalFile()]
        if files:
            self.files_dropped.emit(files)


class QueueItem(QListWidgetItem):
    """Represents a file in the transcription queue."""

    def __init__(self, filepath: str, is_url: bool = False):
        super().__init__()
        self.filepath = filepath
        self.is_url = is_url
        self.status = "pending"
        self.audio_path = None
        self.update_display()

    def update_display(self):
        import os
        name = self.filepath if self.is_url else os.path.basename(self.filepath)
        icons = {"pending": "⏳", "downloading": "⬇️", "transcribing": "🎙️", "completed": "✅", "error": "❌", "warning": "⚠️"}
        self.setText(f"{icons.get(self.status, '')} {name}")


class ClickablePreview(QTextBrowser):
    """Text preview with clickable timestamps."""

    timestamp_clicked = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setOpenLinks(False)
        self.anchorClicked.connect(self._handle_anchor)
        self.setStyleSheet("""
            QTextBrowser { font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 13px; }
            a { color: #2962ff; text-decoration: none; }
            a:hover { text-decoration: underline; }
        """)

    def _handle_anchor(self, url: QUrl):
        if url.scheme() == "ts":
            try:
                self.timestamp_clicked.emit(float(url.host()))
            except ValueError:
                pass


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper Transcriber")
        self.setMinimumSize(900, 750)

        self.transcription_worker = None
        self.downloader = None
        self.queue = []
        self.current_item = None
        self.search_match_count = 0
        self.is_transcribing = False

        # Audio playback
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.current_audio_path = None

        self.setup_ui()

        self.media_player.positionChanged.connect(self._on_position_changed)
        self.media_player.durationChanged.connect(self._on_duration_changed)

    def setup_ui(self):
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

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Queue panel
        queue_widget = QWidget()
        queue_layout = QVBoxLayout(queue_widget)
        queue_layout.setContentsMargins(0, 0, 0, 0)

        queue_label = QLabel("Queue")
        queue_label.setStyleSheet("font-weight: bold;")
        queue_layout.addWidget(queue_label)

        self.queue_list = QListWidget()
        self.queue_list.setMinimumWidth(200)
        queue_layout.addWidget(self.queue_list)

        # Queue controls
        queue_controls = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_current)
        self.cancel_btn.setEnabled(False)
        queue_controls.addWidget(self.cancel_btn)

        self.retry_btn = QPushButton("Retry")
        self.retry_btn.clicked.connect(self.retry_selected)
        self.retry_btn.setEnabled(False)
        queue_controls.addWidget(self.retry_btn)

        self.clear_btn = QPushButton("Remove Finished")
        self.clear_btn.clicked.connect(self.clear_completed)
        queue_controls.addWidget(self.clear_btn)
        queue_layout.addLayout(queue_controls)

        # Language selection
        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("Language:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems([
            "Auto-detect", "English", "Spanish", "French", "German",
            "Italian", "Portuguese", "Dutch", "Russian", "Chinese",
            "Japanese", "Korean", "Arabic", "Hindi", "Turkish",
            "Polish", "Ukrainian", "Vietnamese", "Thai", "Indonesian",
            "Malay", "Swedish", "Norwegian", "Danish", "Finnish",
            "Greek", "Czech", "Romanian", "Hungarian", "Hebrew",
        ])
        lang_layout.addWidget(self.language_combo)
        lang_layout.addStretch()
        queue_layout.addLayout(lang_layout)

        splitter.addWidget(queue_widget)

        # Preview panel
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        # Search bar
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Find:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search transcript...")
        self.search_input.textChanged.connect(self._on_search_changed)
        self.search_input.returnPressed.connect(self._find_next)
        search_layout.addWidget(self.search_input)

        self.search_count_label = QLabel("")
        self.search_count_label.setMinimumWidth(80)
        search_layout.addWidget(self.search_count_label)

        self.find_prev_btn = QPushButton("◀")
        self.find_prev_btn.setFixedWidth(30)
        self.find_prev_btn.clicked.connect(self._find_prev)
        search_layout.addWidget(self.find_prev_btn)

        self.find_next_btn = QPushButton("▶")
        self.find_next_btn.setFixedWidth(30)
        self.find_next_btn.clicked.connect(self._find_next)
        search_layout.addWidget(self.find_next_btn)

        preview_layout.addLayout(search_layout)

        # Search match notification
        self.search_alert = QLabel("")
        self.search_alert.setStyleSheet("background: #ffe082; padding: 4px 8px; border-radius: 4px; font-weight: bold;")
        self.search_alert.setVisible(False)
        preview_layout.addWidget(self.search_alert)

        # Preview text
        preview_header = QLabel("Live Preview (click timestamps to play)")
        preview_header.setStyleSheet("font-weight: bold;")
        preview_layout.addWidget(preview_header)

        self.preview_text = ClickablePreview()
        self.preview_text.timestamp_clicked.connect(self._play_from_timestamp)
        preview_layout.addWidget(self.preview_text)

        # Audio player controls
        player_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedWidth(40)
        self.play_btn.clicked.connect(self._toggle_playback)
        self.play_btn.setEnabled(False)
        player_layout.addWidget(self.play_btn)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setEnabled(False)
        self.position_slider.sliderMoved.connect(self._seek_position)
        player_layout.addWidget(self.position_slider)

        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setMinimumWidth(100)
        player_layout.addWidget(self.time_label)

        preview_layout.addLayout(player_layout)

        splitter.addWidget(preview_widget)
        splitter.setSizes([250, 650])
        layout.addWidget(splitter, 1)

        # Status section
        status_layout = QVBoxLayout()

        self.file_info_label = QLabel("No file selected")
        self.file_info_label.setStyleSheet("color: #666;")
        status_layout.addWidget(self.file_info_label)

        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.progress_bar)

        self.eta_label = QLabel("")
        self.eta_label.setMinimumWidth(100)
        progress_layout.addWidget(self.eta_label)
        status_layout.addLayout(progress_layout)

        self.model_label = QLabel("Model: medium")
        self.model_label.setStyleSheet("color: #666;")
        status_layout.addWidget(self.model_label)

        layout.addLayout(status_layout)

        # Error display
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #c00; padding: 8px; background: #fee;")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

    # --- Search functionality ---

    def _on_search_changed(self, text: str):
        """Highlight all matches when search text changes."""
        self._highlight_search(text)

    def _highlight_search(self, search_text: str):
        """Highlight all occurrences of search text."""
        # Reset formatting
        cursor = self.preview_text.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        fmt = QTextCharFormat()
        fmt.setBackground(QBrush(Qt.GlobalColor.transparent))
        cursor.mergeCharFormat(fmt)

        if not search_text:
            self.search_count_label.setText("")
            self.search_match_count = 0
            return

        # Find and highlight all matches
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

        self.search_match_count = count
        self.search_count_label.setText(f"{count} match{'es' if count != 1 else ''}")

    def _find_next(self):
        """Find next occurrence of search text."""
        search_text = self.search_input.text()
        if not search_text:
            return

        cursor = self.preview_text.textCursor()
        found = self.preview_text.document().find(search_text, cursor)

        if found.isNull():
            # Wrap around to beginning
            found = self.preview_text.document().find(search_text, QTextCursor(self.preview_text.document()))

        if not found.isNull():
            self.preview_text.setTextCursor(found)
            self.preview_text.ensureCursorVisible()

    def _find_prev(self):
        """Find previous occurrence of search text."""
        search_text = self.search_input.text()
        if not search_text:
            return

        cursor = self.preview_text.textCursor()
        cursor.setPosition(cursor.selectionStart())
        found = self.preview_text.document().find(search_text, cursor, QTextCursor.MoveMode.MoveAnchor)

        # Try backwards
        from PyQt6.QtGui import QTextDocument
        found = self.preview_text.document().find(search_text, cursor, QTextDocument.FindFlag.FindBackward)

        if found.isNull():
            # Wrap to end
            end_cursor = QTextCursor(self.preview_text.document())
            end_cursor.movePosition(QTextCursor.MoveOperation.End)
            found = self.preview_text.document().find(search_text, end_cursor, QTextDocument.FindFlag.FindBackward)

        if not found.isNull():
            self.preview_text.setTextCursor(found)
            self.preview_text.ensureCursorVisible()

    def _check_search_match(self, text: str):
        """Check if new text contains search term and alert."""
        search_text = self.search_input.text()
        if search_text and search_text.lower() in text.lower():
            self.search_alert.setText(f"🔍 Found: \"{search_text}\"")
            self.search_alert.setVisible(True)
            # Flash effect
            QTimer.singleShot(3000, lambda: self.search_alert.setVisible(False))
            # Re-highlight
            self._highlight_search(search_text)

    # --- Audio playback ---

    def _format_time(self, ms: int) -> str:
        secs = ms // 1000
        return f"{secs // 60}:{secs % 60:02d}"

    def _on_position_changed(self, position: int):
        if not self.position_slider.isSliderDown():
            self.position_slider.setValue(position)
        self.time_label.setText(f"{self._format_time(position)} / {self._format_time(self.media_player.duration())}")

    def _on_duration_changed(self, duration: int):
        self.position_slider.setRange(0, duration)

    def _toggle_playback(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.play_btn.setText("▶")
        else:
            self.media_player.play()
            self.play_btn.setText("⏸")

    def _seek_position(self, position: int):
        self.media_player.setPosition(position)

    def _play_from_timestamp(self, seconds: float):
        if self.current_audio_path:
            self.media_player.setPosition(int(seconds * 1000))
            self.media_player.play()
            self.play_btn.setText("⏸")

    def _load_audio(self, audio_path: str):
        self.current_audio_path = audio_path
        self.media_player.setSource(QUrl.fromLocalFile(audio_path))
        self.play_btn.setEnabled(True)
        self.position_slider.setEnabled(True)

    # --- Queue management ---

    def queue_files(self, files: list):
        for filepath in files:
            item = QueueItem(filepath, is_url=False)
            self.queue.append(item)
            self.queue_list.addItem(item)
        if not self.current_item:
            self.process_next()

    def add_url(self):
        url = self.url_input.text().strip()
        if not url:
            return
        if not is_url(url):
            self.show_error("Invalid URL", "Please enter a valid URL starting with http:// or https://")
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
            self.download_and_transcribe(self.current_item)
        else:
            self.start_transcription(self.current_item)

    def download_and_transcribe(self, item: QueueItem):
        import os
        item.status = "downloading"
        item.update_display()
        self.file_info_label.setText(f"Downloading: {item.filepath}")
        self.progress_bar.setValue(0)
        self.eta_label.setText("Downloading...")

        self.downloader = VideoDownloader(url=item.filepath, output_dir=os.path.expanduser("~/Downloads"))
        self.downloader.progress.connect(self.on_download_progress)
        self.downloader.completed.connect(self.on_download_complete)
        self.downloader.error.connect(self.on_download_error)
        self.downloader.start()

    def on_download_progress(self, percent: float, status: str):
        self.progress_bar.setValue(int(percent))
        self.eta_label.setText(status)

    def on_download_complete(self, filepath: str):
        if self.current_item:
            self.current_item.filepath = filepath
            self.current_item.is_url = False
            self.start_transcription(self.current_item)

    def on_download_error(self, error_msg: str):
        if self.current_item:
            self.current_item.status = "error"
            self.current_item.update_display()
            self.show_error("Download Failed", error_msg)
            self.retry_btn.setEnabled(True)
            self.process_next()

    def start_transcription(self, item: QueueItem):
        import os
        from src.utils.file_utils import get_file_info

        item.status = "transcribing"
        item.update_display()
        self.is_transcribing = True

        info = get_file_info(item.filepath)
        self.file_info_label.setText(
            f"File: {os.path.basename(item.filepath)} | "
            f"Type: {info.get('format', 'Unknown')} | "
            f"Duration: {info.get('duration_str', 'Unknown')}"
        )

        self.progress_bar.setValue(0)
        self.eta_label.setText("Starting...")

        # Load audio for playback immediately
        self._load_audio(item.filepath)

        selected_lang = self.language_combo.currentText()
        language = None if selected_lang == "Auto-detect" else selected_lang.lower()

        self.transcription_worker = TranscriptionWorker(item.filepath, language=language)
        self.transcription_worker.progress.connect(self.on_transcription_progress)
        self.transcription_worker.text_chunk.connect(self.on_text_chunk)
        self.transcription_worker.segment_ready.connect(self.on_segment_ready)
        self.transcription_worker.language_detected.connect(self.on_language_detected)
        self.transcription_worker.model_upgraded.connect(self.on_model_upgraded)
        self.transcription_worker.quality_warning.connect(self.on_quality_warning)
        self.transcription_worker.hardware_info.connect(self.on_hardware_info)
        self.transcription_worker.completed.connect(self.on_transcription_complete)
        self.transcription_worker.error.connect(self.on_transcription_error)
        self.transcription_worker.start()

    def on_transcription_progress(self, percent: float, eta_seconds: int):
        self.progress_bar.setValue(int(percent))
        if eta_seconds > 0:
            mins, secs = divmod(eta_seconds, 60)
            self.eta_label.setText(f"ETA: {mins}m {secs}s")
        else:
            self.eta_label.setText("Processing...")

    def on_text_chunk(self, text: str):
        """Handle status messages."""
        if text.startswith('['):
            self.preview_text.append(f"<i style='color: #666;'>{text}</i>")
        else:
            self.preview_text.insertPlainText(text)
        scrollbar = self.preview_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_segment_ready(self, start: float, end: float, text: str, speaker: str):
        """Handle a new transcribed segment with timestamp."""
        # Format timestamp
        mins = int(start // 60)
        secs = int(start % 60)
        ts_display = f"{mins}:{secs:02d}"

        # Create clickable timestamp link
        html = f"<a href='ts://{start}' style='color: #2962ff;'>[{ts_display}]</a> {text} "
        self.preview_text.append(html)

        # Auto-scroll
        scrollbar = self.preview_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

        # Check for search match
        self._check_search_match(text)

    def on_language_detected(self, language: str, confidence: float):
        self.preview_text.append(f"<i style='color: #666;'>[Detected: {language} ({confidence:.0%})]</i>")

    def on_hardware_info(self, info: str):
        self.model_label.setText(f"Model: medium | {info}")

    def on_model_upgraded(self, old_model: str, new_model: str, reason: str):
        self.model_label.setText(f"Model: {new_model} (upgraded)")
        self.preview_text.append(f"<i style='color: #666;'>[Upgraded to {new_model}]</i>")

    def on_quality_warning(self, message: str):
        if self.current_item:
            self.current_item.status = "warning"
            self.current_item.update_display()
        self.preview_text.append(f"<i style='color: #c90;'>[Warning: {message}]</i>")

    def on_transcription_complete(self, vtt_path: str, txt_path: str, audio_path: str):
        import subprocess

        if self.current_item:
            self.current_item.status = "completed"
            self.current_item.audio_path = audio_path
            self.current_item.update_display()

        self.progress_bar.setValue(100)
        self.eta_label.setText("Complete!")
        self.is_transcribing = False

        # Update audio path if different
        if audio_path and audio_path != self.current_audio_path:
            self._load_audio(audio_path)

        subprocess.run(["open", vtt_path], check=False)
        self.process_next()

    def on_transcription_error(self, error_msg: str):
        if self.current_item:
            self.current_item.status = "error"
            self.current_item.update_display()
        self.show_error("Transcription Failed", error_msg)
        self.retry_btn.setEnabled(True)
        self.is_transcribing = False
        self.process_next()

    def show_error(self, title: str, message: str):
        from src.utils.error_handler import get_error_suggestion
        suggestion = get_error_suggestion(message)
        full_message = f"{message}\n\n{suggestion}" if suggestion else message
        self.error_label.setText(f"<b>{title}:</b> {full_message}")
        self.error_label.setVisible(True)

    def cancel_current(self):
        if self.transcription_worker:
            self.transcription_worker.cancel()
            self.transcription_worker = None
        if self.downloader:
            self.downloader.cancel()
            self.downloader = None
        if self.current_item:
            self.current_item.status = "error"
            self.current_item.update_display()
        self.process_next()

    def retry_selected(self):
        selected = self.queue_list.currentItem()
        if selected and isinstance(selected, QueueItem) and selected.status in ("error", "warning"):
            selected.status = "pending"
            selected.update_display()
            if not self.current_item:
                self.process_next()

    def clear_completed(self):
        for item in [i for i in self.queue if i.status == "completed"]:
            self.queue.remove(item)
            self.queue_list.takeItem(self.queue_list.row(item))
