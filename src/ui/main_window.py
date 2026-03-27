"""
Main application window with drag-drop zone, URL input, and transcription controls.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QProgressBar,
    QTextEdit, QFileDialog, QMessageBox, QFrame,
    QListWidget, QListWidgetItem, QSplitter, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QMimeData
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

from core.transcriber import TranscriptionWorker
from core.downloader import VideoDownloader
from utils.file_utils import get_supported_extensions, is_url


class DropZone(QFrame):
    """Drag-and-drop zone that also acts as a clickable file picker."""

    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(120)
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
        """Open file dialog on click."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.open_file_dialog()

    def open_file_dialog(self):
        extensions = get_supported_extensions()
        filter_str = f"Media Files ({' '.join('*' + ext for ext in extensions)})"

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Audio/Video Files",
            "",
            filter_str
        )

        if files:
            self.files_dropped.emit(files)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                DropZone {
                    border: 2px solid #4a90d9;
                    border-radius: 10px;
                    background-color: #d0e4fc;
                }
            """)

    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            DropZone {
                border: 2px dashed #888;
                border-radius: 10px;
                background-color: #f5f5f5;
            }
        """)

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet("""
            DropZone {
                border: 2px dashed #888;
                border-radius: 10px;
                background-color: #f5f5f5;
            }
        """)

        files = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                files.append(path)

        if files:
            self.files_dropped.emit(files)


class QueueItem(QListWidgetItem):
    """Represents a file in the transcription queue."""

    def __init__(self, filepath: str, is_url: bool = False):
        super().__init__()
        self.filepath = filepath
        self.is_url = is_url
        self.status = "pending"
        self.update_display()

    def update_display(self):
        import os
        name = self.filepath if self.is_url else os.path.basename(self.filepath)
        status_icons = {
            "pending": "⏳",
            "downloading": "⬇️",
            "transcribing": "🎙️",
            "completed": "✅",
            "error": "❌",
            "warning": "⚠️"
        }
        icon = status_icons.get(self.status, "")
        self.setText(f"{icon} {name}")


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper Transcriber")
        self.setMinimumSize(700, 600)

        self.transcription_worker = None
        self.downloader = None
        self.queue = []
        self.current_item = None

        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Drop zone
        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self.queue_files)
        layout.addWidget(self.drop_zone)

        # URL input section
        url_layout = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Paste YouTube, Instagram, TikTok URL...")
        self.url_input.returnPressed.connect(self.add_url)
        url_layout.addWidget(self.url_input)

        self.url_button = QPushButton("Add URL")
        self.url_button.clicked.connect(self.add_url)
        url_layout.addWidget(self.url_button)
        layout.addLayout(url_layout)

        # Splitter for queue and preview
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
        self.clear_btn.setToolTip("Remove all completed items from the queue list")
        self.clear_btn.clicked.connect(self.clear_completed)
        queue_controls.addWidget(self.clear_btn)
        queue_layout.addLayout(queue_controls)

        # Language selection
        lang_layout = QHBoxLayout()
        lang_label = QLabel("Language:")
        lang_layout.addWidget(lang_label)

        self.language_combo = QComboBox()
        self.language_combo.addItems([
            "Auto-detect",
            "English",
            "Spanish",
            "French",
            "German",
            "Italian",
            "Portuguese",
            "Dutch",
            "Russian",
            "Chinese",
            "Japanese",
            "Korean",
            "Arabic",
            "Hindi",
            "Turkish",
            "Polish",
            "Ukrainian",
            "Vietnamese",
            "Thai",
            "Indonesian",
            "Malay",
            "Swedish",
            "Norwegian",
            "Danish",
            "Finnish",
            "Greek",
            "Czech",
            "Romanian",
            "Hungarian",
            "Hebrew",
        ])
        self.language_combo.setToolTip("Select source language, or Auto-detect to let Whisper determine it")
        lang_layout.addWidget(self.language_combo)
        lang_layout.addStretch()
        queue_layout.addLayout(lang_layout)

        splitter.addWidget(queue_widget)

        # Preview panel
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        preview_label = QLabel("Live Preview")
        preview_label.setStyleSheet("font-weight: bold;")
        preview_layout.addWidget(preview_label)

        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setPlaceholderText("Transcription will appear here...")
        preview_layout.addWidget(self.preview_text)

        splitter.addWidget(preview_widget)
        splitter.setSizes([250, 450])

        layout.addWidget(splitter, 1)

        # Status section
        status_layout = QVBoxLayout()

        # File info
        self.file_info_label = QLabel("No file selected")
        self.file_info_label.setStyleSheet("color: #666;")
        status_layout.addWidget(self.file_info_label)

        # Progress bar
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.eta_label = QLabel("")
        self.eta_label.setMinimumWidth(100)
        progress_layout.addWidget(self.eta_label)
        status_layout.addLayout(progress_layout)

        # Model info
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

    def queue_files(self, files: list):
        """Add files to the transcription queue."""
        for filepath in files:
            item = QueueItem(filepath, is_url=False)
            self.queue.append(item)
            self.queue_list.addItem(item)

        if not self.current_item:
            self.process_next()

    def add_url(self):
        """Add a URL to the queue for download and transcription."""
        url = self.url_input.text().strip()
        if not url:
            return

        if not is_url(url):
            self.show_error(
                "Invalid URL",
                "Please enter a valid YouTube, Instagram, or TikTok URL.\n\n"
                "Suggestion: Make sure the URL starts with 'http://' or 'https://'"
            )
            return

        item = QueueItem(url, is_url=True)
        self.queue.append(item)
        self.queue_list.addItem(item)
        self.url_input.clear()

        if not self.current_item:
            self.process_next()

    def process_next(self):
        """Process the next item in the queue."""
        # Find next pending item
        for item in self.queue:
            if item.status == "pending":
                self.current_item = item
                break
        else:
            self.current_item = None
            self.cancel_btn.setEnabled(False)
            return

        self.cancel_btn.setEnabled(True)
        self.preview_text.clear()
        self.error_label.setVisible(False)

        if self.current_item.is_url:
            self.download_and_transcribe(self.current_item)
        else:
            self.start_transcription(self.current_item)

    def download_and_transcribe(self, item: QueueItem):
        """Download video from URL, then transcribe."""
        import os

        item.status = "downloading"
        item.update_display()
        self.file_info_label.setText(f"Downloading: {item.filepath}")
        self.progress_bar.setValue(0)
        self.eta_label.setText("Downloading...")

        self.downloader = VideoDownloader(
            url=item.filepath,
            output_dir=os.path.expanduser("~/Downloads")
        )
        self.downloader.progress.connect(self.on_download_progress)
        self.downloader.completed.connect(self.on_download_complete)
        self.downloader.error.connect(self.on_download_error)
        self.downloader.start()

    def on_download_progress(self, percent: float, status: str):
        self.progress_bar.setValue(int(percent))
        self.eta_label.setText(status)

    def on_download_complete(self, filepath: str):
        """Download finished, start transcription."""
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
        """Start transcribing a local file."""
        import os
        from utils.file_utils import get_file_info

        item.status = "transcribing"
        item.update_display()

        # Get and display file info
        info = get_file_info(item.filepath)
        self.file_info_label.setText(
            f"File: {os.path.basename(item.filepath)} | "
            f"Type: {info.get('format', 'Unknown')} | "
            f"Duration: {info.get('duration_str', 'Unknown')}"
        )

        self.progress_bar.setValue(0)
        self.eta_label.setText("Starting...")

        # Get selected language (None for auto-detect)
        selected_lang = self.language_combo.currentText()
        language = None if selected_lang == "Auto-detect" else selected_lang.lower()

        self.transcription_worker = TranscriptionWorker(item.filepath, language=language)
        self.transcription_worker.progress.connect(self.on_transcription_progress)
        self.transcription_worker.text_chunk.connect(self.on_text_chunk)
        self.transcription_worker.language_detected.connect(self.on_language_detected)
        self.transcription_worker.model_upgraded.connect(self.on_model_upgraded)
        self.transcription_worker.quality_warning.connect(self.on_quality_warning)
        self.transcription_worker.completed.connect(self.on_transcription_complete)
        self.transcription_worker.error.connect(self.on_transcription_error)
        self.transcription_worker.start()

    def on_transcription_progress(self, percent: float, eta_seconds: int):
        self.progress_bar.setValue(int(percent))
        if eta_seconds > 0:
            mins, secs = divmod(eta_seconds, 60)
            self.eta_label.setText(f"ETA: {mins}m {secs}s")
        else:
            self.eta_label.setText("Calculating...")

    def on_text_chunk(self, text: str):
        """Append new transcribed text to preview."""
        self.preview_text.append(text)
        # Auto-scroll to bottom
        scrollbar = self.preview_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_language_detected(self, language: str, confidence: float):
        """Show detected language in preview (non-blocking)."""
        self.preview_text.append(
            f"[Detected language: {language} ({confidence:.0%} confidence)]\n"
        )

    def on_model_upgraded(self, old_model: str, new_model: str, reason: str):
        """Notify user of automatic model upgrade."""
        self.model_label.setText(f"Model: {new_model} (upgraded from {old_model})")
        self.preview_text.append(f"\n[Model upgraded to {new_model}: {reason}]\n")

    def on_quality_warning(self, message: str):
        """Show quality warning but continue."""
        if self.current_item:
            self.current_item.status = "warning"
            self.current_item.update_display()
        self.preview_text.append(f"\n[Warning: {message}]\n")

    def on_transcription_complete(self, vtt_path: str, txt_path: str):
        """Transcription finished successfully."""
        import subprocess

        if self.current_item:
            self.current_item.status = "completed"
            self.current_item.update_display()

        self.progress_bar.setValue(100)
        self.eta_label.setText("Complete!")

        # Open the result file
        subprocess.run(["open", vtt_path], check=False)

        self.process_next()

    def on_transcription_error(self, error_msg: str):
        """Handle transcription error."""
        if self.current_item:
            self.current_item.status = "error"
            self.current_item.update_display()

        self.show_error("Transcription Failed", error_msg)
        self.retry_btn.setEnabled(True)
        self.process_next()

    def show_error(self, title: str, message: str):
        """Display error with smart suggestions."""
        from utils.error_handler import get_error_suggestion

        suggestion = get_error_suggestion(message)
        full_message = f"{message}\n\n{suggestion}" if suggestion else message

        self.error_label.setText(f"<b>{title}:</b> {full_message}")
        self.error_label.setVisible(True)

    def cancel_current(self):
        """Cancel the current transcription."""
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
        """Retry the selected failed item."""
        selected = self.queue_list.currentItem()
        if selected and isinstance(selected, QueueItem):
            if selected.status in ("error", "warning"):
                selected.status = "pending"
                selected.update_display()
                if not self.current_item:
                    self.process_next()

    def clear_completed(self):
        """Remove completed items from queue."""
        items_to_remove = [
            item for item in self.queue
            if item.status == "completed"
        ]
        for item in items_to_remove:
            self.queue.remove(item)
            row = self.queue_list.row(item)
            self.queue_list.takeItem(row)
