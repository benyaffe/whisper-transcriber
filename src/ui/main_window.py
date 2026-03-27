"""
Main application window with drag-drop zone, URL input, and transcription controls.
"""

import re
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QProgressBar,
    QTextEdit, QFileDialog, QMessageBox, QFrame,
    QListWidget, QListWidgetItem, QSplitter, QComboBox,
    QSlider, QTextBrowser
)
from PyQt6.QtCore import Qt, pyqtSignal, QUrl
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QTextCursor
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
        if event.button() == Qt.MouseButton.LeftButton:
            self.open_file_dialog()

    def open_file_dialog(self):
        extensions = get_supported_extensions()
        filter_str = f"Media Files ({' '.join('*' + ext for ext in extensions)})"

        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio/Video Files", "", filter_str
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
        self.audio_path = None  # For playback
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


class ClickablePreview(QTextBrowser):
    """Text preview that handles timestamp clicks for audio playback."""

    timestamp_clicked = pyqtSignal(float)  # Emits seconds

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setOpenLinks(False)
        self.anchorClicked.connect(self._handle_anchor)
        self.setStyleSheet("""
            QTextBrowser {
                font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            }
            a {
                color: #2962ff;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
        """)

    def _handle_anchor(self, url: QUrl):
        """Handle clicks on timestamp links."""
        scheme = url.scheme()
        if scheme == "ts":
            # Parse timestamp: ts://123.456
            try:
                seconds = float(url.host())
                self.timestamp_clicked.emit(seconds)
            except ValueError:
                pass


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper Transcriber")
        self.setMinimumSize(800, 700)

        self.transcription_worker = None
        self.downloader = None
        self.queue = []
        self.current_item = None

        # Audio playback
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.current_audio_path = None

        self.setup_ui()

        # Connect player signals
        self.media_player.positionChanged.connect(self._on_position_changed)
        self.media_player.durationChanged.connect(self._on_duration_changed)

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
            "Auto-detect", "English", "Spanish", "French", "German",
            "Italian", "Portuguese", "Dutch", "Russian", "Chinese",
            "Japanese", "Korean", "Arabic", "Hindi", "Turkish",
            "Polish", "Ukrainian", "Vietnamese", "Thai", "Indonesian",
            "Malay", "Swedish", "Norwegian", "Danish", "Finnish",
            "Greek", "Czech", "Romanian", "Hungarian", "Hebrew",
        ])
        self.language_combo.setToolTip("Select source language")
        lang_layout.addWidget(self.language_combo)
        lang_layout.addStretch()
        queue_layout.addLayout(lang_layout)

        splitter.addWidget(queue_widget)

        # Preview panel
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        preview_label = QLabel("Live Preview (click timestamps to play)")
        preview_label.setStyleSheet("font-weight: bold;")
        preview_layout.addWidget(preview_label)

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
        splitter.setSizes([250, 550])

        layout.addWidget(splitter, 1)

        # Status section
        status_layout = QVBoxLayout()

        self.file_info_label = QLabel("No file selected")
        self.file_info_label.setStyleSheet("color: #666;")
        status_layout.addWidget(self.file_info_label)

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

    def _format_time(self, ms: int) -> str:
        """Format milliseconds as m:ss."""
        secs = ms // 1000
        mins = secs // 60
        secs = secs % 60
        return f"{mins}:{secs:02d}"

    def _on_position_changed(self, position: int):
        if not self.position_slider.isSliderDown():
            self.position_slider.setValue(position)
        duration = self.media_player.duration()
        self.time_label.setText(f"{self._format_time(position)} / {self._format_time(duration)}")

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
        """Play audio from a specific timestamp."""
        if self.current_audio_path:
            self.media_player.setPosition(int(seconds * 1000))
            self.media_player.play()
            self.play_btn.setText("⏸")

    def _load_audio(self, audio_path: str):
        """Load audio file for playback."""
        self.current_audio_path = audio_path
        self.media_player.setSource(QUrl.fromLocalFile(audio_path))
        self.play_btn.setEnabled(True)
        self.position_slider.setEnabled(True)

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

        info = get_file_info(item.filepath)
        self.file_info_label.setText(
            f"File: {os.path.basename(item.filepath)} | "
            f"Type: {info.get('format', 'Unknown')} | "
            f"Duration: {info.get('duration_str', 'Unknown')}"
        )

        self.progress_bar.setValue(0)
        self.eta_label.setText("Starting...")

        selected_lang = self.language_combo.currentText()
        language = None if selected_lang == "Auto-detect" else selected_lang.lower()

        self.transcription_worker = TranscriptionWorker(item.filepath, language=language)
        self.transcription_worker.progress.connect(self.on_transcription_progress)
        self.transcription_worker.text_chunk.connect(self.on_text_chunk)
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
            self.eta_label.setText("Calculating...")

    def on_text_chunk(self, text: str):
        """Append text, converting timestamps to clickable links."""
        # Check if this is a timestamped segment [HH:MM:SS]
        timestamp_pattern = r'\[(\d{2}):(\d{2}):(\d{2})\]'

        # For status messages in brackets, just append
        if text.startswith('[') and not re.match(timestamp_pattern, text):
            self.preview_text.append(f"<i style='color: #666;'>{text}</i>")
        else:
            self.preview_text.insertPlainText(text)

        # Auto-scroll
        scrollbar = self.preview_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_language_detected(self, language: str, confidence: float):
        self.preview_text.append(
            f"<i style='color: #666;'>[Detected language: {language} ({confidence:.0%} confidence)]</i>"
        )

    def on_hardware_info(self, info: str):
        self.model_label.setText(f"Model: medium | {info}")

    def on_model_upgraded(self, old_model: str, new_model: str, reason: str):
        self.model_label.setText(f"Model: {new_model} (upgraded)")
        self.preview_text.append(
            f"<i style='color: #666;'>[Model upgraded to {new_model}: {reason}]</i>"
        )

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

        # Load audio for playback
        if audio_path:
            self._load_audio(audio_path)

        # Load VTT with clickable timestamps
        self._display_vtt_with_links(vtt_path)

        # Open result
        subprocess.run(["open", vtt_path], check=False)

        self.process_next()

    def _display_vtt_with_links(self, vtt_path: str):
        """Load VTT and display with clickable timestamp links."""
        try:
            with open(vtt_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse VTT and create HTML with clickable timestamps
            html_parts = ["<div style='font-family: -apple-system, sans-serif;'>"]

            # Match VTT cues: timestamp --> timestamp\ntext
            pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> \d{2}:\d{2}:\d{2}\.\d{3}\n(.+?)(?=\n\n|\n\d+\n|$)'

            for match in re.finditer(pattern, content, re.DOTALL):
                timestamp = match.group(1)
                text = match.group(2).strip()

                # Parse timestamp to seconds
                parts = timestamp.replace(',', '.').split(':')
                hours, mins, secs = int(parts[0]), int(parts[1]), float(parts[2])
                total_secs = hours * 3600 + mins * 60 + secs

                # Format display timestamp
                display_ts = f"{int(mins)}:{int(secs):02d}"
                if hours > 0:
                    display_ts = f"{hours}:{int(mins):02d}:{int(secs):02d}"

                # Handle speaker tags <v Speaker>text</v>
                speaker_match = re.match(r'<v ([^>]+)>(.+)</v>', text, re.DOTALL)
                if speaker_match:
                    speaker = speaker_match.group(1)
                    text = speaker_match.group(2)
                    html_parts.append(
                        f"<p><a href='ts://{total_secs}' style='color: #2962ff;'>[{display_ts}]</a> "
                        f"<b>{speaker}:</b> {text}</p>"
                    )
                else:
                    html_parts.append(
                        f"<p><a href='ts://{total_secs}' style='color: #2962ff;'>[{display_ts}]</a> {text}</p>"
                    )

            html_parts.append("</div>")
            self.preview_text.setHtml('\n'.join(html_parts))

        except Exception as e:
            pass  # Keep existing preview on error

    def on_transcription_error(self, error_msg: str):
        if self.current_item:
            self.current_item.status = "error"
            self.current_item.update_display()

        self.show_error("Transcription Failed", error_msg)
        self.retry_btn.setEnabled(True)
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
        if selected and isinstance(selected, QueueItem):
            if selected.status in ("error", "warning"):
                selected.status = "pending"
                selected.update_display()
                if not self.current_item:
                    self.process_next()

    def clear_completed(self):
        items_to_remove = [
            item for item in self.queue
            if item.status == "completed"
        ]
        for item in items_to_remove:
            self.queue.remove(item)
            row = self.queue_list.row(item)
            self.queue_list.takeItem(row)
