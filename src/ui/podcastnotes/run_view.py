"""
The screen a trip runs on, and the one it finishes on.

Reuses the live transcript and the audio player from the old app rather than
rebuilding them: watching the text appear is the main reassurance that a
twenty-minute job is working, and being able to click a timestamp and hear that
moment is needed for confirming who is speaking later anyway.
"""

import os
import subprocess

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from src.ui.theme import role
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QProgressBar, QPushButton, QSlider, QVBoxLayout, QWidget,
)

from src.ui.widgets import ClickablePreview


class RunView(QWidget):
    """Progress, the transcript as it appears, and where the files ended up."""

    cancel_requested = pyqtSignal()
    new_trip_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._audio_path = None
        self._outputs = None
        self._build()

    # --- construction ---------------------------------------------------------

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        self.title = QLabel("")
        role(self.title, "h2")
        layout.addWidget(self.title)

        progress_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_row.addWidget(self.progress_bar)
        self.stage_label = QLabel("")
        # Wide enough for the longest stage text plus a count, measured rather
        # than guessed; a clipped label defeats the point of having one.
        self.stage_label.setFixedWidth(230)
        progress_row.addWidget(self.stage_label)
        layout.addLayout(progress_row)

        self.transcript = ClickablePreview()
        self.transcript.timestamp_clicked.connect(self._play_from)
        layout.addWidget(self.transcript, 1)

        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.positionChanged.connect(self._on_position_changed)
        self.media_player.durationChanged.connect(lambda d: self.position_slider.setRange(0, d))

        player_row = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setFixedWidth(60)
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self._toggle_playback)
        player_row.addWidget(self.play_button)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setEnabled(False)
        self.position_slider.sliderMoved.connect(self.media_player.setPosition)
        player_row.addWidget(self.position_slider)

        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setMinimumWidth(100)
        player_row.addWidget(self.time_label)
        layout.addLayout(player_row)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setVisible(False)
        layout.addWidget(self.summary)

        buttons = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch()

        self.reveal_button = QPushButton("Show files")
        self.reveal_button.setVisible(False)
        self.reveal_button.clicked.connect(self._reveal)
        buttons.addWidget(self.reveal_button)

        self.new_trip_button = QPushButton("New trip")
        self.new_trip_button.setVisible(False)
        self.new_trip_button.clicked.connect(self.new_trip_requested.emit)
        buttons.addWidget(self.new_trip_button)
        layout.addLayout(buttons)

    # --- running --------------------------------------------------------------

    def begin(self, trip_name: str, source_count: int):
        """Reset for a fresh run."""
        self._outputs = None
        self._audio_path = None
        self.title.setText(trip_name)
        self.transcript.clear()
        self.progress_bar.setValue(0)
        self.stage_label.setText("Starting")
        self.summary.setVisible(False)
        self.reveal_button.setVisible(False)
        self.new_trip_button.setVisible(False)
        self.cancel_button.setVisible(True)
        self.play_button.setEnabled(False)
        self.position_slider.setEnabled(False)
        recordings = "recording" if source_count == 1 else "recordings"
        self.append_status(f"[{source_count} {recordings}]")

    def on_progress(self, percent: float, stage: str):
        self.progress_bar.setValue(int(percent))
        self.stage_label.setText(stage)

    def append_status(self, message: str):
        at_bottom = self._at_bottom()
        escaped = message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.transcript.append(f"<span class='status'>{escaped}</span>")
        if at_bottom:
            self._scroll_to_bottom()

    def append_segment(self, start: float, end: float, text: str, speaker: str):
        at_bottom = self._at_bottom()
        stamp = f"{int(start // 60)}:{int(start % 60):02d}"
        who = f"<b>{speaker}:</b> " if speaker else ""
        self.transcript.append(
            f"<a href='timestamp:///{start}' class='ts'>[{stamp}]</a> {who}{text}"
        )
        if at_bottom:
            self._scroll_to_bottom()

    def set_audio(self, path: str):
        self._audio_path = path
        self.media_player.setSource(QUrl.fromLocalFile(path))
        self.play_button.setEnabled(True)
        self.position_slider.setEnabled(True)

    # --- finishing ------------------------------------------------------------

    def on_finished(self, outputs):
        self._outputs = outputs
        self.progress_bar.setValue(100)
        self.stage_label.setText("Done")
        self.cancel_button.setVisible(False)
        self.reveal_button.setVisible(True)
        self.new_trip_button.setVisible(True)

        speakers = ""
        if outputs.speaker_count:
            who = "speaker" if outputs.speaker_count == 1 else "speakers"
            speakers = f", {outputs.speaker_count} {who}"
        self.summary.setText(
            f"Saved to <b>{os.path.basename(outputs.work_dir)}</b>{speakers}. "
            f"Transcript, plain text and data file are all in there."
        )
        self.summary.setVisible(True)

    def on_failed(self, message: str):
        self.stage_label.setText("Stopped")
        self.cancel_button.setVisible(False)
        self.new_trip_button.setVisible(True)
        self.summary.setText(f"<b>Could not finish:</b> {message}")
        self.summary.setVisible(True)

    def _reveal(self):
        if self._outputs:
            subprocess.run(["open", self._outputs.work_dir], check=False)

    # --- playback -------------------------------------------------------------

    @staticmethod
    def _format_time(ms: int) -> str:
        seconds = ms // 1000
        return f"{seconds // 60}:{seconds % 60:02d}"

    def _on_position_changed(self, position: int):
        if not self.position_slider.isSliderDown():
            self.position_slider.setValue(position)
        self.time_label.setText(
            f"{self._format_time(position)} / {self._format_time(self.media_player.duration())}"
        )

    def _toggle_playback(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.play_button.setText("Play")
        else:
            self.media_player.play()
            self.play_button.setText("Pause")

    def _play_from(self, seconds: float):
        if not self._audio_path:
            return
        self.media_player.setPosition(int(seconds * 1000))
        self.media_player.play()
        self.play_button.setText("Pause")

    # --- scrolling ------------------------------------------------------------

    def _at_bottom(self) -> bool:
        bar = self.transcript.verticalScrollBar()
        return bar.value() >= bar.maximum() - 20

    def _scroll_to_bottom(self):
        bar = self.transcript.verticalScrollBar()
        bar.setValue(bar.maximum())
