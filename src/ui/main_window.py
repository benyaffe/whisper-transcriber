"""
The application window.

One flow: describe a trip, then watch it run. A single recording is a trip of
one, and a lecture or a webinar is a trip too, because both have context worth
feeding in and a summary worth producing.

This replaced a queue-based transcriber where each file was an independent job.
That framing is gone: several unrelated recordings are several trips, run one
after another, not one queue.
"""

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMainWindow, QMessageBox, QStackedWidget

from src.core.config import get_hf_token
from src.podcastnotes.project import TripProject
from src.podcastnotes.readiness import remember_working, was_working_last_time
from src.ui.podcastnotes.intake_view import IntakeView
from src.ui.podcastnotes.run_view import RunView
from src.ui.podcastnotes.setup_view import SetupView
from src.ui.podcastnotes.trip_worker import TripWorker
from src.utils.file_utils import check_ffmpeg_health
from src.utils.logger import get_logger


class MainWindow(QMainWindow):
    """Describe a trip, run it, see where the files went."""

    # How quiet the pipeline has to go before we say so, and how often to look.
    STALL_WARN_AFTER_S = 120
    STALL_POLL_MS = 15000

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PodcastNotesWT")
        self.setMinimumSize(900, 750)

        self._logger = get_logger()
        self.worker = None
        self._stall_timer = None
        self._stall_warned = False

        self.setup_view = SetupView()
        self.setup_view.ready.connect(self._show_intake)
        self.setup_view.settings_requested.connect(self._open_settings)

        self.intake = IntakeView()
        self.intake.start_requested.connect(self._start_trip)
        self.intake.settings_requested.connect(self._open_settings)

        self.run_view = RunView()
        self.run_view.cancel_requested.connect(self._cancel_trip)
        self.run_view.new_trip_requested.connect(self._show_intake)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.setup_view)
        self.stack.addWidget(self.intake)
        self.stack.addWidget(self.run_view)
        # Stated rather than left to insertion order. Whether the checklist
        # takes the screen is check_setup_on_launch's decision and nothing
        # else's, and a stack that quietly defaults to it would make that
        # policy depend on which line addWidget was called on.
        self.stack.setCurrentWidget(self.intake)
        self.setCentralWidget(self.stack)

        self._setup_menu()

    def _setup_menu(self):
        app_menu = self.menuBar().addMenu("PodcastNotesWT")
        settings_action = app_menu.addAction("Settings...")
        settings_action.setShortcut("Cmd+,")
        settings_action.triggered.connect(self._open_settings)
        setup_action = app_menu.addAction("Setup...")
        setup_action.triggered.connect(self.show_setup)

    def show_setup(self):
        """Open the checklist and run it.

        Reachable from the menu at any time, not only when something has
        already broken, because "is this thing working?" is a fair question to
        ask before starting a forty-minute trip rather than after.
        """
        self.stack.setCurrentWidget(self.setup_view)
        self.setup_view.start_checks()

    def check_setup_on_launch(self):
        """Decide what to open on, then verify in the background.

        Three cases, and they want different things. Somebody who has never
        had this working should land on the list. Somebody whose setup was
        fine last time should land on the trip form and not have to dismiss a
        checklist every morning. Somebody whose token expired overnight looks
        identical to the second case until the checks come back, so the app
        opens on the form and moves them to the list only if something has
        actually broken.

        The interruption is therefore only ever paid by the person who needs
        it, which is the whole point.
        """
        if not was_working_last_time():
            self.show_setup()
            return

        self.stack.setCurrentWidget(self.intake)
        self.setup_view.all_done.connect(self._on_launch_checks_done)
        self.setup_view.start_checks()

    def _on_launch_checks_done(self, everything_works: bool):
        remember_working(everything_works)
        if everything_works:
            return
        # Do not yank the screen out from under somebody who has already
        # started describing a trip.
        if self.stack.currentWidget() is self.intake and not self.intake.has_input():
            self.stack.setCurrentWidget(self.setup_view)

    def _open_settings(self):
        from src.ui.settings_dialog import SettingsDialog

        SettingsDialog(self).exec()

    # --- entry points ---------------------------------------------------------

    def open_files(self, paths: list):
        """Recordings arriving from outside: launch arguments, or the Dock.

        They join the trip being described. If one is already running the
        operator is told, rather than the files being silently dropped, which
        is what the previous version did.
        """
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(
                self, "Trip in progress",
                "A trip is already running. Wait for it to finish, then start a new one.",
            )
            return
        self._show_intake()
        self.intake.add_sources(paths)

    def _show_intake(self):
        self.stack.setCurrentWidget(self.intake)

    # --- running a trip -------------------------------------------------------

    def _start_trip(self, name: str, description: str, sources: list, identify_speakers: bool):
        ok, message = check_ffmpeg_health()
        if not ok:
            QMessageBox.critical(
                self, "FFmpeg problem",
                f"{message}\n\nRecordings cannot be read without it.",
            )
            return

        try:
            trip = TripProject.create(name, sources, description=description)
        except ValueError as e:
            QMessageBox.warning(self, "Cannot start", str(e))
            return

        self._logger.info(f"Starting trip '{name}' with {len(sources)} source(s)")
        self.run_view.begin(name, len(sources))
        self.stack.setCurrentWidget(self.run_view)

        self.worker = TripWorker(
            trip,
            identify_speakers=identify_speakers,
            hf_token=get_hf_token() if identify_speakers else "",
        )
        self.worker.progress.connect(self.run_view.on_progress)
        self.worker.status.connect(self.run_view.append_status)
        self.worker.segment.connect(self.run_view.append_segment)
        self.worker.audio_ready.connect(self.run_view.set_audio)
        self.worker.completed.connect(self._on_trip_finished)
        self.worker.failed.connect(self._on_trip_failed)
        self.worker.start()
        self._start_stall_watch()

    def _cancel_trip(self):
        self._stop_stall_watch()
        if self.worker is not None:
            self.worker.cancel()
            self.worker.wait(500)
        self.run_view.on_failed("Cancelled.")

    def _on_trip_finished(self, outputs):
        self._stop_stall_watch()
        self._logger.info(f"Trip finished: {outputs.work_dir}")
        self.run_view.on_finished(outputs)

    def _on_trip_failed(self, message: str):
        self._stop_stall_watch()
        from src.utils.error_handler import get_error_suggestion

        suggestion = get_error_suggestion(message)
        self.run_view.on_failed(f"{message}\n\n{suggestion}" if suggestion else message)

    # --- stall watch ----------------------------------------------------------

    def _start_stall_watch(self):
        """Warn if the pipeline goes quiet for a long time.

        The runner's own check only fires when the segment generator yields, so
        it cannot notice faster-whisper wedged inside a blocking call. This
        timer lives on the GUI event loop and therefore keeps running when the
        worker thread is stuck, which is exactly the case worth reporting.
        It only warns; nothing here can safely interrupt native inference.
        """
        self._stall_warned = False
        if self._stall_timer is None:
            self._stall_timer = QTimer(self)
            self._stall_timer.timeout.connect(self._check_for_stall)
        self._stall_timer.start(self.STALL_POLL_MS)

    def _stop_stall_watch(self):
        if self._stall_timer is not None:
            self._stall_timer.stop()

    def _check_for_stall(self):
        if self.worker is None or not self.worker.isRunning():
            self._stop_stall_watch()
            return

        quiet_for = self.worker.seconds_since_last_segment()
        if quiet_for < self.STALL_WARN_AFTER_S:
            self._stall_warned = False
            return

        # One warning per stall, not one every poll.
        if not self._stall_warned:
            self._stall_warned = True
            self.run_view.append_status(
                f"[Still working. No new speech for {quiet_for / 60:.0f} minutes; "
                f"long silences and dense audio both look like this.]"
            )
