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
from src.ui.podcastnotes.writeup_view import WriteUpView
from src.ui.podcastnotes.writeup_worker import WriteUpWorker
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
        self.trip = None
        self.writeup = None
        self.writeup_worker = None
        self._last_step = "context"
        self._estimate_timer = None
        # Whether this write-up has been put somewhere the person chose.
        self._kept = False
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

        self.writeup_view = WriteUpView()
        self.writeup_view.context_approved.connect(self._context_approved)
        self.writeup_view.speakers_confirmed.connect(self._speakers_confirmed)
        self.writeup_view.answers_given.connect(self._answers_given)
        self.writeup_view.publish_requested.connect(self._publish)
        self.writeup_view.save_requested.connect(self._save_write_up)
        self.writeup_view.reveal_requested.connect(self._reveal_write_up)
        self.writeup_view.new_trip_requested.connect(self._start_another_trip)
        self.writeup_view.retry_requested.connect(self._retry_writeup)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.setup_view)
        self.stack.addWidget(self.intake)
        self.stack.addWidget(self.run_view)
        self.stack.addWidget(self.writeup_view)
        # Stated rather than left to insertion order. Whether the checklist
        # takes the screen is check_setup_on_launch's decision and nothing
        # else's, and a stack that quietly defaults to it would make that
        # policy depend on which line addWidget was called on.
        self.stack.setCurrentWidget(self.intake)
        self.setCentralWidget(self.stack)

        self._setup_menu()

    def _setup_menu(self):
        """Two entries, and they used to be near-synonyms sitting together.

        "Settings..." and "Setup..." meant different screens covering the same
        four accounts, one to type them in and one to check they work, and
        somebody told to go to Setup opened Settings instead. The dialog is now
        called Accounts, which is what it holds, and the checklist keeps Setup.
        Setup comes first because it is the one to open when something is wrong.
        """
        app_menu = self.menuBar().addMenu("PodcastNotesWT")
        setup_action = app_menu.addAction("Setup...")
        setup_action.triggered.connect(self.show_setup)
        settings_action = app_menu.addAction("Accounts...")
        settings_action.setShortcut("Cmd+,")
        settings_action.triggered.connect(self._open_settings)

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
        """Open Accounts, and let the intake screen know its cache is stale.

        The intake screen reads the HuggingFace token once rather than on every
        keystroke, so the one place that token can change has to say so.
        """
        from src.ui.settings_dialog import SettingsDialog

        SettingsDialog(self).exec()
        self.intake.forget_hf_token()

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
        self.intake.clear()
        self.stack.setCurrentWidget(self.intake)

    def _start_another_trip(self):
        """Leave the finished write-up, once it is clear that is intended.

        The documents took about half an hour to make and are dropped by this,
        so it asks first, and only when they have not been published or saved.
        Somebody who has already put them in a Doc has no reason to be stopped.
        """
        from PyQt6.QtWidgets import QMessageBox

        if self.writeup is not None and self.writeup.summary and not self._kept:
            answer = QMessageBox.question(
                self, "Leave this write-up?",
                "The transcript and summary for this trip have not been copied "
                "or saved anywhere you chose. Starting another trip leaves them "
                "behind.\n\nThe files stay in the trip folder either way.",
                QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Discard,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Discard:
                return
        self._show_intake()

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

        # Kept, because the write-up needs the description the person typed and
        # it is the only place that text survives after this method returns.
        self.trip = trip
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
        self._start_writeup(outputs)

    # --- writing it up --------------------------------------------------------

    def _start_writeup(self, outputs):
        """Carry straight on into the write-up rather than stopping at the files.

        No button in between, because the transcript on its own is not what
        anybody came for and asking somebody to discover a second phase is how
        the twenty-minute promise turns into a two-hour one. A write-up that
        cannot start says so on its own screen, and the transcript is already
        saved either way.
        """
        payload = _read_json(getattr(outputs, "json_path", ""))
        if not payload:
            self._logger.warning("No segment export, so there is nothing to write up.")
            return

        from src.podcastnotes.pipeline import WriteUp

        self._kept = False
        self.writeup = WriteUp.resume(
            outputs.work_dir, payload,
            boundaries=_read_json(getattr(outputs, "boundaries_path", "")),
        )
        self.writeup.description = getattr(self.trip, "description", "") or ""
        self.writeup_view.set_audio(getattr(outputs, "audio_path", ""))
        self.stack.setCurrentWidget(self.writeup_view)
        self._run_step("context")

    def _run_step(self, step: str, answers: dict = None):
        self._last_step = step
        self.writeup_worker = WriteUpWorker(self.writeup, step, answers=answers)
        # Here rather than only at the start, so that retrying after a failure
        # cannot leave the progress and the error on a screen nobody is looking
        # at.
        self.stack.setCurrentWidget(self.writeup_view)
        self.writeup_view.working(
            self.writeup_worker.label, phase=self.writeup_worker.phase
        )
        self.writeup_worker.progress.connect(self.writeup_view.note)
        self.writeup_worker.found.connect(self.writeup_view.note_finding)
        self.writeup_worker.failed.connect(self.writeup_view.on_failed)
        self.writeup_worker.completed.connect(
            lambda result, done=step: self._step_finished(done, result)
        )
        self.writeup_worker.start()
        self._start_estimate()

    def _start_estimate(self):
        """Tick the estimated progress bar while a long step runs.

        On a timer rather than from the worker, because the worker is busy
        inside a network call for most of the time it is running and cannot
        report anything while it is.
        """
        from PyQt6.QtCore import QTimer

        if self._estimate_timer is None:
            self._estimate_timer = QTimer(self)
            self._estimate_timer.timeout.connect(self._tick_estimate)
        self._estimate_timer.start(1000)

    def _tick_estimate(self):
        worker = self.writeup_worker
        if worker is None or not worker.isRunning():
            self._estimate_timer.stop()
            return
        fraction = worker.elapsed_fraction()
        self.writeup_view.set_estimate(
            fraction, worker.typical_seconds * (1.0 - fraction)
        )

    def _retry_writeup(self):
        """Whatever failed, try that same step again rather than starting over.

        Every one of the failures this screen reports is transient: a sign-in
        that expired, a search that could not be reached. Restarting the whole
        write-up would discard a context map that took minutes to build.
        """
        self._run_step(getattr(self, "_last_step", "context"))

    def _step_finished(self, step: str, result):
        if step == "context":
            # Shown before anything is applied. Everything downstream treats
            # this map as fact, so it is the last point where a wrong entry is
            # cheap to remove rather than something to spot in a finished
            # document.
            self.writeup_view.review_context(self.writeup.context_map)
        elif step == "speakers":
            self.writeup_view.ask_speakers(
                self.writeup.voices(), suggestions=result or []
            )
        elif step == "answers":
            self._run_step("questions")
        elif step == "questions":
            # An empty round is the pipeline saying there is nothing left worth
            # asking, which is a normal ending rather than a skip.
            if result is None or not result.questions:
                self._run_step("write")
            else:
                self.writeup_view.ask_questions(result)
        elif step == "write":
            self.writeup_view.show_documents(result, notes=self.writeup.notes)

    def _context_approved(self, rejected: list):
        self.writeup.reject_corrections(rejected)
        self.writeup.apply_corrections()
        self._run_step("speakers")

    def _speakers_confirmed(self, names: dict):
        self.writeup.confirm_speakers(names)
        self._run_step("questions")

    def _answers_given(self, answers: dict):
        """Fold the answers in, then look for another round.

        Skipping sends an empty dict rather than a different signal, because
        skipping is a real answer to "can you settle any of these" and the
        pipeline already treats it as one: the round still advances and the
        markers simply stay.
        """
        self._run_step("answers", answers=answers)

    def _publish(self):
        """Save, copy, open a blank document, and say so.

        The result used to go to the log and nowhere else, so a clipboard
        failure left somebody pasting whatever they had copied earlier into a
        blank Google Doc with no idea why.
        """
        import os

        from src.podcastnotes import publish

        target = os.path.join(self.writeup.work_dir, "write-up.md")
        result = publish.publish(self._combined_document(), target)
        self._kept = self._kept or result.copied
        self.writeup_view.show_published(result)
        self._logger.info(f"Published: {result.summary()} {result.problems}")

    def _save_write_up(self):
        """Put the Markdown somewhere the person chose.

        It is always written to the trip folder, but that path is never shown,
        so "it is saved" was true and useless.
        """
        from PyQt6.QtWidgets import QFileDialog

        suggested = f"{getattr(self.trip, 'name', 'write-up') or 'write-up'}.md"
        target, _ = QFileDialog.getSaveFileName(
            self, "Save the write-up", suggested, "Markdown (*.md)"
        )
        if not target:
            return
        try:
            with open(target, "w") as handle:
                handle.write(self._combined_document())
        except OSError as e:
            self.writeup_view.note_saved(f"Could not save it there: {e}")
            return
        self._kept = True
        self.writeup_view.note_saved(f"Saved to {target}")

    def _reveal_write_up(self):
        import subprocess

        subprocess.run(["open", self.writeup.work_dir], check=False)

    def _combined_document(self) -> str:
        """Both documents as one, the way output.Documents defines it.

        Formatted in one place rather than two, because the separator between
        them is a decision and having a second copy of it is how the two drift.
        """
        from src.podcastnotes.output import Documents

        return Documents(
            transcript=self.writeup.transcript, summary=self.writeup.summary
        ).combined

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


def _read_json(path: str) -> dict:
    """One of the trip's JSON artefacts, or an empty dict if it is not there.

    Missing is normal rather than exceptional: boundaries.json only exists for
    a trip of more than one recording, and the segment export is absent if
    transcription was cancelled part way.
    """
    import json

    if not path:
        return {}
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
