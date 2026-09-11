"""
Tests for the write-up wiring in the main window.

What matters here is the sequence and what happens when a step fails. The
steps themselves are tested elsewhere; this is about the window driving them
in the right order, carrying the person's own description through, and not
throwing away minutes of work when something transient goes wrong.

The worker is replaced throughout, so nothing here starts a thread or reaches
the network.

Run with: python -m pytest tests/test_main_window_writeup.py -v
"""

import json

import pytest

from src.ui.main_window import MainWindow, _read_json
from src.ui.podcastnotes.writeup_view import (
    PANE_DONE, PANE_QUESTIONS, PANE_SPEAKERS,
)


class _FakeWorker:
    """Stands in for WriteUpWorker, running the step immediately."""

    started = []

    def __init__(self, writeup, step, answers=None, library_path="", notes="",
                 parent=None):
        self.writeup = writeup
        self.step = step
        self.answers = answers
        self.notes = notes
        _FakeWorker.notes.append(notes)
        self.label = f"doing {step}"
        self.phase = step.title()
        self._on_done = None
        self._on_fail = None
        self.progress = _Signal()
        self.found = _Signal()
        self.failed = _Signal()
        self.completed = _Signal()
        self.typical_seconds = 60

    notes = []

    def isRunning(self):
        return False

    def elapsed_fraction(self):
        return 0.5

    def start(self):
        _FakeWorker.started.append(self.step)
        # A step with no scripted result records that it started and stops
        # there, so a test can assert on one step without the whole cascade
        # running on behind it.
        if self.step not in _FakeWorker.results:
            return
        result = _FakeWorker.results[self.step]
        if isinstance(result, Exception):
            self.failed.emit(str(result))
        else:
            self.completed.emit(result)


class _Signal:
    def __init__(self):
        self._slots = []

    def connect(self, slot):
        self._slots.append(slot)

    def emit(self, *args):
        for slot in list(self._slots):
            slot(*args)


class _FakeWriteUp:
    def __init__(self):
        from src.podcastnotes.context import ContextMap

        self.context_map = ContextMap(likely_errors=[
            {"heard": "Ridgelane", "probably": "Ridgeline", "confidence": "high"},
        ])
        self.description = ""
        self.work_dir = "/tmp/trip"
        self.notes = ["Corrected: a -> b (1x)"]
        self.summary = "SUMMARY"
        self.transcript = "TRANSCRIPT"
        self.confirmed = None
        self.corrections_applied = False
        self.revisions_left = 5
        self.impossible = []
        self.revised = []

    def revise(self, notes, on_search=None):
        self.revised.append(notes)

    def apply_corrections(self):
        self.corrections_applied = True

    def voices(self):
        return []

    def confirm_speakers(self, names):
        self.confirmed = names


@pytest.fixture
def window(qt_app, monkeypatch, tmp_path):
    _FakeWorker.started = []
    _FakeWorker.results = {}
    _FakeWorker.notes = []
    monkeypatch.setattr("src.ui.main_window.WriteUpWorker", _FakeWorker)

    win = MainWindow()
    win.writeup = _FakeWriteUp()
    yield win
    win.close()


class _Outputs:
    def __init__(self, tmp_path, json_path=None, boundaries_path=""):
        self.work_dir = str(tmp_path)
        self.audio_path = str(tmp_path / "combined.m4a")
        self.json_path = json_path if json_path is not None else str(tmp_path / "c.json")
        self.boundaries_path = boundaries_path


# --- the sequence -------------------------------------------------------------


def test_the_context_step_runs_first(window):
    window._run_step("context")

    assert _FakeWorker.started == ["context"]


def test_the_corrections_are_applied_without_being_reviewed(window):
    """The gate asked for a judgement before there was anything to judge
    against. The same work now happens on the finished screen, where the
    person can see which words are actually wrong."""
    _FakeWorker.results = {"context": "a map", "speakers": []}

    window._run_step("context")

    assert window.writeup.corrections_applied is True
    assert _FakeWorker.started == ["context", "speakers"]


def test_the_speaker_step_stops_and_asks(window):
    """The one place the write-up waits for a person."""
    _FakeWorker.results = {"speakers": []}

    window._run_step("speakers")

    assert window.stack.currentWidget() is window.writeup_view
    assert window.writeup_view.panes.currentIndex() == PANE_SPEAKERS


def test_confirming_speakers_records_them_and_moves_on(window):
    _FakeWorker.results = {"questions": None}

    window._speakers_confirmed({"Speaker 1": "Marcus Ellery"})

    assert window.writeup.confirmed == {"Speaker 1": "Marcus Ellery"}
    assert "questions" in _FakeWorker.started


def test_an_unanswered_round_moves_on_rather_than_stalling(window):
    """Skipping is always allowed by design, so with no question screen yet the
    write-up carries on instead of stopping in a place with no way out."""
    from src.podcastnotes.output import Documents

    _FakeWorker.results = {"questions": None, "write": Documents()}

    window._run_step("questions")

    assert _FakeWorker.started == ["questions", "write"]


def test_the_finished_documents_are_shown(window):
    from src.podcastnotes.output import Documents

    _FakeWorker.results = {"write": Documents(transcript="t", summary="s")}

    window._run_step("write")

    assert window.writeup_view.panes.currentIndex() == PANE_DONE
    assert "s" in window.writeup_view.document.toPlainText(), "the summary shows first"


# --- failure and retry ---------------------------------------------------------


def test_a_failing_step_is_shown_on_the_write_up_screen(window):
    _FakeWorker.results = {"context": RuntimeError("Your sign-in has expired.")}

    window._run_step("context")

    assert "expired" in window.writeup_view.problem.text()


def test_retrying_repeats_the_step_that_failed_not_the_whole_run(window):
    """These failures are transient, and starting over would discard a context
    map that took minutes to build."""
    _FakeWorker.results = {"speakers": RuntimeError("no route to host")}
    window._run_step("speakers")
    _FakeWorker.started = []
    _FakeWorker.results = {"speakers": []}

    window._retry_writeup()

    assert _FakeWorker.started == ["speakers"]


# --- starting from a finished trip ---------------------------------------------


def test_the_description_the_person_typed_reaches_the_write_up(window, tmp_path):
    """Without it the context stage is searching for a trip it knows nothing
    about."""
    (tmp_path / "c.json").write_text(json.dumps({"segments": [{"text": "hello"}]}))
    window.trip = type("T", (), {"description": "Ashford ED site visits"})()

    window._start_writeup(_Outputs(tmp_path))

    assert window.writeup.description == "Ashford ED site visits"


def test_a_trip_with_no_segment_export_does_not_start_a_write_up(window, tmp_path):
    """Cancelled part way through transcription, so there is nothing to write
    up and pretending otherwise would fail several steps later."""
    window._start_writeup(_Outputs(tmp_path, json_path=""))

    assert _FakeWorker.started == []


def test_the_audio_is_handed_over_so_the_clips_can_play(window, tmp_path):
    (tmp_path / "c.json").write_text(json.dumps({"segments": []}))
    window.trip = type("T", (), {"description": ""})()

    window._start_writeup(_Outputs(tmp_path))

    assert window.writeup_view._audio_path.endswith("combined.m4a")


# --- reading the trip's files ---------------------------------------------------


def test_a_missing_file_reads_as_empty_rather_than_raising(tmp_path):
    """boundaries.json only exists for a trip of more than one recording."""
    assert _read_json("") == {}
    assert _read_json(str(tmp_path / "nope.json")) == {}


def test_a_corrupt_file_reads_as_empty(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json")

    assert _read_json(str(bad)) == {}


def test_a_real_file_is_read(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"sources": [1, 2]}))

    assert _read_json(str(good)) == {"sources": [1, 2]}


# --- question rounds ------------------------------------------------------------


class _Round:
    def __init__(self, questions, remaining=0):
        self.questions = questions
        self.remaining = remaining


class _Q:
    def __init__(self, marker):
        self.marker = marker
        self.ask = "What did they mean?"
        self.why_it_matters = ""
        self.occurrences = 1


def test_a_round_with_questions_stops_and_asks(window):
    _FakeWorker.results = {"questions": _Round([_Q("Kestler")])}

    window._run_step("questions")

    assert window.writeup_view.panes.currentIndex() == PANE_QUESTIONS
    assert "write" not in _FakeWorker.started


def test_an_empty_round_is_the_end_rather_than_a_skip(window):
    """The pipeline returning no questions means there is nothing left worth
    asking, so the write-up finishes rather than waiting for somebody."""
    from src.podcastnotes.output import Documents

    _FakeWorker.results = {"questions": _Round([]), "write": Documents()}

    window._run_step("questions")

    assert _FakeWorker.started == ["questions", "write"]


def test_answers_are_folded_in_and_another_round_is_sought(window):
    _FakeWorker.results = {"answers": None, "questions": _Round([])}

    window._answers_given({"Kestler": "It is Kessler."})

    assert _FakeWorker.started[0] == "answers"
    assert "questions" in _FakeWorker.started


def test_skipping_still_advances_the_round(window):
    """Otherwise skipping would offer the same questions forever."""
    _FakeWorker.results = {"answers": None, "questions": _Round([])}

    window._answers_given({})

    assert _FakeWorker.started[0] == "answers"


# --- the two menu entries used to mean the same thing ----------------------------


def test_the_menu_does_not_offer_two_words_for_the_same_idea(window):
    """"Settings..." and "Setup..." sat next to each other meaning different
    screens over the same four accounts, one to type them in and one to check
    they work. Somebody told to open Setup opened Settings instead."""
    menu = window.menuBar().actions()[0].menu()
    labels = [a.text() for a in menu.actions()]

    assert "Setup..." in labels
    assert "Settings..." not in labels
    assert "Accounts..." in labels


def test_setup_comes_first_because_it_is_the_one_to_open_when_stuck(window):
    menu = window.menuBar().actions()[0].menu()
    labels = [a.text() for a in menu.actions()]

    assert labels.index("Setup...") < labels.index("Accounts...")


def test_each_menu_entry_matches_the_window_it_opens(qt_app):
    """The mismatch that started this: the label clicked and the heading that
    appeared were different words."""
    from src.ui.podcastnotes.setup_view import SetupView
    from src.ui.settings_dialog import SettingsDialog

    view = SetupView()
    dialog = SettingsDialog()
    try:
        assert view.heading.text() == "Setup"
        assert dialog.windowTitle() == "Accounts"
    finally:
        view.close()
        dialog.close()


# --- publishing ------------------------------------------------------------------


def test_publishing_reports_back_to_the_screen(window, monkeypatch, tmp_path):
    """The result used to go to the log and nowhere else."""
    from src.podcastnotes import publish

    class _Result:
        copied = True
        browser_opened = True
        problems = ()
        markdown_path = str(tmp_path / "write-up.md")

        def summary(self):
            return "Saved, copied, and a blank document is open."

    window.writeup.work_dir = str(tmp_path)
    monkeypatch.setattr(publish, "publish", lambda text, target: _Result())

    window._publish()

    assert window.writeup_view.published.isHidden() is False


def test_the_two_documents_are_joined_the_way_output_defines_it(window):
    """The separator between them is a decision, and a second copy of it in the
    window is how the two drift apart."""
    from src.podcastnotes.output import Documents

    window.writeup.summary = "SUMMARY"
    window.writeup.transcript = "TRANSCRIPT"

    assert window._combined_document() == Documents(
        summary="SUMMARY", transcript="TRANSCRIPT"
    ).combined


# --- leaving a finished write-up -------------------------------------------------


def test_leaving_unsaved_documents_asks_first(window, monkeypatch):
    """They took about half an hour to make and this drops them."""
    from PyQt6.QtWidgets import QMessageBox

    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: (asked.append(1), QMessageBox.StandardButton.Cancel)[1])
    window.writeup.summary = "SUMMARY"
    window._kept = False
    window.stack.setCurrentWidget(window.writeup_view)

    window._start_another_trip()

    assert asked == [1]
    assert window.stack.currentWidget() is window.writeup_view, "left anyway"


def test_discarding_when_asked_goes_ahead(window, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Discard)
    window.writeup.summary = "SUMMARY"
    window._kept = False

    window._start_another_trip()

    assert window.stack.currentWidget() is window.intake


def test_documents_already_copied_do_not_stop_anybody(window, monkeypatch):
    """Somebody who has put them in a Doc has no reason to be interrupted."""
    from PyQt6.QtWidgets import QMessageBox

    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(1))
    window.writeup.summary = "SUMMARY"
    window._kept = True

    window._start_another_trip()

    assert asked == []
    assert window.stack.currentWidget() is window.intake


def test_publishing_counts_as_keeping_them(window, monkeypatch, tmp_path):
    from src.podcastnotes import publish

    class _Result:
        copied = True
        browser_opened = True
        problems = ()
        markdown_path = str(tmp_path / "write-up.md")

        def summary(self):
            return "Saved and copied."

    window.writeup.work_dir = str(tmp_path)
    monkeypatch.setattr(publish, "publish", lambda text, target: _Result())

    window._publish()

    assert window._kept is True


def test_a_failed_copy_does_not_count_as_keeping_them(window, monkeypatch, tmp_path):
    from src.podcastnotes import publish

    class _Result:
        copied = False
        browser_opened = True
        problems = ("clipboard failed",)
        markdown_path = str(tmp_path / "write-up.md")

        def summary(self):
            return "Saved."

    window.writeup.work_dir = str(tmp_path)
    monkeypatch.setattr(publish, "publish", lambda text, target: _Result())

    window._publish()

    assert window._kept is False


def test_saving_somewhere_chosen_counts(window, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QFileDialog

    target = tmp_path / "chosen.md"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    window.writeup.summary = "S"
    window.writeup.transcript = "T"

    window._save_write_up()

    assert window._kept is True
    assert "S" in target.read_text()


def test_cancelling_the_save_dialog_writes_nothing(window, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))

    window._save_write_up()

    assert window._kept is False


# --- correcting the finished pair ------------------------------------------------


def _revise(window, notes="the coordinator is Simone Vasari"):
    window.writeup_view.revision_requested.emit(notes)


def test_the_note_reaches_the_worker(window):
    _FakeWorker.results = {}

    _revise(window)

    assert _FakeWorker.started == ["revise"]
    assert _FakeWorker.notes[-1] == "the coordinator is Simone Vasari"


def test_retrying_a_failed_rewrite_sends_the_same_note_again(window):
    """Otherwise the retry spends a full pass on an empty note, which returns
    nothing and looks like the retry silently failing."""
    _FakeWorker.results = {"revise": ConnectionError("Glean is unavailable")}
    _revise(window, "Speaker 3 is Miles Nadeau")
    _FakeWorker.started = []
    _FakeWorker.results = {}

    window._retry_writeup()

    assert _FakeWorker.started == ["revise"]
    assert _FakeWorker.notes[-1] == "Speaker 3 is Miles Nadeau"


def test_the_rewritten_pair_replaces_the_old_one(window):
    from src.podcastnotes.output import Documents

    found = object()
    documents = Documents(transcript="NEW T", summary="NEW S")
    _FakeWorker.results = {"revise": (found, documents)}

    _revise(window)

    assert window.writeup_view._documents is documents


def test_what_could_not_be_done_is_carried_to_the_screen(window):
    """Shown even when the rewrite otherwise succeeded, because "I did four of
    the five things you asked" is the honest report."""
    from src.podcastnotes.output import Documents

    window.writeup.impossible = ["Nobody mentions a Dr Okafor."]
    _FakeWorker.results = {"revise": (object(), Documents(transcript="t", summary="s"))}

    _revise(window)

    assert "Dr Okafor" in window.writeup_view.impossible.text()


def test_the_first_writing_clears_a_complaint_from_a_previous_trip(window):
    from src.podcastnotes.output import Documents

    window.writeup_view.show_impossible(["stale"])
    _FakeWorker.results = {"write": Documents(transcript="t", summary="s")}

    window._run_step("write")

    assert window.writeup_view.impossible.isHidden() is True


def test_how_many_rewrites_are_left_reaches_the_screen(window):
    from src.podcastnotes.output import Documents

    window.writeup.revisions_left = 1
    _FakeWorker.results = {"write": Documents(transcript="t", summary="s")}

    window._run_step("write")

    assert "One more rewrite" in window.writeup_view.revisions_left.text()


def _asks(monkeypatch, answer):
    """Stand in for the confirmation, and record whether it was reached."""
    from PyQt6.QtWidgets import QMessageBox

    shown = []

    def fake(parent, title, text, buttons=None, default=None):
        shown.append(text)
        return answer

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fake))
    return shown


def test_the_last_rewrite_is_confirmed_before_it_starts(window, monkeypatch):
    """Ten minutes and the end of the road. Worth being sure about, unlike the
    first four, which are cheap to be wrong about."""
    from PyQt6.QtWidgets import QMessageBox

    window.writeup.revisions_left = 1
    shown = _asks(monkeypatch, QMessageBox.StandardButton.Ok)
    _FakeWorker.results = {}

    _revise(window)

    assert "last rewrite" in shown[0]
    assert _FakeWorker.started == ["revise"]


def test_cancelling_the_last_rewrite_spends_nothing(window, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    window.writeup.revisions_left = 1
    _asks(monkeypatch, QMessageBox.StandardButton.Cancel)

    _revise(window)

    assert _FakeWorker.started == []


def test_the_earlier_rewrites_are_not_worth_interrupting_for(window, monkeypatch):
    """Four confirmations for four cheap decisions is the dialog nobody reads."""
    from PyQt6.QtWidgets import QMessageBox

    window.writeup.revisions_left = 3
    shown = _asks(monkeypatch, QMessageBox.StandardButton.Cancel)
    _FakeWorker.results = {}

    _revise(window)

    assert shown == []
    assert _FakeWorker.started == ["revise"]
