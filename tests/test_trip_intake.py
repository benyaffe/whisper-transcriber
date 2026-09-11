"""
Tests for the trip intake screen.

Run with: python -m pytest tests/test_trip_intake.py -v
"""

import os

import pytest

from src.ui.podcastnotes.intake_view import IntakeView, RecordingList


@pytest.fixture
def audio_files(tmp_path):
    paths = []
    for name in ("one.m4a", "two.m4a", "three.m4a"):
        p = tmp_path / name
        p.write_bytes(b"pretend audio " * 100)
        paths.append(str(p))
    return paths


@pytest.fixture
def view(qt_app, monkeypatch):
    # A token exists by default, so the speaker check is not the thing failing
    # in tests that are about something else.
    monkeypatch.setattr(IntakeView, "_hf_token", staticmethod(lambda: "hf_test"))
    # _report_problem opens a modal dialog, which would block the test runner
    # forever. Record what it was told instead.
    reported = []
    monkeypatch.setattr(IntakeView, "_report_problem", lambda self, p: reported.append(p))
    v = IntakeView()
    v.reported = reported
    yield v
    v.close()


def ready(view, name="Ashford", sources=(), speakers=True):
    view.name_input.setText(name)
    for s in sources:
        view.recordings.add(s)
    view.speakers_checkbox.setChecked(speakers)
    return view


def started(view):
    """Capture what the view emits when Start is pressed."""
    calls = []
    view.start_requested.connect(lambda *a: calls.append(a))
    view._start()
    return calls


# --- the recordings list ------------------------------------------------------


def test_recordings_keep_the_order_they_were_added(qt_app, audio_files):
    listing = RecordingList()

    for p in reversed(audio_files):
        listing.add(p)

    assert listing.sources() == list(reversed(audio_files))


def test_the_same_recording_cannot_be_added_twice(qt_app, audio_files):
    listing = RecordingList()

    assert listing.add(audio_files[0]) is True
    assert listing.add(audio_files[0]) is False
    assert listing.sources() == [audio_files[0]]


def test_a_url_is_shown_in_full_and_a_file_by_name(qt_app, audio_files):
    listing = RecordingList()
    listing.add(audio_files[0])
    listing.add("https://youtube.com/watch?v=abc")

    assert listing.item(0).text() == "one.m4a"
    assert listing.item(1).text() == "https://youtube.com/watch?v=abc"
    # The full path is still what gets used.
    assert listing.sources()[0] == audio_files[0]


def test_the_list_is_not_reorderable(qt_app, audio_files):
    """Which meeting came first is worked out after transcription, from what
    was said. Offering a drag handle would imply the app needs to be told,
    and would let somebody get it wrong.
    """
    listing = RecordingList()
    for p in audio_files:
        listing.add(p)

    assert listing.dragDropMode() == listing.DragDropMode.NoDragDrop


def test_file_timestamps_are_not_used_to_infer_order(qt_app, audio_files):
    """Measured on the real Ashford recordings, where all three report a
    creation time within four seconds of each other: that is when they were
    copied off the device, not when they were recorded. Sorting by it gives a
    confident wrong order, which is worse than none.
    """
    import inspect

    from src.ui.podcastnotes import intake_view

    source = inspect.getsource(intake_view.RecordingList)

    assert "getmtime" not in source
    assert "getctime" not in source
    assert "st_birthtime" not in source
    assert "sort" not in source


# --- validation ---------------------------------------------------------------


def test_a_trip_needs_a_name(view, audio_files):
    ready(view, name="", sources=audio_files[:1])

    assert "name" in view._problem().lower()
    assert started(view) == []
    assert "name" in view.reported[-1].lower(), "the operator was not told why"


def test_a_trip_needs_a_recording(view):
    ready(view, name="Ashford")

    assert "recording" in view._problem().lower()
    assert started(view) == []
    assert "recording" in view.reported[-1].lower(), "the operator was not told why"


def test_speakers_without_a_token_blocks_and_names_the_screen_to_open(view, audio_files, monkeypatch):
    """Catch it here, not thirty minutes into a transcription.

    It says Accounts rather than Settings now: the menu had both words for two
    different screens over the same four accounts, and sending somebody to the
    wrong one of them is the whole point of naming it in this message.
    """
    monkeypatch.setattr(IntakeView, "_hf_token", staticmethod(lambda: ""))
    ready(view, sources=audio_files[:1], speakers=True)

    problem = view._problem()

    assert "Accounts" in problem
    assert "Name the speakers" in problem, "name the control the way the screen does"
    assert started(view) == []
    assert IntakeView._needs_token(problem), "should offer to open Accounts"


def test_turning_speakers_off_unblocks_a_trip_with_no_token(view, audio_files, monkeypatch):
    """A solo lecture should not need a HuggingFace account."""
    monkeypatch.setattr(IntakeView, "_hf_token", staticmethod(lambda: ""))
    ready(view, sources=audio_files[:1], speakers=False)

    assert view._problem() == ""
    assert len(started(view)) == 1


def test_a_valid_trip_has_no_problem(view, audio_files):
    ready(view, sources=audio_files)

    assert view._problem() == ""


# --- starting -----------------------------------------------------------------


def test_start_emits_everything_the_worker_needs(view, audio_files):
    ready(view, name="  Ashford hospital tour  ", sources=audio_files)
    view.description_input.setPlainText("  Ridgeline and Lakeside General, with Rosa  ")

    calls = started(view)

    assert len(calls) == 1
    name, description, sources, speakers = calls[0]
    assert name == "Ashford hospital tour"          # trimmed
    assert description == "Ridgeline and Lakeside General, with Rosa"
    assert sources == audio_files
    assert speakers is True


def test_start_passes_the_reordered_sources(view, audio_files):
    ready(view, sources=list(reversed(audio_files)))

    _, _, sources, _ = started(view)[0]

    assert sources == list(reversed(audio_files))


def test_speakers_off_is_carried_through(view, audio_files):
    ready(view, sources=audio_files[:1], speakers=False)

    assert started(view)[0][3] is False


def test_a_description_is_optional(view, audio_files):
    ready(view, sources=audio_files[:1])

    assert started(view)[0][1] == ""


# --- adding ------------------------------------------------------------------


def test_a_url_and_a_local_file_can_share_a_trip(view, audio_files):
    view.recordings.add(audio_files[0])
    view.url_input.setText("https://youtube.com/watch?v=abc")
    view._add_url()

    assert view.recordings.sources() == [audio_files[0], "https://youtube.com/watch?v=abc"]
    assert view.url_input.text() == "", "the box should clear after adding"


def test_something_that_is_not_a_url_is_refused(view, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    view.url_input.setText("just some words")
    view._add_url()

    assert view.recordings.sources() == []


def test_unusable_files_are_rejected_with_a_reason(view, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    shown = {}
    monkeypatch.setattr(QMessageBox, "warning", lambda p, t, m, *a, **k: shown.update(msg=m))

    empty = tmp_path / "empty.m4a"
    empty.write_bytes(b"")
    document = tmp_path / "notes.txt"
    document.write_text("not audio")

    view.add_sources([str(empty), str(document)])

    assert view.recordings.sources() == []
    assert "empty.m4a" in shown["msg"]
    assert "notes.txt" in shown["msg"]


def test_good_files_are_kept_when_others_are_rejected(view, audio_files, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    bad = tmp_path / "bad.txt"
    bad.write_text("nope")

    view.add_sources([audio_files[0], str(bad), audio_files[1]])

    assert view.recordings.sources() == [audio_files[0], audio_files[1]]


# --- what the screen asks for -------------------------------------------------
#
# Three pieces of copy were actively misleading. These pin the fixes, because
# wording drifts back without something holding it.


def test_the_screen_does_not_ask_for_recordings_in_order(qt_app):
    """Order is recoverable from the recordings themselves. Asking for it put
    the cost of a solvable problem onto the person.

    Reads the widgets rather than the source, because the module docstring
    quotes the old wording while explaining why it went.
    """
    from PyQt6.QtWidgets import QLabel

    from src.ui.podcastnotes.intake_view import IntakeView

    view = IntakeView()
    try:
        shown = " ".join(
            label.text() for label in view.findChildren(QLabel)
        ).lower()

        assert "order they happened" not in shown
        assert "in the order" not in shown
    finally:
        view.deleteLater()


def test_the_description_reads_as_a_seed_not_as_homework(qt_app):
    """It looked like a request for full context, which made the quality of
    the write-up seem to depend on how much you typed."""
    from src.ui.podcastnotes.intake_view import DESCRIPTION_HINT

    assert "Glean" in DESCRIPTION_HINT, "say where the rest of the context comes from"
    assert "search" in DESCRIPTION_HINT.lower()
    assert "plenty" in DESCRIPTION_HINT or "sentence" in DESCRIPTION_HINT


def test_the_speaker_option_describes_the_outcome(qt_app):
    """"Multiple speakers" named a mechanism. What somebody wants to know is
    whether the transcript will say who spoke."""
    from src.ui.podcastnotes.intake_view import SPEAKERS_HINT, IntakeView

    view = IntakeView()
    try:
        assert view.speakers_checkbox.text() == "Name the speakers"
    finally:
        view.deleteLater()

    assert "who spoke" in SPEAKERS_HINT
    assert "Speaker 1" in SPEAKERS_HINT, "show what the alternative looks like"


def test_the_link_field_says_a_link_to_what(qt_app):
    """"Add URL..." did not say a URL to what."""
    from src.ui.podcastnotes.intake_view import IntakeView

    view = IntakeView()
    try:
        placeholder = view.url_input.placeholderText().lower()

        assert "recording" in placeholder
        assert any(site in placeholder for site in ("youtube", "vimeo", "drive"))
    finally:
        view.deleteLater()


# --- the screen gets out of its own way ---------------------------------------


def test_the_drop_zone_shrinks_once_there_are_recordings(qt_app, tmp_path):
    """A full-height target is right for an empty screen and wrong once it
    has been used, where it only pushes the content down."""
    from src.ui.podcastnotes.intake_view import IntakeView

    view = IntakeView()
    try:
        tall = view.drop_zone.minimumHeight()
        view.recordings.add(str(tmp_path / "a.m4a"))
        view._refresh()

        assert view.drop_zone.minimumHeight() < tall
        assert view.drop_zone.sublabel.isHidden()
    finally:
        view.deleteLater()


def test_the_list_and_remove_button_are_hidden_when_empty(qt_app):
    from src.ui.podcastnotes.intake_view import IntakeView

    view = IntakeView()
    view.show()
    try:
        assert view.recordings.isHidden()
        assert view.remove_button.isHidden()
    finally:
        view.close()
        view.deleteLater()


def test_the_list_grows_with_its_contents_up_to_a_limit(qt_app, tmp_path):
    """Two recordings should not occupy the space of ten, and thirty should
    not push Start off the bottom of the screen."""
    from src.ui.podcastnotes.intake_view import RecordingList

    listing = RecordingList()
    try:
        listing.add(str(tmp_path / "one.m4a"))
        one = listing.height()
        listing.add(str(tmp_path / "two.m4a"))
        two = listing.height()

        assert two > one

        for i in range(30):
            listing.add(str(tmp_path / f"more{i}.m4a"))

        # The cap is measured now rather than a constant, because a hardcoded
        # row height went out of step with the theme and clipped every size.
        assert listing.height() >= (
            RecordingList.MAX_VISIBLE_ROWS * listing.sizeHintForRow(0)
        )
        capped = listing.height()
        listing.add(str(tmp_path / "one-more.m4a"))
        assert listing.height() == capped
    finally:
        listing.deleteLater()


# --- typing must not touch the disk or the Keychain ------------------------------


def test_typing_a_name_spawns_no_subprocess(view, audio_files, monkeypatch):
    """The lag. Every keystroke ran the validation, which probed each recording
    with ffprobe, and then ran the whole scan a second time because the hint was
    `self._problem() or self._advisory()` and both call it. Two subprocess
    spawns per file per character, on the GUI thread."""
    import subprocess

    ready(view, sources=audio_files)
    view._hint_text()                       # warm the caches, as real use does

    spawned = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: spawned.append(a))

    for character in "Ashford hospital tour":
        view.name_input.setText(view.name_input.text() + character)

    assert spawned == [], f"{len(spawned)} subprocesses while typing"


def test_typing_a_name_reads_the_keychain_once_at_most(view, audio_files, monkeypatch):
    """`get_hf_token` is a macOS Keychain round trip, and it sat between every
    character typed and that character appearing."""
    from src.ui.podcastnotes.intake_view import IntakeView

    reads = []
    monkeypatch.setattr(IntakeView, "_hf_token",
                        lambda self: (reads.append(1), "hf_test")[1] if not reads else "hf_test")
    ready(view, sources=audio_files)

    for character in "Ashford":
        view.name_input.setText(view.name_input.text() + character)

    assert len(reads) <= 1


def test_the_duration_scan_runs_once_per_refresh(view, audio_files, monkeypatch):
    """It used to run twice on a valid form: the problem check scanned, found
    nothing to complain about, and the advisory scanned again for the same
    answer."""
    calls = []
    real = view._duration
    monkeypatch.setattr(view, "_duration", lambda: (calls.append(1), real())[1])
    ready(view, sources=audio_files)

    calls.clear()
    view._hint_text()

    assert len(calls) == 1


def test_the_advice_still_appears_on_a_valid_form(view, audio_files, monkeypatch):
    """Collapsing two passes into one must not lose the advisory, which is the
    only thing that says a long trip will take a while."""
    from src.ui.podcastnotes import intake_view as module

    monkeypatch.setattr(module, "check_duration", lambda t, e: (True, "This will take a while."))
    ready(view, sources=audio_files)

    assert view._hint_text() == "This will take a while."


def test_a_blocking_problem_still_wins_over_advice(view, audio_files, monkeypatch):
    from src.ui.podcastnotes import intake_view as module

    monkeypatch.setattr(module, "check_duration", lambda t, e: (False, "Too long."))
    ready(view, sources=audio_files)

    assert view._hint_text() == "Too long."


def test_a_changed_token_can_be_picked_up(view, audio_files, monkeypatch):
    """The cache has to be droppable, or changing the token in Accounts would
    not take effect until the app restarted."""
    view._token_cache = "stale"

    view.forget_hf_token()

    assert view._token_cache is None


# --- the list has to show the rows it claims to --------------------------------


def _filled(qt_app, count):
    listing = RecordingList()
    for i in range(count):
        listing.add(f"/tmp/recording-number-{i}.m4a")
    return listing


@pytest.mark.parametrize("count", [1, 2, 3, 5, 7])
def test_every_row_fits_without_a_scrollbar(qt_app, count):
    """The height was `rows * 30 + 12` against a real row of 32, so it
    under-shot at every size and a scrollbar appeared from the first item. Two
    recordings looked like a cramped scrolling box, which reads as the list not
    growing at all."""
    listing = _filled(qt_app, count)

    needed = listing.sizeHintForRow(0) * count

    assert listing.height() >= needed, (
        f"{count} rows need {needed}px, the list is {listing.height()}px"
    )


def test_the_list_stops_growing_at_the_cap(qt_app):
    """A trip with thirty files must not push Start off the bottom."""
    at_cap = _filled(qt_app, RecordingList.MAX_VISIBLE_ROWS).height()
    over_cap = _filled(qt_app, RecordingList.MAX_VISIBLE_ROWS + 6).height()

    assert over_cap == at_cap


def test_the_height_is_measured_rather_than_assumed(qt_app):
    """A constant here goes out of step the moment the theme's item padding or
    the body font size changes."""
    listing = _filled(qt_app, 3)

    assert listing.height() >= 3 * listing.sizeHintForRow(0)


def test_an_empty_list_still_has_a_sensible_height(qt_app):
    """There is no row to measure, so this is the one case with a constant."""
    listing = RecordingList()

    assert listing.height() >= RecordingList.FALLBACK_ROW_HEIGHT


def test_removing_a_recording_shrinks_the_list(qt_app):
    listing = _filled(qt_app, 4)
    tall = listing.height()

    listing.setCurrentRow(0)
    listing.remove_selected()

    assert listing.height() < tall
