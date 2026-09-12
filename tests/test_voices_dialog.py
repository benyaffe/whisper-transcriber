"""
Tests for the screen that shows what the app remembers about voices.

The feature it undoes is a good one: naming a speaker teaches the app that
voice, so the same colleague across several trips is suggested rather than
asked about again. Its failure mode is the reason this screen exists. A name
given in haste, or given to the wrong row, is then suggested confidently on
every trip afterwards, and there was no way to see that had happened.

So the properties worth defending are that the list says enough to tell a wrong
row from a right one, that forgetting asks first, and that forgetting is the
only thing it does.

Run with: python -m pytest tests/test_voices_dialog.py -v
"""

import pytest

from src.podcastnotes import speaker_library
from src.ui.podcastnotes.voices_dialog import VoicesDialog, _describe


@pytest.fixture
def library(tmp_path):
    path = str(tmp_path / "voices.db")
    with speaker_library.Library(path) as db:
        db.remember("Anya Petrov-Hale", [1.0, 0.0, 0.0], trip="Ashford")
        db.remember("Anya Petrov-Hale", [0.9, 0.1, 0.0], trip="Cleveland")
        db.remember("Marcus Ellery", [0.0, 1.0, 0.0], trip="Ashford")
    return path


@pytest.fixture
def dialog(qt_app, library):
    view = VoicesDialog(library)
    yield view
    view.close()


def _rows(view):
    return [view.list.item(i).text() for i in range(view.list.count())]


def _row_for(view, name):
    """The row for one person, found by name rather than by position.

    The list is alphabetical, so picking row 0 makes a test that silently
    changes meaning the moment somebody edits a fixture name.
    """
    return next(r for r in _rows(view) if r.startswith(name))


def _select(view, name):
    for i in range(view.list.count()):
        if view.list.item(i).text().startswith(name):
            view.list.setCurrentRow(i)
            return
    raise AssertionError(f"no row for {name}")


# --- seeing --------------------------------------------------------------------


def test_everyone_remembered_is_listed(dialog):
    assert len(_rows(dialog)) == 2


def test_a_row_says_where_the_voice_came_from(dialog):
    """The count on its own does not answer the question somebody opens this to
    ask, which is whether a given row is the wrong one."""
    anya = [r for r in _rows(dialog) if r.startswith("Anya")][0]

    assert "2 recordings" in anya
    assert "Ashford" in anya and "Cleveland" in anya


def test_one_recording_is_not_one_recordings():
    assert "1 recording," in _describe(
        {"name": "Marcus", "samples": 1, "trips": ["Ashford"], "last_heard": ""}
    )


def test_a_voice_with_no_trip_recorded_still_reads(qt_app):
    line = _describe({"name": "Marcus", "samples": 1, "trips": [], "last_heard": ""})

    assert line == "Marcus, 1 recording"


def test_nothing_remembered_says_so_rather_than_showing_an_empty_box(qt_app, tmp_path):
    """An empty list widget looks like a list that failed to load."""
    view = VoicesDialog(str(tmp_path / "empty.db"))

    assert view.empty.isHidden() is False
    assert view.list.isHidden() is True
    view.close()


def test_forgetting_needs_something_chosen_first(dialog):
    dialog.list.setCurrentItem(None)

    assert dialog.forget.isEnabled() is False


# --- forgetting ----------------------------------------------------------------


def _answers(monkeypatch, button):
    from PyQt6.QtWidgets import QMessageBox

    asked = []

    def fake(parent, title, text, buttons=None, default=None):
        asked.append(text)
        return button

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fake))
    return asked


def test_forgetting_asks_first_and_says_what_goes(dialog, monkeypatch):
    """Irreversible, and the samples are the only copy: the embeddings cannot
    be recovered from the recordings without running the whole diarization
    again."""
    from PyQt6.QtWidgets import QMessageBox

    asked = _answers(monkeypatch, QMessageBox.StandardButton.Cancel)
    _select(dialog, "Marcus Ellery")

    dialog.forget.click()

    assert asked and "Marcus Ellery" in asked[0]
    assert "cannot be undone" in asked[0]
    assert len(_rows(dialog)) == 2, "it forgot the voice anyway"


def test_confirming_forgets_that_voice_and_only_that_one(dialog, monkeypatch, library):
    from PyQt6.QtWidgets import QMessageBox

    _answers(monkeypatch, QMessageBox.StandardButton.Ok)
    _select(dialog, "Marcus Ellery")

    dialog.forget.click()

    with speaker_library.Library(library) as db:
        assert [n for n, _ in db.names()] == ["Anya Petrov-Hale"]
    assert len(_rows(dialog)) == 1


def test_the_list_updates_without_being_reopened(dialog, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    _answers(monkeypatch, QMessageBox.StandardButton.Ok)
    _select(dialog, "Marcus Ellery")

    dialog.forget.click()

    assert "Marcus" not in " ".join(_rows(dialog))


def test_the_warning_promises_that_written_documents_are_untouched(dialog, monkeypatch):
    """The screen is about what gets suggested next time. Somebody who thinks
    it might rewrite a published document will not press the button."""
    from PyQt6.QtWidgets import QMessageBox

    asked = _answers(monkeypatch, QMessageBox.StandardButton.Cancel)
    _select(dialog, "Marcus Ellery")

    dialog.forget.click()

    assert "already written keeps the name" in asked[0]


def test_a_long_row_is_elided_rather_than_scrolled(dialog):
    """A long row scrolls sideways by default, which puts a horizontal
    scrollbar under a list of names and hides the end of the one row somebody
    is trying to read. Rendered and caught."""
    from PyQt6.QtCore import Qt

    assert dialog.list.textElideMode() == Qt.TextElideMode.ElideRight
    assert (dialog.list.horizontalScrollBarPolicy()
            == Qt.ScrollBarPolicy.ScrollBarAlwaysOff)


def test_nothing_elided_away_is_actually_lost(dialog):
    """The tail is the trips, which is the part that says whether this is the
    wrong voice, so it has to be reachable."""
    item = dialog.list.item(0)

    assert item.toolTip() == item.text()
