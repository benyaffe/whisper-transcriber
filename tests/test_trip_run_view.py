"""
Tests for the screen a trip runs on.

Run with: python -m pytest tests/test_trip_run_view.py -v
"""

import pytest

from src.ui.podcastnotes.run_view import RunView
from src.ui.podcastnotes.trip_worker import TripOutputs


@pytest.fixture
def view(qt_app):
    v = RunView()
    yield v
    v.close()


def outputs(tmp_path, **kwargs):
    return TripOutputs(
        work_dir=kwargs.pop("work_dir", str(tmp_path / "ashford")),
        audio_path=str(tmp_path / "combined.m4a"),
        vtt_path=str(tmp_path / "combined.vtt"),
        txt_path=str(tmp_path / "combined.txt"),
        json_path=str(tmp_path / "combined.json"),
        **kwargs,
    )


# --- running ------------------------------------------------------------------


def test_begin_names_the_trip_and_counts_the_recordings(view):
    view.begin("Ashford hospital tour", 3)

    assert view.title.text() == "Ashford hospital tour"
    assert "3 recordings" in view.transcript.toPlainText()


def test_a_single_recording_is_not_called_recordings(view):
    view.begin("Lecture", 1)

    assert "1 recording" in view.transcript.toPlainText()
    assert "recordings" not in view.transcript.toPlainText()


def test_progress_drives_the_bar_and_the_stage_label(view):
    view.begin("Ashford", 1)

    view.on_progress(42.7, "Analyzing voices 12/40")

    assert view.progress_bar.value() == 42
    assert view.stage_label.text() == "Analyzing voices 12/40"


def test_the_stage_label_is_wide_enough_for_what_goes_in_it(view):
    """A clipped label defeats the point of having one.

    Measures the real diarization stage names rather than a copy of them, so
    renaming a stage to something longer fails here instead of silently
    clipping in the one place the operator is looking.
    """
    from PyQt6.QtGui import QFontMetrics

    from src.core.diarization import (
        DIARIZATION_POST_STEP_LABELS,
        DIARIZATION_STAGE_LABELS,
    )

    candidates = [
        "Transcribing, about 12m 34s left",
        "Joining 3 recordings",
        "Loading speaker model",
    ]
    for label in list(DIARIZATION_STAGE_LABELS.values()) + list(DIARIZATION_POST_STEP_LABELS.values()):
        candidates.append(f"{label} 1200/1200")

    metrics = QFontMetrics(view.stage_label.font())
    for text in candidates:
        assert metrics.horizontalAdvance(text) <= view.stage_label.width(), text


def test_the_transcript_accumulates(view):
    view.begin("Ashford", 1)

    view.append_segment(0.0, 2.0, "Hello there", "Speaker 1")
    view.append_segment(2.0, 4.0, "And back", "Speaker 2")

    shown = view.transcript.toPlainText()
    assert "Hello there" in shown and "And back" in shown
    assert "Speaker 1" in shown and "Speaker 2" in shown


def test_a_segment_with_no_speaker_still_shows(view):
    """Speaker identification can be off, or can have failed."""
    view.begin("Lecture", 1)

    view.append_segment(0.0, 2.0, "One voice only", "")

    assert "One voice only" in view.transcript.toPlainText()


def test_timestamps_are_clickable_links(view):
    view.begin("Ashford", 1)

    view.append_segment(65.0, 70.0, "Later on", "Speaker 1")

    assert "timestamp:///65.0" in view.transcript.toHtml()
    assert "[1:05]" in view.transcript.toPlainText()


def test_status_text_is_escaped_not_rendered(view):
    """Status lines are bracketed and can contain anything ffmpeg said."""
    view.begin("Ashford", 1)

    view.append_status("[<b>not bold</b> & fine]")

    assert "<b>not bold</b>" in view.transcript.toPlainText()


# --- finishing ----------------------------------------------------------------


def test_finishing_shows_where_the_files_went(view, tmp_path):
    view.begin("Ashford", 3)

    view.on_finished(outputs(tmp_path, speaker_count=3))

    assert view.progress_bar.value() == 100
    assert view.stage_label.text() == "Done"
    assert "ashford" in view.summary.text()
    assert "3 speakers" in view.summary.text()
    # isHidden rather than isVisible: a widget whose parent was never shown
    # is not visible, which would make an isVisible assertion vacuous.
    assert not view.summary.isHidden()


def test_one_speaker_is_not_called_speakers(view, tmp_path):
    view.begin("Lecture", 1)

    view.on_finished(outputs(tmp_path, speaker_count=1))

    assert "1 speaker" in view.summary.text()
    assert "1 speakers" not in view.summary.text()


def test_no_speaker_count_is_simply_omitted(view, tmp_path):
    """Speaker identification off: do not claim zero speakers."""
    view.begin("Lecture", 1)

    view.on_finished(outputs(tmp_path, speaker_count=0))

    assert "speaker" not in view.summary.text()


def test_cancel_gives_way_to_the_finish_buttons(view, tmp_path):
    view.begin("Ashford", 1)
    assert view.reveal_button.isHidden()

    view.on_finished(outputs(tmp_path))

    assert view.cancel_button.isHidden()
    assert not view.reveal_button.isHidden()
    assert not view.new_trip_button.isHidden()


def test_failing_explains_and_offers_a_fresh_start(view):
    view.begin("Ashford", 2)

    view.on_failed("Could not download https://example.com/gone")

    assert "Could not finish" in view.summary.text()
    assert "example.com/gone" in view.summary.text()
    assert view.cancel_button.isHidden()
    assert not view.new_trip_button.isHidden()


def test_reveal_opens_the_trip_folder(view, tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr("src.ui.podcastnotes.run_view.subprocess.run",
                        lambda cmd, **k: opened.append(cmd))
    result = outputs(tmp_path)
    view.on_finished(result)

    view._reveal()

    assert opened == [["open", result.work_dir]]


def test_reveal_before_finishing_does_nothing(view, monkeypatch):
    opened = []
    monkeypatch.setattr("src.ui.podcastnotes.run_view.subprocess.run",
                        lambda cmd, **k: opened.append(cmd))

    view._reveal()

    assert opened == []


def test_cancel_is_reported(view):
    asked = []
    view.cancel_requested.connect(lambda: asked.append(True))

    view.cancel_button.click()

    assert asked == [True]


# --- reuse between trips ------------------------------------------------------


def test_beginning_again_clears_the_previous_trip(view, tmp_path):
    view.begin("First", 1)
    view.append_segment(0.0, 1.0, "old text", "Speaker 1")
    view.on_finished(outputs(tmp_path))

    view.begin("Second", 2)

    assert "old text" not in view.transcript.toPlainText()
    assert view.title.text() == "Second"
    assert view.progress_bar.value() == 0
    assert view.reveal_button.isHidden()
    assert not view.cancel_button.isHidden()


# --- playback -----------------------------------------------------------------


def test_the_player_is_disabled_until_there_is_audio(view):
    view.begin("Ashford", 1)

    assert not view.play_button.isEnabled()
    assert not view.position_slider.isEnabled()


def test_audio_enables_the_player(view, tmp_path):
    audio = tmp_path / "combined.m4a"
    audio.write_bytes(b"x")
    view.begin("Ashford", 1)

    view.set_audio(str(audio))

    assert view.play_button.isEnabled()
    assert view.position_slider.isEnabled()


def test_clicking_a_timestamp_before_audio_exists_is_harmless(view):
    view.begin("Ashford", 1)

    view._play_from(42.0)  # must not raise

    assert view.play_button.text() == "Play"


def test_every_stage_label_fits_without_clipping(view):
    """The width is fixed and the comment beside it says a clipped label
    defeats the point of having one. Moving the estimate to whole minutes made
    the strings longer, so the measurement has to be redone rather than
    assumed: "less than a minute left" is now the worst case at 211 of 230."""
    from src.ui.podcastnotes.trip_worker import remaining

    metrics = view.stage_label.fontMetrics()
    candidates = [
        remaining(0), remaining(40), remaining(60), remaining(785), remaining(3540),
        "Identifying speakers",
    ]

    for text in candidates:
        width = metrics.horizontalAdvance(text)
        assert width <= view.stage_label.width(), f"{text!r} needs {width}px"
