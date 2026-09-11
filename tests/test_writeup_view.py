"""
Tests for the write-up screen.

The rule this screen has to keep is the one the attribution module already
keeps: naming a speaker is a decision, so a suggestion arrives filled in and a
blank stays blank. A screen that quietly submitted the suggestion for a voice
somebody skipped would defeat the guarantee underneath it.

The rest is about a person being able to tell what happened: a failure has to
read as a next move, and the change log has to be legible rather than fifty
lines nobody reads.

Run with: python -m pytest tests/test_writeup_view.py -v
"""

import pytest

from src.podcastnotes.attribution import Suggestion, Voice
from src.podcastnotes.output import Documents
from src.ui.podcastnotes.writeup_view import (
    PANE_CONTEXT, PANE_DONE, PANE_QUESTIONS, PANE_SPEAKERS, PANE_WAITING,
    WriteUpView, _summarise,
)


@pytest.fixture
def view(qt_app):
    v = WriteUpView()
    yield v
    v.close()


def _voice(label, seconds=300.0, share=0.4, lines=None, first=12.0):
    return Voice(
        speaker=label,
        pyannote_label="P0",
        total_speech_s=seconds,
        share=share,
        lines=lines or [f"something {label} said at length about the site"],
        first_heard=first,
    )


VOICES = [_voice("Speaker 1"), _voice("Speaker 2", seconds=200.0, share=0.3)]


# --- naming speakers ----------------------------------------------------------


def test_a_suggestion_arrives_filled_in(view):
    view.ask_speakers(VOICES, [Suggestion("Speaker 1", name="Marcus Ellery",
                                          confidence="high", evidence="carries the site facts")])

    assert view._rows[0].chosen == "Marcus Ellery"


def test_a_voice_with_no_suggestion_starts_blank(view):
    view.ask_speakers(VOICES, [])

    assert view._rows[0].chosen == ""


def test_confirming_emits_only_the_names_that_were_filled_in(view):
    """A voice left blank stays unnamed. Submitting the suggestion for it would
    defeat the guarantee the attribution module makes underneath."""
    view.ask_speakers(VOICES, [
        Suggestion("Speaker 1", name="Marcus Ellery", confidence="high"),
        Suggestion("Speaker 2", name="Anya Petrov-Hale", confidence="high"),
    ])
    view._rows[1].name.setText("   ")

    got = {}
    view.speakers_confirmed.connect(got.update)
    view._confirm()

    assert got == {"Speaker 1": "Marcus Ellery"}


def test_typing_over_a_suggestion_wins(view):
    view.ask_speakers(VOICES, [Suggestion("Speaker 1", name="Wrong Person", confidence="high")])
    view._rows[0].name.setText("Marcus Ellery")

    got = {}
    view.speakers_confirmed.connect(got.update)
    view._confirm()

    assert got["Speaker 1"] == "Marcus Ellery"


def test_the_count_of_unnamed_voices_is_shown(view):
    view.ask_speakers(VOICES, [Suggestion("Speaker 1", name="Marcus", confidence="high")])

    assert "1 voice will stay unnamed" in view.unnamed.text()
    assert view.confirm.text() == "Continue with 1 unnamed"


def test_no_count_is_shown_once_everything_is_named(view):
    view.ask_speakers(VOICES, [
        Suggestion("Speaker 1", name="Marcus", confidence="high"),
        Suggestion("Speaker 2", name="Anya", confidence="high"),
    ])

    assert view.unnamed.text() == ""
    assert view.confirm.text() == "Use these names"


def test_a_barely_speaking_voice_is_called_out(view):
    """The diarizer inventing a person is common enough that the screen should
    say so rather than leaving somebody to wonder who the fourth person was."""
    view.ask_speakers([_voice("Speaker 4", seconds=21.8, share=0.01)], [])

    texts = [c.text() for c in view._rows[0].findChildren(type(view.unnamed))]
    assert any("inventing a person" in t for t in texts)


def test_a_substantial_voice_is_not_called_out(view):
    view.ask_speakers([_voice("Speaker 1", seconds=900.0)], [])

    texts = [c.text() for c in view._rows[0].findChildren(type(view.unnamed))]
    assert not any("inventing a person" in t for t in texts)


def test_asking_twice_does_not_stack_the_previous_voices(view):
    view.ask_speakers(VOICES, [])
    view.ask_speakers(VOICES, [])

    assert len(view._rows) == 2


def test_playing_a_voice_asks_for_its_own_moment(view):
    view.ask_speakers([_voice("Speaker 3", first=412.5)], [])

    got = []
    view._rows[0].play_requested.connect(got.append)
    view._rows[0]._play()

    assert got == [412.5]


# --- waiting and failing ------------------------------------------------------


def test_the_waiting_pane_says_what_is_happening(view):
    view.working("Reading the background for this trip", "Looking up: Lanternfish")

    assert view.panes.currentIndex() == PANE_WAITING
    assert view.stage.text() == "Reading the background for this trip"
    assert view.detail.text() == "Looking up: Lanternfish"


def test_a_failure_offers_a_way_forward(view):
    """`isHidden` rather than `isVisible`, because an offscreen widget whose
    window was never shown reports invisible either way, so the visible check
    would pass no matter what this method did."""
    view.working("Reading the background for this trip")

    view.on_failed("Your sign-in has expired. Sign in again on the setup screen.")

    assert view.problem.isHidden() is False
    assert view.retry.isHidden() is False
    assert "expired" in view.problem.text()
    assert view.bar.isHidden() is True, "a spinner kept running under a failure"
    assert view.title.text() == "The write-up stopped"


def test_a_failure_clears_what_it_was_last_doing(view):
    """Left in place, "Looking up: Miles Nadeau" sits under the failure message
    and reads as though that lookup were still running."""
    view.working("Reading the background for this trip", "Looking up: Miles Nadeau")

    view.on_failed("Could not reach your company's knowledge search.")

    assert view.detail.text() == ""
    assert view.stage.text() == ""


def test_the_retry_button_asks_the_window_to_try_again(view):
    view.on_failed("Something went wrong.")

    fired = []
    view.retry_requested.connect(lambda: fired.append(True))
    view.retry.click()

    assert fired == [True]


def test_starting_a_new_step_clears_the_last_failure(view):
    """Otherwise a stale error sits under a running step and reads as though
    the retry failed too."""
    view.on_failed("Something went wrong.")

    view.working("Working out who is speaking")

    assert view.problem.isHidden()
    assert view.retry.isHidden()


# --- the finished documents ---------------------------------------------------


def test_the_documents_pane_summarises_the_changes(view):
    view.show_documents(Documents(transcript="t", summary="s"),
                        notes=["Corrected: a -> b (2x)", "Flagged: c -> d (1x)"])

    assert view.panes.currentIndex() == PANE_DONE
    assert "1 correction applied" in view.changes.text()
    assert "1 marked as uncertain" in view.changes.text()


def test_dropped_markers_are_warned_about(view):
    """An uncertain term that did not survive into the document has quietly
    become a fact, which is the one thing worth interrupting somebody for."""
    view.show_documents(Documents(transcript="t", summary="s",
                                  dropped_markers=["Errol Marchetti"]))

    assert view.warnings.isHidden() is False
    assert "Errol Marchetti" in view.warnings.text()


def test_no_warning_when_every_marker_survived(view):
    view.show_documents(Documents(transcript="t", summary="s"))

    assert view.warnings.isHidden()


def test_a_clean_run_says_so_rather_than_showing_nothing():
    assert _summarise([]) == "No corrections were needed."


def test_held_back_corrections_are_explained_not_just_counted():
    text = _summarise(["Held back for review: a -> b"])

    assert "removed words" in text


def test_the_change_log_is_counted_rather_than_listed():
    """A real trip produces around fifty of these and nobody reads a list that
    long, so the counts go on screen and the detail goes in the file."""
    notes = [f"Corrected: term{i} -> fixed{i} (1x)" for i in range(50)]

    text = _summarise(notes)

    assert "50 corrections applied" in text
    assert "term7" not in text


# --- reading the documents before publishing ----------------------------------


def test_the_summary_is_shown_first(view):
    """A screen that reports "49 corrections applied" and shows none of the
    result is asking to be trusted rather than checked."""
    view.show_documents(Documents(transcript="THE TRANSCRIPT", summary="THE SUMMARY"))

    assert "THE SUMMARY" in view.document.toPlainText()
    assert view.show_summary.isChecked() is True


def test_the_transcript_can_be_read_too(view):
    view.show_documents(Documents(transcript="THE TRANSCRIPT", summary="THE SUMMARY"))

    view.show_transcript.click()

    assert "THE TRANSCRIPT" in view.document.toPlainText()
    assert view.show_transcript.isChecked() is True
    assert view.show_summary.isChecked() is False, "both tabs looked selected at once"


def test_switching_back_works(view):
    view.show_documents(Documents(transcript="THE TRANSCRIPT", summary="THE SUMMARY"))
    view.show_transcript.click()

    view.show_summary.click()

    assert "THE SUMMARY" in view.document.toPlainText()


def test_markdown_is_rendered_rather_than_shown_raw(view):
    view.show_documents(Documents(transcript="t", summary="# A heading\n\nSome text."))

    shown = view.document.toPlainText()
    assert "A heading" in shown
    assert "#" not in shown


def test_the_heading_stops_saying_it_is_still_working(view):
    """Left as it was, the finished screen reads as though the job is still
    running and the publish button is premature."""
    view.working("Writing the transcript and the summary")

    view.show_documents(Documents(transcript="t", summary="s"))

    assert view.title.text() == "Ready to publish"


def test_the_heading_goes_back_when_a_later_step_runs(view):
    """A retry after publishing failed must not still say "Ready to publish"."""
    view.show_documents(Documents(transcript="t", summary="s"))

    view.working("Applying your answers")

    assert view.title.text() == "Writing this up"


# --- the question rounds -------------------------------------------------------


class _Q:
    def __init__(self, marker, ask="What did they mean?", why=""):
        self.marker = marker
        self.ask = ask
        self.why_it_matters = why
        self.occurrences = 1


class _Round:
    def __init__(self, questions, remaining=0):
        self.questions = questions
        self.remaining = remaining


def test_each_question_gets_its_own_box(view):
    view.ask_questions(_Round([_Q("Kestler"), _Q("z-lanes")]))

    assert view.panes.currentIndex() == PANE_QUESTIONS
    assert len(view._questions) == 2


def test_only_answered_questions_are_sent(view):
    """Leaving one blank keeps its marker, which is the honest outcome and the
    reason the markers exist."""
    view.ask_questions(_Round([_Q("Kestler"), _Q("z-lanes")]))
    view._questions[0].box.setText("It is Kessler.")
    view._questions[1].box.setText("   ")

    got = {}
    view.answers_given.connect(got.update)
    view._send_answers()

    assert got == {"Kestler": "It is Kessler."}


def test_skipping_sends_an_empty_answer_rather_than_nothing(view):
    """Skipping is a real answer to "can you settle any of these", so the
    pipeline still advances the round and the markers simply stay."""
    view.ask_questions(_Round([_Q("Kestler")]))
    view._questions[0].box.setText("would be ignored")

    got = []
    view.answers_given.connect(got.append)
    view.skip.click()

    assert got == [{}]


def test_the_number_still_waiting_is_shown(view):
    """Three rounds of four is the cap, so somebody with seventeen markers
    should know some will not be asked."""
    view.ask_questions(_Round([_Q("Kestler")], remaining=13))

    assert "13 more" in view.round_blurb.text()


def test_no_leftovers_means_no_mention_of_them(view):
    view.ask_questions(_Round([_Q("Kestler")], remaining=0))

    assert "more" not in view.round_blurb.text()


def test_a_second_round_replaces_the_first(view):
    view.ask_questions(_Round([_Q("Kestler"), _Q("z-lanes")]))

    view.ask_questions(_Round([_Q("60 hold")]))

    assert len(view._questions) == 1
    assert view._questions[0].question.marker == "60 hold"


def test_why_it_matters_is_shown_when_there_is_a_reason(view):
    view.ask_questions(_Round([_Q("Kestler", why="It gets published against a name.")]))

    labels = [c.text() for c in view._questions[0].findChildren(type(view.unnamed))]
    assert any("published against a name" in t for t in labels)


# --- reviewing the context map before it is used --------------------------------


def _map(rows=None, **kwargs):
    from src.podcastnotes.context import ContextMap

    return ContextMap(likely_errors=rows if rows is not None else [
        {"heard": "Ridgelane", "probably": "Ridgeline", "confidence": "high",
         "evidence": "the project plan spells it Ridgeline"},
        {"heard": "Errol Markety", "probably": "Errol Marchetti", "confidence": "low"},
    ], **kwargs)


def test_every_proposed_correction_is_listed(view):
    view.review_context(_map())

    assert view.panes.currentIndex() == PANE_CONTEXT
    assert len(view._corrections) == 2


def test_corrections_start_ticked(view):
    """The map is usually right, so the default is to accept it and the work is
    in spotting the exception."""
    view.review_context(_map())

    assert all(box.tick.isChecked() for box in view._corrections)


def test_unticking_one_reports_it_as_rejected(view):
    view.review_context(_map())
    view._corrections[1].tick.setChecked(False)

    got = []
    view.context_approved.connect(got.append)
    view.approve.click()

    assert got == [["Errol Markety"]]


def test_approving_everything_rejects_nothing(view):
    view.review_context(_map())

    got = []
    view.context_approved.connect(got.append)
    view.approve.click()

    assert got == [[]]


def test_the_number_being_dropped_is_shown(view):
    view.review_context(_map())
    view._corrections[0].tick.setChecked(False)

    assert "1 correction will not be applied" in view.rejected_count.text()


def test_an_uncertain_correction_says_it_will_be_marked(view):
    """So somebody can tell the difference between a change that will be
    invisible and one that leaves a [?] in the document."""
    view.review_context(_map())

    assert "marked uncertain" in view._corrections[1].label
    assert "marked uncertain" not in view._corrections[0].label


def test_what_was_searched_is_reported(view):
    """The user should be able to see what it looked at before trusting it."""
    view.review_context(_map(people=[{"name": "Anya"}], searches=["a", "b", "c"]))

    assert "1 people" in view.found.text()
    assert "3 searches" in view.found.text()


def test_reviewing_twice_does_not_stack_the_rows(view):
    view.review_context(_map())

    view.review_context(_map())

    assert len(view._corrections) == 2


def test_a_map_with_no_corrections_is_still_reviewable(view):
    view.review_context(_map(rows=[]))

    assert view.panes.currentIndex() == PANE_CONTEXT
    assert view._corrections == []


# --- asking, when a guess would be a coin toss ----------------------------------


def _unsure(label="Speaker 4"):
    return Suggestion(label, name="Anya Petrov-Hale", confidence="low",
                      evidence="Only backchannels, could be Anya or Devan.")


def test_a_low_confidence_guess_is_not_filled_in(view):
    """Across three runs of the same recording with identical diarization, the
    22-second speaker came back Anya, Anya, then Devan. In the box it looks the
    same as the three confident rows above it and gets accepted with them."""
    view.ask_speakers([_voice("Speaker 4", seconds=21.8, share=0.01)], [_unsure()])

    assert view._rows[0].chosen == ""


def test_the_row_asks_outright(view):
    view.ask_speakers([_voice("Speaker 4", seconds=21.8, share=0.01)], [_unsure()])

    texts = [c.text() for c in view._rows[0].findChildren(type(view.unnamed))]
    assert any("Who is this?" in t for t in texts)


def test_the_guess_is_still_offered_as_a_hint(view):
    """Withholding it entirely would throw away the only lead there is."""
    view.ask_speakers([_voice("Speaker 4", seconds=21.8, share=0.01)], [_unsure()])

    texts = " ".join(c.text() for c in view._rows[0].findChildren(type(view.unnamed)))
    assert "not confident enough to fill in" in texts
    assert "could be Anya or Devan" in texts


def test_a_confident_guess_is_still_filled_in(view):
    view.ask_speakers([_voice("Speaker 1")],
                      [Suggestion("Speaker 1", name="Marcus Ellery", confidence="high")])

    assert view._rows[0].chosen == "Marcus Ellery"
    assert view._rows[0].asking is False


def test_a_suggestion_with_no_stated_confidence_is_not_trusted(view):
    """An absent confidence is not a high one. Treating it as fillable would
    make the safest default the one nobody wrote down."""
    view.ask_speakers([_voice("Speaker 1")], [Suggestion("Speaker 1", name="Marcus Ellery")])

    assert view._rows[0].chosen == ""


def test_typing_a_name_answers_the_question(view):
    view.ask_speakers([_voice("Speaker 4", seconds=21.8)], [_unsure()])
    view._rows[0].name.setText("Anya Petrov-Hale")

    got = {}
    view.speakers_confirmed.connect(got.update)
    view._confirm()

    assert got == {"Speaker 4": "Anya Petrov-Hale"}


def test_carrying_on_without_answering_is_a_deliberate_act(view):
    """Leaving it blank is allowed and sometimes right, but the button says
    which of the two things pressing it does."""
    view.ask_speakers([_voice("Speaker 4", seconds=21.8)], [_unsure()])

    assert view.confirm.text() == "Continue with 1 unnamed"
    assert "will stay unnamed in the documents" in view.unnamed.text()


def test_the_button_goes_back_once_the_question_is_answered(view):
    view.ask_speakers([_voice("Speaker 4", seconds=21.8)], [_unsure()])

    view._rows[0].name.setText("Anya Petrov-Hale")

    assert view.confirm.text() == "Use these names"


# --- playing a voice -------------------------------------------------------------


class _FakePlayer:
    """Stands in for QMediaPlayer, with the load-then-seek behaviour that matters."""

    def __init__(self, seekable=False):
        self._seekable = seekable
        self.position = None
        self.played = 0
        self.stopped = 0

    def isSeekable(self):
        return self._seekable

    def setPosition(self, ms):
        self.position = ms

    def play(self):
        self.played += 1

    def stop(self):
        self.stopped += 1


def _ready_to_play(view, tmp_path, seekable=False):
    audio = tmp_path / "combined.m4a"
    audio.write_bytes(b"not really audio")
    view.set_audio(str(audio))
    view._player = _FakePlayer(seekable=seekable)
    return view._player


def test_a_seek_before_the_media_loads_is_held_not_lost(view, tmp_path):
    """Measured on the real file: setPosition immediately after setSource leaves
    the position at 0 with status LoadingMedia, and playback then starts at 0:00
    of a forty-two minute recording. Clicking Play on the third speaker and
    hearing the first is why this was reported as the buttons doing nothing."""
    player = _ready_to_play(view, tmp_path, seekable=False)

    view._play_from(412.5)

    assert player.position is None, "seeked before the media was ready"
    assert player.played == 0
    assert view._wanted_at == 412500


def test_the_held_seek_is_applied_once_the_media_is_ready(view, tmp_path):
    from PyQt6.QtMultimedia import QMediaPlayer

    player = _ready_to_play(view, tmp_path, seekable=False)
    view._play_from(412.5)

    view._media_status_changed(QMediaPlayer.MediaStatus.LoadedMedia)

    assert player.position == 412500
    assert player.played == 1
    assert view._wanted_at is None, "the seek should not be applied twice"


def test_a_seek_once_loaded_happens_straight_away(view, tmp_path):
    """The second click, which always worked."""
    player = _ready_to_play(view, tmp_path, seekable=True)

    view._play_from(90.0)

    assert player.position == 90000
    assert player.played == 1


def test_a_status_change_with_nothing_pending_does_not_replay(view, tmp_path):
    from PyQt6.QtMultimedia import QMediaPlayer

    player = _ready_to_play(view, tmp_path, seekable=True)
    view._play_from(90.0)

    view._media_status_changed(QMediaPlayer.MediaStatus.BufferedMedia)

    assert player.played == 1


def test_a_status_that_is_not_ready_does_not_seek(view, tmp_path):
    from PyQt6.QtMultimedia import QMediaPlayer

    player = _ready_to_play(view, tmp_path, seekable=False)
    view._play_from(412.5)

    view._media_status_changed(QMediaPlayer.MediaStatus.LoadingMedia)

    assert player.position is None
    assert view._wanted_at == 412500


def test_playing_a_second_voice_stops_the_first(view, tmp_path):
    """Two people talking at once is nobody's idea of help."""
    player = _ready_to_play(view, tmp_path, seekable=True)

    view._play_one(10.0)
    after_first = player.stopped
    view._play_one(200.0)

    assert player.stopped > after_first, "the first voice kept playing"
    assert player.position == 200000


def test_stopping_forgets_a_pending_seek(view, tmp_path):
    """Otherwise the media finishes loading and starts playing after the person
    has already asked for silence."""
    from PyQt6.QtMultimedia import QMediaPlayer

    player = _ready_to_play(view, tmp_path, seekable=False)
    view._play_from(412.5)

    view.stop_playing()
    view._media_status_changed(QMediaPlayer.MediaStatus.LoadedMedia)

    assert player.played == 0


def test_missing_audio_is_ignored_rather_than_crashing(view, tmp_path):
    view.set_audio(str(tmp_path / "gone.m4a"))

    view._play_from(10.0)

    assert view._player is None
