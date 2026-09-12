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
    _Readable,
    PANE_DONE, PANE_QUESTIONS, PANE_SPEAKERS, PANE_WAITING,
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
    view._rows[1].name.setCurrentText("   ")

    got = {}
    view.speakers_confirmed.connect(got.update)
    view._confirm()

    assert got == {"Speaker 1": "Marcus Ellery"}


def test_typing_over_a_suggestion_wins(view):
    view.ask_speakers(VOICES, [Suggestion("Speaker 1", name="Wrong Person", confidence="high")])
    view._rows[0].name.setCurrentText("Marcus Ellery")

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


def test_the_finished_screen_does_not_recount_the_corrections(view):
    """By the time somebody is reading the finished pair, how many
    substitutions happened three steps ago is not news, and it pushed the thing
    they came for further down the screen."""
    view.show_documents(Documents(transcript="t", summary="s"),
                        notes=["Corrected: a -> b (2x)", "Flagged: c -> d (1x)"])

    assert view.panes.currentIndex() == PANE_DONE
    assert view.changes.isHidden() is True


def test_a_missing_marker_is_still_worth_interrupting_for(view):
    """The counts go, this stays: a marker that did not survive has quietly
    become a fact."""
    view.show_documents(Documents(transcript="t", summary="s",
                                  dropped_markers=["Errol Marchetti"]))

    assert view.warnings.isHidden() is False


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


def test_leaving_them_all_blank_is_how_you_skip(view):
    """There is one button now. A separate "Skip these" implied the two were
    different, and leaving every box empty already did the same thing: the
    round advances and the markers simply stay."""
    view.ask_questions(_Round([_Q("Kestler"), _Q("z-lanes")]))

    got = []
    view.answers_given.connect(got.append)
    view.send_answers.click()

    assert got == [{}]


def test_the_button_says_which_of_the_two_things_it_will_do(view):
    """Skipping is allowed, and it should be a thing somebody chose rather
    than something they did without noticing."""
    view.ask_questions(_Round([_Q("Kestler")]))
    assert "Skip" in view.send_answers.text()

    view._questions[0].box.setText("It is Crockett.")

    assert view.send_answers.text() == "Use these answers"


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
    view._rows[0].name.setCurrentText("Anya Petrov-Hale")

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

    view._rows[0].name.setCurrentText("Anya Petrov-Hale")

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


# --- saying what publishing did --------------------------------------------------


class _Published:
    def __init__(self, copied=True, browser=True, problems=(), path="/tmp/write-up.md"):
        self.copied = copied
        self.browser_opened = browser
        self.problems = problems
        self.markdown_path = path

    def summary(self):
        return "Saved as write-up.md, copied, and a blank document is open."


def test_publishing_says_what_it_did(view):
    """It did three things and mentioned none of them. The browser tab opening
    was the only sign, which is indistinguishable from the app having done
    nothing but open a tab."""
    view.show_documents(Documents(transcript="t", summary="s"))

    view.show_published(_Published())

    assert view.published.isHidden() is False
    assert "Saved as write-up.md" in view.published.text()


def test_it_says_to_paste_from_markdown(view):
    """A plain paste puts the raw Markdown source on the page as literal text.
    The instruction has existed in publish.py since it was written and was
    never once shown."""
    view.show_documents(Documents(transcript="t", summary="s"))

    view.show_published(_Published())

    assert "Paste from Markdown" in view.published.text()


def test_a_clipboard_failure_is_shown_not_logged(view):
    """Otherwise somebody pastes whatever they copied earlier into a blank
    Google Doc and has no idea why."""
    view.show_documents(Documents(transcript="t", summary="s"))

    view.show_published(_Published(
        copied=False, problems=("Could not reach the clipboard, so copy from the file.",)
    ))

    assert "Could not reach the clipboard" in view.published.text()


def test_there_is_no_paste_advice_when_nothing_was_copied(view):
    """Telling somebody to paste when the clipboard failed is worse than
    saying nothing."""
    view.show_documents(Documents(transcript="t", summary="s"))

    view.show_published(_Published(copied=False, problems=("clipboard failed",)))

    assert "Paste from Markdown" not in view.published.text()


def test_a_new_write_up_does_not_inherit_the_last_publish_message(view):
    view.show_documents(Documents(transcript="t", summary="s"))
    view.show_published(_Published())

    view.show_documents(Documents(transcript="t2", summary="s2"))

    assert view.published.isHidden() is True


def test_the_heading_says_where_in_the_job_it_is(view):
    view.working("Reading the background for this trip", phase="Background")

    assert view.title.text() == "Background"


def test_the_heading_changes_between_phases(view):
    view.working("Reading the background", phase="Background")
    first = view.title.text()
    view.working("Writing the transcript and the summary", phase="Writing")

    assert view.title.text() != first


def test_the_round_speaks_in_the_first_person(view):
    """"Nobody could work out" is the tool talking about itself in the third
    person. It did the work; it can say so."""
    view.ask_questions(_Round([_Q("Kestler")]))

    assert "I could not work out" in view.round_blurb.text()
    assert "nobody" not in view.round_blurb.text().lower()


def test_the_preview_says_which_document_is_on_screen(view):
    """Both open almost identically, and the only signal was which toggle
    happened to be checked."""
    view.show_documents(Documents(transcript="T", summary="S"))
    assert "summary" in view.showing.text().lower()

    view.show_transcript.click()

    assert "transcript" in view.showing.text().lower()


# --- something to look at while it works -----------------------------------------


def test_findings_appear_as_they_are_found(view):
    """Real information rather than decoration: a name appearing is proof the
    thing is working, in a way a spinner never is."""
    view.working("Reading the background", phase="Background")

    view.note_finding("Searched for Miles Nadeau")
    view.note_finding("Searched for Simone Vasari")

    assert view.findings.isHidden() is False
    assert "Miles Nadeau" in view.findings.toPlainText()
    assert "Simone Vasari" in view.findings.toPlainText()


def test_findings_belong_to_the_step_that_found_them(view):
    view.working("Reading the background", phase="Background")
    view.note_finding("Searched for Miles Nadeau")

    view.working("Writing the documents", phase="Writing")

    assert view.findings.toPlainText() == ""
    assert view.findings.isHidden() is True


def test_the_list_does_not_grow_without_limit(view):
    view.working("Reading the background", phase="Background")

    for i in range(200):
        view.note_finding(f"Searched for thing {i}")

    assert len(view.findings.toPlainText().splitlines()) <= 40


@pytest.mark.parametrize("seconds,expected", [
    (240, "about 4 minutes left"),
    (60, "about 1 minute left"),
    (30, "less than a minute left"),
    (0, "Nearly there"),
])
def test_the_estimate_is_in_whole_minutes(view, seconds, expected):
    """Same as the transcription estimate, for the same reason: the number
    swings, and seconds claim a precision it does not have."""
    view.set_estimate(0.5, seconds)

    assert expected in view.estimate.text()


def test_the_estimate_says_it_is_an_estimate(view):
    """The agentic loop cannot predict its own length. Saying so is the
    difference between a guess and a claim."""
    view.set_estimate(0.5, 240)

    assert "Estimate" in view.estimate.text()


def test_the_bar_moves(view):
    view.set_estimate(0.42, 200)

    assert view.bar.value() == 42


def test_the_bar_cannot_exceed_full(view):
    view.set_estimate(1.8, 0)

    assert view.bar.value() == 100


def test_a_new_step_starts_the_bar_over(view):
    view.set_estimate(0.9, 20)

    view.working("Writing the documents", phase="Writing")

    assert view.bar.value() == 0
    assert view.estimate.text() == ""


# --- saying what is wrong, with the document in front of you ---------------------


def _docs():
    return Documents(transcript="t", summary="s")


def test_the_note_goes_out_as_typed(view):
    view.show_documents(_docs())
    view.notes_box.setPlainText("  The coordinator is Simone Vasari.  ")

    got = []
    view.revision_requested.connect(got.append)
    view.send_revision.click()

    assert got == ["The coordinator is Simone Vasari."]


def test_an_empty_box_sends_nothing(view):
    """It is a full agentic Glean pass. Sending it whitespace costs minutes."""
    view.show_documents(_docs())
    view.notes_box.setPlainText("   \n  ")

    got = []
    view.revision_requested.connect(got.append)
    view.send_revision.click()

    assert got == []


def test_a_failed_pass_does_not_throw_away_what_they_typed(view):
    """Three sentences about who was in the room, lost because Glean was down,
    and they have to remember them again."""
    view.show_documents(_docs())
    view.notes_box.setPlainText("Speaker 3 is Miles Nadeau")

    view.on_failed("Glean is unavailable")

    assert view.notes_box.toPlainText() == "Speaker 3 is Miles Nadeau"


def test_the_box_clears_once_the_rewrite_lands(view):
    view.show_documents(_docs())
    view.notes_box.setPlainText("Speaker 3 is Miles Nadeau")

    view.show_documents(_docs())

    assert view.notes_box.toPlainText() == ""


def test_the_last_rewrite_says_so_before_it_is_spent(view):
    """Somebody typing their fifth note should know it is their last before
    they spend a quarter of an hour on it."""
    view.set_revisions(1)

    assert "One more rewrite" in view.revisions_left.text()
    assert view.send_revision.isHidden() is False


def test_plenty_left_is_not_worth_saying(view):
    view.set_revisions(5)

    assert view.revisions_left.text() == ""


def test_the_end_of_the_road_offers_the_document_instead(view):
    """Past five, a person on their sixth "still wrong" has hit something this
    tool cannot fix, and a disabled button with no explanation is worse than
    the limit."""
    view.set_revisions(0)

    assert view.send_revision.isHidden() is True, "a dead button invites the click"
    assert view.notes_box.isHidden() is True
    assert "edit the document yourself" in view.revise_blurb.text()


def test_what_it_could_not_do_is_said_out_loud(view):
    """Doing nothing is indistinguishable from ignoring them, so they retype it
    and pay for another pass."""
    view.show_documents(_docs())
    view.show_impossible(["Nobody in the recording mentions a Dr Okafor."])

    assert view.impossible.isHidden() is False
    assert "Dr Okafor" in view.impossible.text()
    assert "changed nothing for it" in view.impossible.text()


def test_a_pass_with_nothing_to_report_says_nothing(view):
    view.show_documents(_docs())
    view.show_impossible(["something"])

    view.show_impossible([])

    assert view.impossible.isHidden() is True


def test_a_reopened_box_says_what_it_is_for_again(view):
    """Reaching the limit rewrites the blurb. A new trip in the same window
    would otherwise open on "this has been rewritten five times"."""
    view.set_revisions(0)

    view.set_revisions(5)

    assert "Say it in your own words" in view.revise_blurb.text()


def test_the_document_keeps_its_measure_however_wide_the_window(view):
    """`document().setTextWidth()` alone does not survive: QTextBrowser resets
    it to the viewport width on every resize, so the measure set at
    construction vanished the moment the window was shown. Green tests, wrong
    screen, caught by rendering it."""
    from src.ui.podcastnotes.writeup_view import READABLE_WIDTH

    view.document.resize(1500, 600)
    view.document.apply_measure()

    shape = view.document.document().rootFrame().frameFormat()
    column = 1500 - shape.leftMargin() - shape.rightMargin()
    assert column <= READABLE_WIDTH, f"prose ran {column}px wide"


def test_a_narrow_window_still_leaves_a_gutter(view):
    """Below the measure there is nothing to centre, and text against the frame
    edge is worse than a short line."""
    view.document.resize(400, 300)
    view.document.apply_measure()

    assert view.document.document().rootFrame().frameFormat().leftMargin() == _Readable.GUTTER


def test_the_measure_does_not_walk_inwards_on_repeated_resizes(view):
    """Measuring the viewport and then setting the margins that shrink it is a
    feedback loop: every resize takes another bite and the column walks towards
    nothing. It has to be computed from the widget width."""
    view.document.resize(1500, 600)
    view.document.apply_measure()
    first = view.document.document().rootFrame().frameFormat().leftMargin()

    for _ in range(5):
        view.document.apply_measure()

    assert view.document.document().rootFrame().frameFormat().leftMargin() == first


def test_being_resized_is_what_applies_it(view):
    """The measure has to be re-derived on resize, because that is the moment
    QTextBrowser throws the old one away."""
    from PyQt6.QtCore import QSize
    from PyQt6.QtGui import QResizeEvent

    view.document.resize(1500, 600)

    view.document.resizeEvent(QResizeEvent(QSize(1500, 600), QSize(400, 600)))

    shape = view.document.document().rootFrame().frameFormat()
    assert shape.leftMargin() > _Readable.GUTTER


def test_the_prose_is_not_wider_than_the_viewport(view):
    """`setViewportMargins` narrows the viewport without re-laying out the
    text, so the lines kept their old width, spilled past the right edge and
    raised a horizontal scrollbar under clipped prose."""
    view.show_documents(_docs())
    view.document.resize(1500, 600)
    view.document.apply_measure()

    assert view.document.viewportMargins().left() == 0
    assert view.document.viewportMargins().right() == 0


def test_a_pane_that_asks_owns_its_heading(view):
    """Left to whatever set it last, a retry that succeeds lands on the
    speakers under "The write-up stopped", which says the opposite of what the
    screen is doing. Caught by rendering, not by a test."""
    view.on_failed("Glean is unavailable")

    view.ask_speakers([_voice("Speaker 1")])

    assert view.title.text() == "Speakers"


def test_the_questions_pane_owns_its_heading_too(view):
    view.on_failed("Glean is unavailable")

    view.ask_questions(_Round([_Q("Kestler")]))

    assert view.title.text() == "Questions"


def test_the_progress_bar_is_as_wide_as_the_prose_it_sits_with(view):
    """Centred by the layout, a widget is given its size hint rather than
    stretched, and a progress bar's hint is about 180px. The cap alone left a
    stub in the middle of a thousand-pixel window that read as a decoration
    rather than as progress."""
    from src.ui.podcastnotes.writeup_view import READABLE_WIDTH

    assert view.bar.width() == READABLE_WIDTH
    assert view.findings.width() == READABLE_WIDTH


# --- offering a name rather than a blank field -----------------------------------


PEOPLE = ["Anya Petrov-Hale", "Marcus Ellery", "Miles Nadeau"]


def test_the_people_the_search_found_are_on_offer(view):
    """Almost every voice on a work recording is one of them, and recognising
    "Anya Petrov-Hale" is easier than spelling it from memory."""
    view.ask_speakers([_voice("Speaker 1")], names=PEOPLE)

    box = view._rows[0].name
    assert [box.itemText(i) for i in range(box.count())] == [""] + PEOPLE


def test_blank_is_the_first_option_and_the_one_it_starts_on(view):
    """Leaving a voice unnamed has to be something you choose rather than
    something you clear, and nothing should be selected by accident because the
    list happened to open on it."""
    view.ask_speakers([_voice("Speaker 1")], names=PEOPLE)

    assert view._rows[0].name.currentText() == ""
    assert view._rows[0].chosen == ""


def test_a_name_it_has_never_heard_of_can_still_be_typed(view):
    """The list is never complete. A recording can contain somebody the company
    has never written down, and a closed list would make them unnameable."""
    view.ask_speakers([_voice("Speaker 1")], names=PEOPLE)

    view._rows[0].name.setCurrentText("Idris Vahali")

    assert view._rows[0].chosen == "Idris Vahali"


def test_typing_does_not_add_to_the_list(view):
    """Left on Qt's default, an abandoned half-typed name becomes a permanent
    option, and the next row down offers "Anya Pe" beside "Anya Petrov-Hale"."""
    view.ask_speakers([_voice("Speaker 1")], names=PEOPLE)
    box = view._rows[0].name
    before = box.count()

    # Enter is what makes Qt insert on the default policy, so pressing it is
    # the only way this is actually tested.
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    box.setCurrentText("Anya Pe")
    QTest.keyClick(box.lineEdit(), Qt.Key.Key_Return)

    assert box.count() == before
    assert "Anya Pe" not in [box.itemText(i) for i in range(box.count())]


def test_the_rows_own_suggestion_leads_the_list(view):
    view.ask_speakers(
        [_voice("Speaker 1")],
        [Suggestion("Speaker 1", name="Anya Petrov-Hale", confidence="high")],
        names=PEOPLE,
    )

    assert view._rows[0].name.itemText(1) == "Anya Petrov-Hale"


def test_a_name_is_offered_once_however_many_places_it_came_from(view):
    view.ask_speakers(
        [_voice("Speaker 1")],
        [Suggestion("Speaker 1", name="Anya Petrov-Hale", confidence="high")],
        names=["anya petrov-hale", "Marcus Ellery"],
    )

    box = view._rows[0].name
    assert [box.itemText(i) for i in range(box.count())] == [
        "", "Anya Petrov-Hale", "Marcus Ellery",
    ]


def test_a_guess_it_is_unsure_of_is_offered_but_not_chosen(view):
    """The rule the whole screen turns on. Being offered a guess to accept is a
    different thing from having it accepted for you, and Speaker 4 came back as
    a different person on two runs of identical diarization."""
    view.ask_speakers([_voice("Speaker 4")], [_unsure("Speaker 4")], names=PEOPLE)

    box = view._rows[0].name
    assert box.currentText() == ""
    assert "Anya Petrov-Hale" in [box.itemText(i) for i in range(box.count())]


def test_choosing_from_the_list_counts_as_naming_the_voice(view):
    """The unnamed count is driven by the field, so a combo that reports
    through a different signal leaves the button saying "continue with 1
    unnamed" after the name has been picked."""
    view.ask_speakers([_voice("Speaker 1")], names=PEOPLE)

    view._rows[0].name.setCurrentIndex(1)

    assert "unnamed" not in view.confirm.text()
