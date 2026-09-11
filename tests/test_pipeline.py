"""
Tests for the write-up sequence.

Two properties carry the weight here and neither is obvious from reading the
code.

Corrections are re-applied from the original transcript every time, never on
top of the previous pass. Answering a question changes an earlier decision, and
rebuilding from already-corrected text substitutes into text that has already
been substituted.

State survives a closed window. These runs take minutes with a person answering
questions in the middle, so losing the context map because somebody closed
something would be worse than never having asked.

Run with: python -m pytest tests/test_pipeline.py -v
"""

import json

import pytest

from src.podcastnotes import pipeline, qa
from src.podcastnotes.context import ContextMap
from src.podcastnotes.pipeline import OutOfOrder, Step, WriteUp


def _seg(speaker, text, start=0.0):
    return {"speaker": speaker, "text": text, "start": start, "end": start + 1.0, "words": []}


def _original():
    return {
        "segments": [
            _seg("Speaker 1", "We went to Ridgelane Northgate.", 0.0),
            _seg("Speaker 2", "The director was Kestler.", 1.0),
        ],
        "speakers": [
            {"id": "Speaker 1", "pyannote_label": "P0", "total_speech_s": 10.0,
             "embedding": [1.0, 0.0, 0.0]},
            {"id": "Speaker 2", "pyannote_label": "P1", "total_speech_s": 8.0,
             "embedding": [0.0, 1.0, 0.0]},
        ],
    }


def _map():
    return ContextMap(likely_errors=[
        {"heard": "Ridgelane", "probably": "Ridgeline", "confidence": "high"},
        {"heard": "Kestler", "probably": "Kessler", "confidence": "low"},
    ])


def _ready(tmp_path, step=Step.ATTRIBUTE):
    """A write-up already past the context stage."""
    run = WriteUp(work_dir=str(tmp_path), description="Ashford", original=_original())
    run.context_map = _map()
    run.step = step
    return run


# --- order --------------------------------------------------------------------


def test_a_step_cannot_run_before_the_one_it_depends_on(tmp_path):
    """Running early does not produce a worse result, it produces a confident
    result computed from nothing."""
    run = WriteUp(work_dir=str(tmp_path), original=_original())

    with pytest.raises(OutOfOrder):
        run.write_documents()


def test_correcting_needs_the_context_map_first(tmp_path):
    run = WriteUp(work_dir=str(tmp_path), original=_original())

    with pytest.raises(OutOfOrder):
        run.apply_corrections()


def test_the_step_only_ever_moves_forward(tmp_path):
    """A repeated step must not rewind a run that has gone further."""
    run = _ready(tmp_path, step=Step.WRITE)

    run.apply_corrections()

    assert run.step is Step.WRITE


def test_confirming_speakers_opens_the_question_rounds(tmp_path):
    run = _ready(tmp_path)

    run.confirm_speakers({"Speaker 1": "Marcus"}, library_path=str(tmp_path / "s.db"))

    assert run.step is Step.QUESTIONS


# --- corrections are rebuilt, not layered -------------------------------------


def test_the_working_transcript_is_corrected(tmp_path):
    run = _ready(tmp_path)

    text = " ".join(s["text"] for s in run.working()["segments"])

    assert "Ridgeline" in text
    assert "[?Kessler]" in text


def test_the_original_is_never_modified(tmp_path):
    run = _ready(tmp_path)

    run.working()
    run.working()

    assert run.original["segments"][0]["text"] == "We went to Ridgelane Northgate."


def test_rebuilding_twice_gives_the_same_result(tmp_path):
    """The tell for corrections being layered on top of each other."""
    run = _ready(tmp_path)

    first = json.dumps(run.working(), sort_keys=True)
    second = json.dumps(run.working(), sort_keys=True)

    assert first == second


def test_a_later_answer_changes_an_earlier_correction(tmp_path):
    """The whole reason the transcript is rebuilt from the original. The Q&A
    stage supersedes the guess that produced a marker, and that can only take
    effect if the substitution runs again from untouched text."""
    run = _ready(tmp_path, step=Step.QUESTIONS)
    assert "[?Kessler]" in " ".join(s["text"] for s in run.working()["segments"])

    run.context_map = ContextMap(likely_errors=[
        {"heard": "Ridgelane", "probably": "Ridgeline", "confidence": "high"},
        {"heard": "Kestler", "probably": "Crockett", "confidence": "high"},
    ])

    text = " ".join(s["text"] for s in run.working()["segments"])
    assert "Crockett" in text
    assert "[?Kessler]" not in text


def test_confirmed_speaker_names_survive_a_rebuild(tmp_path):
    """They are applied after the corrections, so a later question round must
    not silently return the transcript to Speaker 1 and Speaker 2."""
    run = _ready(tmp_path)
    run.confirm_speakers({"Speaker 1": "Marcus"}, library_path=str(tmp_path / "s.db"))

    assert run.working()["segments"][0]["speaker"] == "Marcus"


def test_only_confirmed_names_are_kept(tmp_path):
    run = _ready(tmp_path)

    run.confirm_speakers(
        {"Speaker 1": "Marcus", "Speaker 2": "   "}, library_path=str(tmp_path / "s.db")
    )

    assert run.speaker_names == {"Speaker 1": "Marcus"}


def test_the_change_log_records_what_happened(tmp_path):
    run = _ready(tmp_path)

    run.working()

    joined = " | ".join(run.notes)
    assert "Corrected: Ridgelane" in joined
    assert "Flagged: Kestler" in joined


# --- the voice library ---------------------------------------------------------


def test_confirming_speakers_remembers_their_voices(tmp_path):
    db = str(tmp_path / "s.db")
    run = _ready(tmp_path)

    run.confirm_speakers({"Speaker 1": "Marcus Ellery"}, library_path=db)

    with speaker_library_open(db) as store:
        assert [n for n, _ in store.names()] == ["Marcus Ellery"]


def speaker_library_open(path):
    from src.podcastnotes import speaker_library

    return speaker_library.Library(path)


def test_an_unnamed_speaker_is_filled_in_from_the_library(tmp_path):
    """Claude and the library know different things: one has read what this
    person said today, the other has heard them before."""
    from src.podcastnotes import speaker_library

    db = str(tmp_path / "s.db")
    with speaker_library.Library(db) as store:
        store.remember("Anya Petrov-Hale", [0.0, 1.0, 0.0])

    run = _ready(tmp_path)
    client = _client({"speakers": [
        {"speaker": "Speaker 1", "name": "Marcus", "confidence": "high"},
        {"speaker": "Speaker 2", "name": "", "confidence": "low"},
    ]})

    found = run.speaker_suggestions(client=client, library_path=db)

    by_label = {s.speaker: s for s in found}
    assert by_label["Speaker 1"].name == "Marcus", "Claude's answer was overwritten"
    assert by_label["Speaker 2"].name == "Anya Petrov-Hale"


def test_the_library_does_not_overrule_claude(tmp_path):
    from src.podcastnotes import speaker_library

    db = str(tmp_path / "s.db")
    with speaker_library.Library(db) as store:
        store.remember("Somebody Else", [1.0, 0.0, 0.0])

    run = _ready(tmp_path)
    client = _client({"speakers": [
        {"speaker": "Speaker 1", "name": "Marcus", "confidence": "high"},
    ]})

    found = run.speaker_suggestions(client=client, library_path=db)

    assert found[0].name == "Marcus"


# --- question rounds -----------------------------------------------------------


def test_a_round_with_nothing_to_ask_moves_on_to_writing(tmp_path):
    run = _ready(tmp_path, step=Step.QUESTIONS)
    run.context_map = ContextMap()  # no corrections, so no markers

    found = run.next_questions(client=_client({"questions": []}))

    assert found.questions == []
    assert run.step is Step.WRITE


def test_answering_advances_to_the_next_round(tmp_path):
    """What was asked is recorded when it is asked, not when it is answered,
    so this only has to move the round on. See the skip test below for why."""
    run = _ready(tmp_path, step=Step.QUESTIONS)

    run.answer({"Kessler": "It is Crockett."}, client=_client({"settled": [], "corrections": []}))

    assert run.round_number == 2


def test_the_rounds_run_out(tmp_path):
    run = _ready(tmp_path, step=Step.QUESTIONS)
    empty = {"settled": [], "corrections": []}

    for _ in range(qa.MAX_ROUNDS):
        run.answer({"Kessler": "no idea"}, client=_client(empty))

    assert run.step is Step.WRITE


def test_answering_nothing_still_advances_the_round(tmp_path):
    """Skipping every question is allowed and must not leave the run stuck."""
    run = _ready(tmp_path, step=Step.QUESTIONS)

    run.answer({})

    assert run.round_number == 2


# --- surviving a closed window -------------------------------------------------


def test_the_state_round_trips(tmp_path):
    run = _ready(tmp_path, step=Step.QUESTIONS)
    run.description = "Ashford trip"
    run.speaker_names = {"Speaker 1": "Marcus"}
    run.asked = ["Kessler"]
    run.round_number = 2
    run.save()

    again = WriteUp.resume(str(tmp_path), _original())

    assert again.step is Step.QUESTIONS
    assert again.description == "Ashford trip"
    assert again.speaker_names == {"Speaker 1": "Marcus"}
    assert again.asked == ["Kessler"]
    assert again.round_number == 2
    assert [e["probably"] for e in again.context_map.likely_errors] == ["Ridgeline", "Kessler"]


def test_resuming_a_trip_that_was_never_started_is_a_fresh_run(tmp_path):
    run = WriteUp.resume(str(tmp_path), _original())

    assert run.step is Step.CONTEXT
    assert run.context_map.is_empty


def test_a_corrupt_state_file_does_not_stop_the_trip_opening(tmp_path):
    (tmp_path / pipeline.STATE_FILENAME).write_text("{ not json")

    run = WriteUp.resume(str(tmp_path), _original())

    assert run.step is Step.CONTEXT


def test_a_step_from_a_newer_version_resumes_from_the_start(tmp_path):
    """Recomputing is right. Guessing which step it meant is not, and starting
    over from nothing would throw away the saved context map."""
    (tmp_path / pipeline.STATE_FILENAME).write_text(json.dumps({
        "step": "some_future_step",
        "context_map": {"people": [{"name": "Anya"}]},
    }))

    run = WriteUp.resume(str(tmp_path), _original())

    assert run.step is Step.CONTEXT
    assert run.context_map.people == [{"name": "Anya"}]


def test_every_step_saves_as_it_goes(tmp_path):
    """A person answers questions in the middle of this, so nothing should
    depend on them not closing the window."""
    run = _ready(tmp_path)

    run.confirm_speakers({"Speaker 1": "Marcus"}, library_path=str(tmp_path / "s.db"))

    assert WriteUp.resume(str(tmp_path), _original()).speaker_names == {"Speaker 1": "Marcus"}


def test_the_transcript_is_not_stored_in_the_state_file(tmp_path):
    """It already lives beside the audio as the segment export, and a second
    copy invites the two to disagree."""
    run = _ready(tmp_path)
    run.save()

    saved = json.loads((tmp_path / pipeline.STATE_FILENAME).read_text())

    assert "segments" not in json.dumps(saved)


# --- helpers -------------------------------------------------------------------


class _Text:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Usage:
    input_tokens = output_tokens = 0
    cache_creation_input_tokens = cache_read_input_tokens = 0


class _Message:
    def __init__(self, content):
        self.content = content
        self.stop_reason = "end_turn"
        self.usage = _Usage()
        self.stop_details = None


def _client(payload):
    class _Stream:
        def __enter__(s):
            return s

        def __exit__(s, *e):
            return False

        def get_final_message(s):
            return _Message([_Text(json.dumps(payload))])

    class _Messages:
        def stream(self, **kw):
            return _Stream()

    class _Client:
        messages = _Messages()

    return _Client()


def test_a_skipped_question_is_not_asked_again(tmp_path):
    """Found on a real run. Skipping recorded nothing, so the next round
    proposed the same markers: Simone, Marchetti and Whitlock were all asked in
    every one of the three rounds, spending the whole budget on four things."""
    run = _ready(tmp_path, step=Step.QUESTIONS)
    asking = _client({"questions": [
        {"marker": "Kessler", "ask": "Who?"},
    ]})

    first = run.next_questions(client=asking)
    run.answer({}, client=_client({"settled": [], "corrections": []}))

    assert [q.marker for q in first.questions] == ["Kessler"]
    assert run.asked == ["Kessler"]


def test_an_answered_question_is_also_recorded(tmp_path):
    run = _ready(tmp_path, step=Step.QUESTIONS)
    run.next_questions(client=_client({"questions": [{"marker": "Kessler", "ask": "Who?"}]}))

    assert run.asked == ["Kessler"]


def test_what_was_asked_survives_a_closed_window(tmp_path):
    """Otherwise reopening the trip re-asks everything already put to them."""
    run = _ready(tmp_path, step=Step.QUESTIONS)
    run.next_questions(client=_client({"questions": [{"marker": "Kessler", "ask": "Who?"}]}))

    assert WriteUp.resume(str(tmp_path), _original()).asked == ["Kessler"]


# --- correcting after the fact ---------------------------------------------------
#
# The gate before the documents existed asked for a judgement with nothing to
# judge against. This is the same work moved to where the person can see what
# is wrong, and it is a fresh Glean pass because a name they supply is a search
# key rather than just a spelling.


def _done(tmp_path):
    run = _ready(tmp_path, step=Step.DONE)
    run.transcript = "# Transcript\n\nold"
    run.summary = "# Summary\n\nold"
    return run


def _revision(monkeypatch, **fields):
    """Stand in for the Glean pass, so these tests are about the pipeline."""
    from src.podcastnotes import revise as revise_module

    seen = {}

    def fake(context_map, notes, transcript, **kw):
        seen["notes"] = notes
        seen["transcript"] = transcript
        seen["speakers"] = kw.get("speakers")
        return revise_module.Revision(**fields)

    monkeypatch.setattr(pipeline.revise, "from_notes", fake)
    return seen


def test_a_revision_can_only_happen_once_there_is_something_to_revise(tmp_path):
    run = _ready(tmp_path, step=Step.QUESTIONS)

    with pytest.raises(OutOfOrder):
        run.revise("the director is Kessler")


def test_a_revision_does_not_move_the_run_backwards(tmp_path, monkeypatch):
    """It is a second pass over a settled record, not the run re-running. If it
    moved the step, every later stage would demand to be redone."""
    _revision(monkeypatch)
    run = _done(tmp_path)

    run.revise("the director is Kessler")

    assert run.step is Step.DONE


def test_an_empty_note_spends_nothing(tmp_path, monkeypatch):
    """A full agentic Glean loop whose only possible input is nothing at all."""
    called = []
    monkeypatch.setattr(pipeline.revise, "from_notes",
                        lambda *a, **k: called.append(1))
    run = _done(tmp_path)

    found = run.revise("   ")

    assert called == []
    assert found.is_empty and run.revision == 0


def test_the_note_is_researched_against_the_raw_transcript(tmp_path, monkeypatch):
    """Corrections are re-applied from the original every time, so a row written
    against the polished document matches nothing."""
    seen = _revision(monkeypatch)
    run = _done(tmp_path)

    run.revise("the director is Kessler")

    # "Ridgelane" is a high-confidence correction, so `working()` has already
    # turned it into "Ridgeline". Sending that version would have the pass write
    # `heard: "Ridgeline"`, which matches nothing when the corrections are next
    # applied from the original.
    assert "We went to Ridgelane Northgate." in seen["transcript"]
    assert "Ridgeline" not in seen["transcript"]
    assert "# Transcript" not in seen["transcript"]
    assert seen["speakers"] == ["Speaker 1", "Speaker 2"]


def test_what_they_say_replaces_the_hedged_row_rather_than_joining_it(tmp_path, monkeypatch):
    """The single most dangerous thing about this feature. `correct._proposals`
    sorts longest-`heard` first with a *stable* sort, so an appended row for
    words an existing row already claims never fires: the old guess wins and the
    person is told their own correction was not found in the transcript."""
    _revision(monkeypatch, context_map=ContextMap(likely_errors=[
        {"heard": "Kestler", "probably": "Kraker", "confidence": "high"},
    ]))
    run = _done(tmp_path)

    run.revise("his name is Kraker, spelt with a K")

    rows = [e for e in run.context_map.likely_errors if e["heard"] == "Kestler"]
    assert len(rows) == 1
    assert "Kraker" in run.working()["segments"][1]["text"]


def test_a_question_they_answered_stops_being_open(tmp_path, monkeypatch):
    """Open questions go into the writing prompt under "do not present these as
    settled", so leaving one there makes the rewrite hedge about the exact thing
    they just settled. That is the original complaint, restated."""
    _revision(monkeypatch, settled=["Who is the director?"])
    run = _done(tmp_path)
    run.context_map.open_questions = ["Who is the director?", "Which site?"]

    run.revise("the director is Kessler")

    assert run.context_map.open_questions == ["Which site?"]


def test_naming_a_voice_is_applied_like_any_other_name(tmp_path, monkeypatch):
    """"The coordinator is Simone Vasari" is far more often an attribution than
    a spelling. `working()` re-applies names from the original every time, so
    this needs no plumbing beyond the merge."""
    _revision(monkeypatch, speakers={"Speaker 2": "Simone Vasari"})
    run = _done(tmp_path)

    run.revise("Speaker 2 is Simone Vasari")

    assert run.working()["segments"][1]["speaker"] == "Simone Vasari"


def test_what_cannot_be_done_is_kept_to_be_shown(tmp_path, monkeypatch):
    """Doing nothing is indistinguishable from ignoring them, so they retype it
    and pay for another pass."""
    _revision(monkeypatch, impossible=["Nobody mentions a Dr Okafor."])
    run = _done(tmp_path)

    run.revise("add Dr Okafor from cardiology")

    assert run.impossible == ["Nobody mentions a Dr Okafor."]


def test_a_new_impossible_list_replaces_the_last_one(tmp_path, monkeypatch):
    """Otherwise the second pass shows a complaint about the first pass's note,
    which by then has usually been dealt with."""
    _revision(monkeypatch, impossible=["first"])
    run = _done(tmp_path)
    run.revise("one")
    _revision(monkeypatch, impossible=["second"])

    run.revise("two")

    assert run.impossible == ["second"]


def test_the_documents_are_marked_stale_until_they_are_rewritten(tmp_path, monkeypatch):
    """Covers a crash between the merge and the rewrite. Without it the finished
    screen shows saved documents beside a change log computed from a newer map,
    which is a worse lie than an error."""
    _revision(monkeypatch)
    run = _done(tmp_path)

    run.revise("the director is Kessler")
    assert run.pending_write is True
    assert WriteUp.resume(str(tmp_path), _original()).pending_write is True

    run.write_documents(client=_client({"transcript": "t", "summary": "s"}))
    assert run.pending_write is False


def test_a_pass_that_died_is_not_charged_as_an_attempt(tmp_path, monkeypatch):
    """So "Try again" after a Glean outage is a repeat, not a second of five."""
    def dead(*a, **k):
        raise pipeline.context.GleanUnavailable("no route to host")

    monkeypatch.setattr(pipeline.revise, "from_notes", dead)
    run = _done(tmp_path)

    with pytest.raises(pipeline.context.GleanUnavailable):
        run.revise("the director is Kessler")

    assert run.revision == 0 and run.revisions_left == pipeline.MAX_REVISIONS


def test_the_fifth_rewrite_is_the_last(tmp_path, monkeypatch):
    """Somebody on their sixth "still wrong" has hit something this tool cannot
    fix, and saying so is more use than spending another pass."""
    _revision(monkeypatch)
    run = _done(tmp_path)
    for i in range(pipeline.MAX_REVISIONS):
        run.revise(f"note {i}")

    assert run.revisions_left == 0
    with pytest.raises(pipeline.RevisionsExhausted):
        run.revise("one more")


def test_everything_they_said_reaches_the_writing(tmp_path, monkeypatch):
    """Much of a note never becomes a substitution. "Site 3 was the busiest"
    changes no words in the transcript and everything about the summary, and
    without this it would vanish between the two."""
    _revision(monkeypatch)
    run = _done(tmp_path)
    run.revise("site 3 was the busiest")
    run.revise("and Kessler had left by then")

    sent = []
    from src.podcastnotes import output
    monkeypatch.setattr(output, "_shared", lambda d, c, s, st, rev=(): sent.append(rev) or "")
    run.write_documents(client=_client({"transcript": "t", "summary": "s"}))

    assert sent[0] == ["site 3 was the busiest", "and Kessler had left by then"]


def test_the_revisions_survive_a_closed_window(tmp_path, monkeypatch):
    _revision(monkeypatch, impossible=["no Dr Okafor"])
    run = _done(tmp_path)
    run.revise("add Dr Okafor")

    again = WriteUp.resume(str(tmp_path), _original())

    assert again.revision == 1
    assert again.revision_notes == ["add Dr Okafor"]
    assert again.impossible == ["no Dr Okafor"]


def test_a_trip_saved_before_revisions_existed_still_opens(tmp_path):
    """The whole migration: it resumes as never revised and behaves as it did."""
    run = _done(tmp_path)
    run.save()
    path = tmp_path / pipeline.STATE_FILENAME
    saved = json.loads(path.read_text())
    for key in ("revision", "revision_notes", "impossible", "pending_write"):
        saved.pop(key)
    path.write_text(json.dumps(saved))

    again = WriteUp.resume(str(tmp_path), _original())

    assert again.revision == 0 and again.revisions_left == pipeline.MAX_REVISIONS
    assert again.impossible == [] and again.pending_write is False
