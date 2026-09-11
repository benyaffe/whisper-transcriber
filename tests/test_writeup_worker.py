"""
Tests for the write-up worker.

The worker itself is thin. What is worth defending is that no failure reaches
a colleague as a stack trace, because every one of these happens in ordinary
use and each has a different next move. An expired Google sign-in is the most
common of them and is routine rather than alarming: the org's session policy
expires these every day or two.

The readiness tests already forbid jargon in check text, and the same standard
applies here, so that is asserted rather than assumed.

Run with: python -m pytest tests/test_writeup_worker.py -v
"""

import pytest

from src.podcastnotes.context import GleanUnavailable
from src.podcastnotes.llm import agent, client
from src.ui.podcastnotes.writeup_worker import LABELS, WriteUpWorker, explain


class _FakeWriteUp:
    """Records which step was asked for, without doing any of the work."""

    def __init__(self, raises=None):
        self.calls = []
        self.raises = raises

    def _record(self, name, result=None, **kwargs):
        self.calls.append((name, kwargs))
        if self.raises:
            raise self.raises
        return result

    def build_context(self, on_search=None):
        if on_search:
            on_search("Project Lanternfish")
        return self._record("context", "a map")

    def speaker_suggestions(self, library_path=""):
        return self._record("speakers", ["a speaker"], library_path=library_path)

    def next_questions(self):
        return self._record("questions", "a round")

    def answer(self, answers):
        return self._record("answers", None, answers=answers)

    def write_documents(self):
        return self._record("write", "documents")


def _run(worker):
    """Run the step synchronously, which is what QThread.run does anyway."""
    got = {}
    worker.completed.connect(lambda r: got.setdefault("result", r))
    worker.failed.connect(lambda m: got.setdefault("failed", m))
    worker.progress.connect(lambda m: got.setdefault("progress", []).append(m)
                            if isinstance(got.get("progress"), list)
                            else got.update(progress=[m]))
    worker.run()
    return got


# --- the steps ----------------------------------------------------------------


@pytest.mark.parametrize("step,expected", [
    ("context", "a map"),
    ("speakers", ["a speaker"]),
    ("questions", "a round"),
    ("write", "documents"),
])
def test_each_step_runs_the_matching_stage(qt_app, step, expected):
    writeup = _FakeWriteUp()

    got = _run(WriteUpWorker(writeup, step))

    assert got["result"] == expected
    assert writeup.calls[0][0] == step


def test_answers_are_passed_through(qt_app):
    writeup = _FakeWriteUp()

    _run(WriteUpWorker(writeup, "answers", answers={"Kestler": "It is Kessler."}))

    assert writeup.calls[0][1]["answers"] == {"Kestler": "It is Kessler."}


def test_the_library_path_reaches_the_speaker_step(qt_app):
    writeup = _FakeWriteUp()

    _run(WriteUpWorker(writeup, "speakers", library_path="/tmp/voices.db"))

    assert writeup.calls[0][1]["library_path"] == "/tmp/voices.db"


def test_searches_are_reported_as_they_happen(qt_app):
    """A context build takes minutes, and a bar that only moves at the end
    looks like a hang."""
    got = _run(WriteUpWorker(_FakeWriteUp(), "context"))

    assert got["progress"] == ["Looking up: Project Lanternfish"]


def test_an_unknown_step_fails_rather_than_doing_nothing(qt_app):
    got = _run(WriteUpWorker(_FakeWriteUp(), "nonsense"))

    assert "nonsense" in got["failed"]
    assert "result" not in got


def test_the_write_up_is_not_stored_under_the_name_run(qt_app):
    """QThread's own entry point is a method called run. Holding the write-up
    under that name shadows it, and the thread then starts and does nothing at
    all, silently."""
    worker = WriteUpWorker(_FakeWriteUp(), "context")

    assert callable(worker.run)
    assert worker.writeup is not None


def test_every_step_has_something_to_show_on_screen():
    for step in ("context", "speakers", "questions", "answers", "write"):
        assert LABELS[step]
    assert WriteUpWorker(_FakeWriteUp(), "context").label == LABELS["context"]


# --- failures a person can act on ---------------------------------------------


def test_a_failure_is_reported_rather_than_raised(qt_app):
    got = _run(WriteUpWorker(_FakeWriteUp(raises=RuntimeError("boom")), "context"))

    assert "boom" in got["failed"]
    assert "result" not in got


def test_glean_being_down_says_the_write_up_stopped_on_purpose():
    """It stopped rather than guessing, and the transcript is still safe. Both
    matter to somebody deciding what to do next."""
    message = explain(GleanUnavailable("no route to host"))

    assert "stopped" in message
    assert "transcript is finished" in message


def test_an_expired_sign_in_is_described_as_routine():
    """It expires every day or two because of the org's session policy, so
    alarming language here would be wrong as well as unhelpful."""
    message = explain(client.NotSignedIn("no credentials"))

    assert "expired" in message
    assert "day or two" in message


def test_claude_not_enabled_carries_the_link_for_that_project():
    message = explain(client.ClaudeNotEnabled("example-vertexai"))

    assert "example-vertexai" in message
    assert "https://" in message


def test_a_pinned_region_is_named_along_with_the_fix():
    message = explain(client.UnsupportedRegion("us-east5"))

    assert "us-east5" in message
    assert "global" in message


def test_a_refusal_says_nothing_was_published():
    message = explain(agent.Refused("cyber", "declined"))

    assert "no document" in message
    assert "published" in message


def test_an_unexpected_failure_still_produces_a_sentence():
    message = explain(ValueError("something odd"))

    assert message.startswith("The write-up could not finish")
    assert "something odd" in message


JARGON = ("traceback", "exception", "oauth", "sdk", "vertex", "api", "token endpoint",
          "json", "http", "stderr")


@pytest.mark.parametrize("error", [
    GleanUnavailable("x"),
    client.NotSignedIn("x"),
    client.NoProjectChosen("x"),
    client.ClaudeNotEnabled("proj"),
    client.UnsupportedRegion("us-east5"),
    agent.Refused("cyber", "declined"),
])
def test_no_failure_message_uses_jargon(error):
    """The same standard the readiness checks are held to. A colleague who has
    never opened a cloud console has to know what to do next.

    URLs are exempt, and only URLs. The Model Garden link genuinely contains
    the word "vertex" and has to be literal to work, so the rule applies to
    the prose a person reads rather than to a link they click.
    """
    import re

    prose = re.sub(r"https?://\S+", "", explain(error)).lower()

    for word in JARGON:
        assert word not in prose, f"{word!r} in: {prose}"


def test_the_line_moves_on_when_a_search_returns(qt_app):
    """Reporting only the start leaves the line sitting on a finished lookup
    while Claude reads what came back, which takes up to a minute at this
    effort and reads as a hang. Watched happening on a real run."""
    worker = WriteUpWorker(_FakeWriteUp(), "context")
    seen = []
    worker.progress.connect(seen.append)

    worker._searched("Devan Shaw")
    worker._searched("Devan Shaw", 16)

    assert seen[0] == "Looking up: Devan Shaw"
    assert "Reading 16 results" in seen[1]
    assert seen[0] != seen[1], "the line did not change when the search returned"


def test_one_result_is_not_reported_as_results(qt_app):
    worker = WriteUpWorker(_FakeWriteUp(), "context")
    seen = []
    worker.progress.connect(seen.append)

    worker._searched("Anya", 1)

    assert "Reading 1 result for" in seen[0]


def test_each_step_has_a_short_name_for_the_heading():
    """"Writing this up" stood on the heading from the first Glean search to
    the finished pair: a quarter of an hour of a screen that never changes,
    and a "this" nobody can point at."""
    from src.ui.podcastnotes.writeup_worker import PHASES

    for step in ("context", "speakers", "questions", "answers", "write"):
        assert PHASES[step]
        assert len(PHASES[step].split()) <= 2, f"{step} is not one or two words"


def test_the_phase_is_not_the_same_for_every_step():
    from src.ui.podcastnotes.writeup_worker import PHASES

    assert len(set(PHASES.values())) >= 4


def test_answering_is_still_the_questions_phase():
    """Folding answers in is not a new place to be; it is the same step of the
    job from the person's side."""
    from src.ui.podcastnotes.writeup_worker import PHASES

    assert PHASES["answers"] == PHASES["questions"]


def test_the_worker_offers_its_phase(qt_app):
    worker = WriteUpWorker(_FakeWriteUp(), "context")

    assert worker.phase == "Background"


# --- estimating ------------------------------------------------------------------


def test_every_step_has_a_measured_typical_length():
    from src.ui.podcastnotes.writeup_worker import TYPICAL_SECONDS

    for step in ("context", "speakers", "questions", "answers", "write"):
        assert TYPICAL_SECONDS[step] > 0


def test_nothing_has_run_means_nothing_to_estimate(qt_app):
    worker = WriteUpWorker(_FakeWriteUp(), "context")

    assert worker.elapsed_fraction() == 0.0


def test_the_estimate_never_claims_to_be_finished(qt_app):
    """Reaching 100% and then continuing turns an estimate that was merely
    wrong into one that is visibly lying."""
    import time

    worker = WriteUpWorker(_FakeWriteUp(), "speakers")
    worker._started = time.monotonic() - worker.typical_seconds * 10

    assert worker.elapsed_fraction() < 1.0


def test_the_estimate_grows_with_time(qt_app):
    import time

    worker = WriteUpWorker(_FakeWriteUp(), "context")
    worker._started = time.monotonic() - worker.typical_seconds / 2

    assert 0.4 < worker.elapsed_fraction() < 0.6


def test_a_search_that_returns_is_reported_as_a_finding(qt_app):
    worker = WriteUpWorker(_FakeWriteUp(), "context")
    seen = []
    worker.found.connect(seen.append)

    worker._searched("Miles Nadeau")
    worker._searched("Miles Nadeau", 12)

    assert seen == ["Searched for Miles Nadeau"], "one finding per completed search"
