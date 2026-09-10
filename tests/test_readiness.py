"""
Tests for the setup checklist.

This is the screen people see when something is already broken, so its own
failure modes matter more than most: a check that crashes, or that blames the
wrong thing, makes a bad moment worse.

Run with: python -m pytest tests/test_readiness.py -v
"""

from src.podcastnotes.readiness import Check, Readiness, State, blocked, failed, ok


def check(key, result, requires=(), title=None):
    return Check(
        key=key,
        title=title or key.title(),
        purpose=f"doing {key}",
        run=lambda: result,
        requires=list(requires),
    )


# --- running the list ---------------------------------------------------------


def test_every_check_runs_and_reports():
    readiness = Readiness([check("a", ok("fine")), check("b", failed("broken"))])

    results = readiness.run()

    assert results["a"].state is State.OK
    assert results["b"].state is State.FAILED
    assert results["b"].detail == "broken"


def test_all_ok_needs_every_check():
    assert Readiness.all_ok({"a": ok(), "b": ok()}) is True
    assert Readiness.all_ok({"a": ok(), "b": failed("no")}) is False


def test_an_empty_list_is_not_ready():
    """Vacuous truth here would mean a broken config reports everything fine."""
    assert Readiness.all_ok({}) is False


def test_checks_run_in_the_order_given():
    order = []
    readiness = Readiness([
        Check("first", "First", "p", lambda: (order.append("first"), ok())[1]),
        Check("second", "Second", "p", lambda: (order.append("second"), ok())[1]),
    ])

    readiness.run()

    assert order == ["first", "second"]


def test_only_runs_the_named_checks():
    """Re-checking one row after a fix should not redo a 3GB download check."""
    ran = []
    readiness = Readiness([
        Check("a", "A", "p", lambda: (ran.append("a"), ok())[1]),
        Check("b", "B", "p", lambda: (ran.append("b"), ok())[1]),
    ])

    readiness.run(only=["b"])

    assert ran == ["b"]


# --- not blaming the wrong thing ----------------------------------------------


def test_a_check_whose_requirement_failed_is_blocked_not_failed():
    """Telling somebody Drive is broken when they have not signed in yet
    sends them to fix the wrong thing."""
    readiness = Readiness([
        check("google", failed("not signed in")),
        check("drive", ok(), requires=["google"], title="Google Drive"),
    ])

    results = readiness.run()

    assert results["drive"].state is State.BLOCKED
    assert results["drive"].state is not State.FAILED


def test_a_blocked_check_names_what_it_is_waiting_on():
    readiness = Readiness([
        check("google", failed("nope"), title="Google sign-in"),
        check("drive", ok(), requires=["google"]),
    ])

    assert "Google sign-in" in readiness.run()["drive"].detail


def test_a_blocked_check_does_not_run_at_all():
    """Calling with a credential that is known to be missing wastes a network
    round trip and produces a second, more confusing error."""
    ran = []
    readiness = Readiness([
        check("google", failed("nope")),
        Check("drive", "Drive", "p", lambda: (ran.append("drive"), ok())[1],
              requires=["google"]),
    ])

    readiness.run()

    assert ran == [], "a blocked check should not execute"


def test_a_requirement_that_passed_lets_the_check_run():
    readiness = Readiness([
        check("google", ok()),
        check("drive", ok("created and deleted a doc"), requires=["google"]),
    ])

    assert readiness.run()["drive"].state is State.OK


def test_first_problem_skips_blocked_checks():
    """The thing to fix is the cause, never the symptom."""
    readiness = Readiness([
        check("models", ok()),
        check("google", failed("not signed in")),
        check("drive", ok(), requires=["google"]),
    ])

    results = readiness.run()

    assert results["drive"].state is State.BLOCKED
    assert Readiness.first_problem(results) == "google"


def test_first_problem_ignores_a_blocked_row_that_comes_first():
    """Built by hand rather than run, and that is the point.

    Within one full run a blocked row always follows the failure that caused
    it, so ordering never bites. The screen does not hold one full run: it
    re-checks single rows and merges, so a stale blocked row can sit above a
    newer failure. Returning it would send somebody to a symptom.
    """
    results = {
        "drive": blocked("Waiting on Google sign-in"),
        "glean": failed("token rejected"),
    }

    assert Readiness.first_problem(results) == "glean"


def test_rechecking_one_row_still_respects_what_is_already_known():
    """Otherwise the fix button on Drive calls Drive while Google is signed out."""
    ran = []
    readiness = Readiness([
        check("google", failed("signed out")),
        Check("drive", "Drive", "p", lambda: (ran.append("drive"), ok())[1],
              requires=["google"]),
    ])
    known = readiness.run()

    merged = readiness.run(only=["drive"], known=known)

    assert ran == [], "Drive was called despite Google being signed out"
    assert merged["drive"].state is State.BLOCKED
    assert merged["google"].state is State.FAILED, "the merge dropped the other rows"


def test_rechecking_keeps_the_display_order():
    """The list must not reshuffle under somebody's cursor when a row updates."""
    readiness = Readiness([check("a", ok()), check("b", failed("x")), check("c", ok())])
    known = readiness.run()

    merged = readiness.run(only=["b"], known=known)

    assert list(merged) == ["a", "b", "c"]


def test_first_problem_is_none_when_everything_works():
    readiness = Readiness([check("a", ok()), check("b", ok())])

    assert Readiness.first_problem(readiness.run()) is None


# --- reporting progress -------------------------------------------------------


def test_each_check_reports_before_and_after_it_runs():
    """Several of these are network calls. A list that sits blank for ten
    seconds and then fills in at once looks broken while it is working."""
    events = []
    readiness = Readiness([check("a", ok("fine")), check("b", failed("no"))])

    readiness.run(
        on_start=lambda key: events.append(("start", key)),
        on_result=lambda key, result: events.append(("result", key, result.state)),
    )

    assert events == [
        ("start", "a"),
        ("result", "a", State.OK),
        ("start", "b"),
        ("result", "b", State.FAILED),
    ]


def test_a_blocked_check_is_reported_without_being_started():
    """It never runs, so announcing that it is being checked would be a lie,
    but the row still has to stop saying "checking"."""
    events = []
    readiness = Readiness([
        check("google", failed("signed out")),
        check("drive", ok(), requires=["google"]),
    ])

    readiness.run(
        on_start=lambda key: events.append(("start", key)),
        on_result=lambda key, result: events.append(("result", key, result.state)),
    )

    assert ("start", "drive") not in events
    assert ("result", "drive", State.BLOCKED) in events


def test_progress_reporting_is_optional():
    readiness = Readiness([check("a", ok())])

    assert readiness.run()["a"].state is State.OK  # no callbacks, no crash


# --- a check that goes wrong --------------------------------------------------


def test_a_check_that_raises_becomes_a_failure_not_a_crash():
    """This list is what people look at when something is already wrong. It
    cannot itself be the thing that takes the app down."""
    def explode():
        raise ConnectionError("network unreachable")

    readiness = Readiness([Check("net", "Network", "p", explode)])

    result = readiness.run()["net"]

    assert result.state is State.FAILED
    assert "ConnectionError" in result.detail
    assert "network unreachable" in result.detail
    assert result.remedy, "an unexpected failure should still say what to do"


def test_one_exploding_check_does_not_stop_the_others():
    def explode():
        raise RuntimeError("boom")

    readiness = Readiness([
        Check("bad", "Bad", "p", explode),
        check("good", ok("fine")),
    ])

    results = readiness.run()

    assert results["bad"].state is State.FAILED
    assert results["good"].state is State.OK


# --- what a person is shown ---------------------------------------------------


def test_a_failure_can_carry_somewhere_to_go():
    result = failed("No token", remedy="Make one", url="https://example.com/tokens")

    assert result.url == "https://example.com/tokens"
    assert result.remedy == "Make one"


def test_purposes_are_written_for_people_not_engineers():
    """The list is read by somebody who does not know what Vertex AI is."""
    from src.podcastnotes.checks_local import LOCAL_CHECKS

    jargon = ("api", "oauth", "token endpoint", "sdk", "vertex", "pyannote", "whisper")
    for c in LOCAL_CHECKS:
        assert c.purpose, f"{c.key} has no purpose"
        assert not any(word in c.purpose.lower() for word in jargon), (
            f"{c.key} purpose is jargon: {c.purpose!r}"
        )
        assert c.purpose[0].isupper(), f"{c.key} purpose should read as a phrase"


# --- the local checks ---------------------------------------------------------


def test_ffmpeg_check_passes_on_a_working_install():
    from src.podcastnotes.checks_local import check_ffmpeg

    assert check_ffmpeg().state is State.OK


def test_ffmpeg_check_recognises_a_lost_execute_bit(tmp_path, monkeypatch):
    """The exact failure that happened during development.

    Everything downstream reported corrupt audio and zero durations; nothing
    said the binary was not runnable.
    """
    from src.podcastnotes import checks_local
    from src.utils import file_utils

    fake = tmp_path / "ffmpeg"
    fake.write_bytes(b"#!/bin/sh\nexit 0\n")
    fake.chmod(0o644)  # readable, not executable

    monkeypatch.setattr(file_utils, "get_bundled_binary", lambda name: str(fake))
    monkeypatch.setattr(file_utils, "check_ffmpeg_health", lambda: (False, "FFmpeg error"))

    result = checks_local.check_ffmpeg()

    assert result.state is State.FAILED
    assert "executable" in result.detail
    assert result.remedy


def test_huggingface_check_reports_a_missing_token(monkeypatch):
    from src.podcastnotes import checks_local

    monkeypatch.setattr("src.core.config.get_hf_token", lambda: "")

    result = checks_local.check_huggingface()

    assert result.state is State.FAILED
    assert "huggingface.co" in result.url
    assert "free" in result.remedy.lower()


def test_huggingface_check_reports_a_rejected_token(monkeypatch):
    """A saved token that has been revoked looks fine until something calls."""
    from src.podcastnotes import checks_local

    monkeypatch.setattr("src.core.config.get_hf_token", lambda: "hf_stale")
    monkeypatch.setattr(
        "src.core.diarization.validate_hf_token",
        lambda t: (False, "Invalid token - please check and re-enter"),
    )

    result = checks_local.check_huggingface()

    assert result.state is State.FAILED
    assert "Invalid token" in result.detail


def test_models_check_reports_what_is_missing(monkeypatch):
    from src.podcastnotes import checks_local

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(checks_local, "_whisper_is_cached", lambda: False)

    result = checks_local.check_models()

    assert result.state is State.FAILED
    assert "download" in result.remedy.lower()
    assert "transcription model" in result.detail
