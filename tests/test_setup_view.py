"""
Tests for the setup checklist screen.

This is the screen somebody sees when the app is already not working, so what
it tells them matters more than usual. The tests are mostly about that: which
row it sends them to, which buttons it offers, and whether it ever takes the
screen away from somebody who is in the middle of something.

The checks themselves are stubbed. Real ones would need real accounts, and
what is under test here is the screen, not the connections.

Run with: python -m pytest tests/test_setup_view.py -v
"""

import pytest

from src.podcastnotes.readiness import Check, Readiness, State, failed, ok
from src.ui.podcastnotes.setup_view import CheckRow, SetupView


def stub(key, result, requires=(), title=None, fixable=True):
    return Check(
        key=key,
        title=title or key.title(),
        purpose=f"Doing {key}",
        run=lambda: result,
        requires=list(requires),
        fixable=fixable,
    )


def build(qt_app, checks):
    """A view whose checks have run, without touching a thread.

    ChecksWorker is a QThread; driving it from a test means an event loop and
    a timeout, and the thing being tested is what the screen does with the
    results rather than how they arrive.
    """
    view = SetupView(checks=checks)
    results = view.readiness.run()
    view._on_all_done(results)
    return view


# --- what it tells somebody to do ---------------------------------------------


def test_it_names_the_one_thing_to_fix(qt_app):
    view = build(qt_app, [
        stub("ffmpeg", ok()),
        stub("google", failed("not signed in"), title="Google account"),
    ])
    try:
        assert "Google account" in view.summary.text()
        assert "One thing" in view.summary.text()
    finally:
        view.deleteLater()


def test_it_sends_them_to_the_cause_not_the_symptom(qt_app):
    """Drive is blocked because Google is signed out. Naming Drive would send
    somebody to fix something that is not broken."""
    view = build(qt_app, [
        stub("google", failed("signed out"), title="Google account"),
        stub("drive", ok(), requires=["google"], title="Google Docs"),
        stub("glean", failed("no token"), title="Glean"),
    ])
    try:
        assert "Google account" in view.summary.text()
        assert "Google Docs" not in view.summary.text()
        assert view.rows["drive"].result.state is State.BLOCKED
    finally:
        view.deleteLater()


def test_a_stale_blocked_row_above_a_failure_is_still_skipped(qt_app):
    """Built by hand, because a single run cannot produce this ordering.

    The screen re-checks one row at a time and merges, so a blocked row left
    over from an earlier run can sit above a newer failure. Naming it would
    send somebody to a symptom.
    """
    from src.podcastnotes.readiness import blocked

    view = SetupView(checks=[
        stub("drive", ok(), title="Google Docs"),
        stub("glean", ok(), title="Glean"),
    ])
    try:
        view._on_all_done({
            "drive": blocked("Waiting on Google account"),
            "glean": failed("token rejected"),
        })

        assert "Glean" in view.summary.text()
        assert "Google Docs" not in view.summary.text()
    finally:
        view.deleteLater()


def test_several_failures_say_the_others_may_clear_up(qt_app):
    view = build(qt_app, [
        stub("a", failed("x"), title="First"),
        stub("b", failed("y"), title="Second"),
    ])
    try:
        assert "2 things" in view.summary.text()
        assert "clear up" in view.summary.text()
    finally:
        view.deleteLater()


def test_everything_green_says_so_plainly(qt_app):
    view = build(qt_app, [stub("a", ok()), stub("b", ok())])
    try:
        assert view.summary.text() == "Everything is working."
    finally:
        view.deleteLater()


# --- the way out --------------------------------------------------------------


def test_continue_is_locked_until_everything_works(qt_app):
    view = build(qt_app, [stub("a", ok()), stub("b", failed("no"))])
    try:
        assert view.continue_button.isEnabled() is False
    finally:
        view.deleteLater()


def test_continue_opens_once_everything_works(qt_app):
    view = build(qt_app, [stub("a", ok()), stub("b", ok())])
    try:
        assert view.continue_button.isEnabled() is True
    finally:
        view.deleteLater()


def test_finishing_reports_whether_it_is_all_working(qt_app):
    """The window uses this to decide whether to take the screen."""
    seen = []
    view = SetupView(checks=[stub("a", failed("no"))])
    view.all_done.connect(seen.append)
    try:
        view._on_all_done(view.readiness.run())

        assert seen == [False]
    finally:
        view.deleteLater()


# --- buttons that do something ------------------------------------------------


def test_no_fix_button_on_something_nobody_can_fix(qt_app):
    """Models download by themselves. A button offering to help would lie."""
    view = build(qt_app, [stub("models", failed("downloading"), fixable=False)])
    try:
        assert view.rows["models"].fix_button.isHidden()
    finally:
        view.deleteLater()


def test_no_fix_button_on_a_row_that_is_only_waiting(qt_app):
    view = build(qt_app, [
        stub("google", failed("signed out")),
        stub("drive", ok(), requires=["google"]),
    ])
    try:
        assert view.rows["drive"].fix_button.isHidden()
    finally:
        view.deleteLater()


def test_no_fix_button_when_this_failure_is_somebody_elses_to_fix(qt_app):
    """The check is normally fixable by pressing a button. This failure is not.

    Until an administrator creates the OAuth client there is no sign-in to
    start, so offering Fix would send every colleague to click something that
    cannot work.
    """
    view = build(qt_app, [
        stub("google", failed("not set up for your organisation", fixable=False)),
    ])
    try:
        assert view.rows["google"].check.fixable is True, "the check itself is fixable"
        assert view.rows["google"].fix_button.isHidden(), "but this failure is not"
    finally:
        view.deleteLater()


def test_a_fix_button_appears_on_something_actionable(qt_app):
    view = build(qt_app, [stub("google", failed("signed out"))])
    try:
        assert not view.rows["google"].fix_button.isHidden()
    finally:
        view.deleteLater()


def test_the_fix_button_goes_away_once_it_works(qt_app):
    view = build(qt_app, [stub("google", failed("signed out"))])
    try:
        view.rows["google"].show_result(ok("Signed in"))

        assert view.rows["google"].fix_button.isHidden()
    finally:
        view.deleteLater()


def test_a_link_is_offered_only_when_there_is_one(qt_app):
    view = build(qt_app, [
        stub("claude", failed("not enabled", url="https://console.example/enable")),
        stub("ffmpeg", failed("broken")),
    ])
    try:
        assert not view.rows["claude"].link_button.isHidden()
        assert view.rows["ffmpeg"].link_button.isHidden()
    finally:
        view.deleteLater()


def test_the_remedy_is_shown_next_to_the_failure(qt_app):
    """A failure with no visible instruction is just a dead end."""
    view = build(qt_app, [
        stub("hf", failed("No token saved.", remedy="Paste one in Settings.")),
    ])
    try:
        text = view.rows["hf"].detail.text()

        assert "No token saved." in text
        assert "Paste one in Settings." in text
    finally:
        view.deleteLater()


def test_a_remedy_is_not_shown_on_a_row_that_passed(qt_app):
    """A green row with an instruction under it reads as still broken."""
    from src.podcastnotes.readiness import Result

    row = CheckRow(stub("a", ok()))
    try:
        row.show_result(
            Result(State.OK, detail="Signed in as someone", remedy="Do a thing")
        )

        assert "Signed in as someone" in row.detail.text()
        assert "Do a thing" not in row.detail.text()
    finally:
        row.deleteLater()


def test_fixing_google_starts_a_sign_in_rather_than_opening_settings(qt_app, monkeypatch):
    """There is nothing to paste. Sending somebody to Settings to look for a
    field that does not exist is the worst kind of dead end.

    The real handler starts a thread that opens a browser, so it is stubbed.
    """
    view = build(qt_app, [stub("google", failed("signed out"))])
    started, settings = [], []
    monkeypatch.setattr(view, "_sign_in", lambda which: started.append(which))
    view.settings_requested.connect(lambda: settings.append(True))
    try:
        view._fix("google")

        assert started == ["google"]
        assert settings == []
    finally:
        view.deleteLater()


def test_fixing_glean_signs_in_when_the_address_is_known(qt_app, monkeypatch):
    """Glean lets applications register themselves, so there is nothing to
    paste and nobody to ask. A browser sign-in beats a token every time."""
    from src.podcastnotes import glean

    monkeypatch.setattr(glean, "instance", lambda: "acme")
    view = build(qt_app, [stub("glean", failed("not signed in"))])
    started, settings = [], []
    monkeypatch.setattr(view, "_sign_in", lambda which: started.append(which))
    view.settings_requested.connect(lambda: settings.append(True))
    try:
        view._fix("glean")

        assert started == ["glean"]
        assert settings == []
    finally:
        view.deleteLater()


def test_fixing_glean_asks_for_the_address_first(qt_app, monkeypatch):
    """There is nowhere to sign in to until we know which Glean it is."""
    from src.podcastnotes import glean

    monkeypatch.setattr(glean, "instance", lambda: "")
    view = build(qt_app, [stub("glean", failed("no address"))])
    started, settings = [], []
    monkeypatch.setattr(view, "_sign_in", lambda which: started.append(which))
    view.settings_requested.connect(lambda: settings.append(True))
    try:
        view._fix("glean")

        assert started == []
        assert settings == [True]
    finally:
        view.deleteLater()


def test_fixing_a_pasted_value_opens_settings(qt_app):
    """Pasted values live in one place rather than several bespoke dialogs.

    HuggingFace rather than Glean, because Glean now signs in through a
    browser and is no longer an example of this.
    """
    view = build(qt_app, [stub("huggingface", failed("no token"))])
    asked = []
    view.settings_requested.connect(lambda: asked.append(True))
    try:
        view._fix("huggingface")

        assert asked == [True]
    finally:
        view.deleteLater()


# --- rows in progress ---------------------------------------------------------


def test_a_row_says_it_is_checking(qt_app):
    """Several of these are network calls. Silence reads as broken."""
    row = CheckRow(stub("glean", ok()))
    try:
        row.show_checking()

        assert row.result.state is State.CHECKING
        assert "Checking" in row.detail.text()
    finally:
        row.deleteLater()


def test_a_row_starts_out_saying_nothing_has_been_checked(qt_app):
    row = CheckRow(stub("glean", ok()))
    try:
        assert row.result.state is State.UNKNOWN
        assert "Not checked" in row.detail.text()
    finally:
        row.deleteLater()


def test_every_state_has_a_mark_and_a_colour():
    """A missing entry is a KeyError on the screen whose job is not crashing."""
    from src.ui.podcastnotes.setup_view import COLOURS, MARKS

    for state in State:
        assert state in MARKS
        assert state in COLOURS


def test_the_real_list_is_seven_rows_not_five():
    """The plan said five. Seven is the honest number.

    Google splits into three because they are three different failures with
    three different remedies: signed out, Claude never enabled in Model
    Garden, and Drive refusing to create a document. Collapsing them into one
    "Google" row would mean a red tick that cannot say which of the three it
    is, on the screen whose only job is saying exactly that.
    """
    from src.ui.podcastnotes.setup_view import ALL_CHECKS

    keys = [c.key for c in ALL_CHECKS]

    assert keys == ["ffmpeg", "models", "huggingface", "google", "claude", "drive", "glean"]


def test_the_fast_local_checks_come_first():
    """Three ticks straight away says the app itself is fine and the rest is
    plumbing, which is a much better thing to see than a blank list."""
    from src.ui.podcastnotes.setup_view import ALL_CHECKS

    network = [c.key for c in ALL_CHECKS if c.key in ("google", "claude", "drive", "glean")]
    local = [c.key for c in ALL_CHECKS if c.key in ("ffmpeg", "models")]
    positions = {c.key: i for i, c in enumerate(ALL_CHECKS)}

    assert max(positions[k] for k in local) < min(positions[k] for k in network)


# --- when the checklist may take the screen -----------------------------------


def make_window(qt_app):
    from src.ui.main_window import MainWindow

    return MainWindow()


def test_a_machine_that_never_worked_opens_on_the_checklist(qt_app, scoped_settings, monkeypatch):
    from src.ui import main_window as mod

    monkeypatch.setattr(mod, "was_working_last_time", lambda: False)
    window = make_window(qt_app)
    started = []
    monkeypatch.setattr(window.setup_view, "start_checks", lambda *a, **k: started.append(True))
    try:
        window.check_setup_on_launch()

        assert window.stack.currentWidget() is window.setup_view
        assert started == [True]
    finally:
        window.close()


def test_a_machine_that_worked_last_time_opens_on_the_trip_form(qt_app, scoped_settings, monkeypatch):
    """Nobody should have to dismiss a checklist every morning."""
    from src.ui import main_window as mod

    monkeypatch.setattr(mod, "was_working_last_time", lambda: True)
    window = make_window(qt_app)
    monkeypatch.setattr(window.setup_view, "start_checks", lambda *a, **k: None)
    try:
        window.check_setup_on_launch()

        assert window.stack.currentWidget() is window.intake
    finally:
        window.close()


def test_a_clean_bill_of_health_leaves_the_screen_alone(qt_app, scoped_settings, monkeypatch):
    """The background check must be invisible when there is nothing to say."""
    from src.ui import main_window as mod

    monkeypatch.setattr(mod, "was_working_last_time", lambda: True)
    window = make_window(qt_app)
    monkeypatch.setattr(window.setup_view, "start_checks", lambda *a, **k: None)
    try:
        window.check_setup_on_launch()

        window._on_launch_checks_done(True)

        assert window.stack.currentWidget() is window.intake
    finally:
        window.close()


def test_something_broken_since_last_time_pulls_up_the_checklist(qt_app, scoped_settings, monkeypatch):
    from src.ui import main_window as mod

    monkeypatch.setattr(mod, "was_working_last_time", lambda: True)
    window = make_window(qt_app)
    monkeypatch.setattr(window.setup_view, "start_checks", lambda *a, **k: None)
    try:
        window.check_setup_on_launch()
        window._on_launch_checks_done(False)

        assert window.stack.currentWidget() is window.setup_view
    finally:
        window.close()


def test_it_does_not_interrupt_a_half_written_trip(qt_app, scoped_settings, monkeypatch):
    """Losing what somebody typed to show them a checklist would be its own
    small betrayal."""
    from src.ui import main_window as mod

    monkeypatch.setattr(mod, "was_working_last_time", lambda: True)
    window = make_window(qt_app)
    monkeypatch.setattr(window.setup_view, "start_checks", lambda *a, **k: None)
    try:
        window.check_setup_on_launch()
        window.intake.name_input.setText("Ashford hospital tour")

        window._on_launch_checks_done(False)

        assert window.stack.currentWidget() is window.intake
    finally:
        window.close()


def test_the_result_is_remembered_for_next_launch(qt_app, scoped_settings, monkeypatch):
    from src.podcastnotes.readiness import was_working_last_time
    from src.ui import main_window as mod

    monkeypatch.setattr(mod, "was_working_last_time", lambda: True)
    window = make_window(qt_app)
    monkeypatch.setattr(window.setup_view, "start_checks", lambda *a, **k: None)
    try:
        window._on_launch_checks_done(True)
        assert was_working_last_time() is True

        window._on_launch_checks_done(False)
        assert was_working_last_time() is False
    finally:
        window.close()


def test_the_remembered_flag_is_never_used_to_skip_a_check(qt_app, scoped_settings):
    """It decides which screen opens, nothing else. A credential that worked
    yesterday tells you nothing about today, which is the premise of the whole
    module."""
    import ast
    import inspect

    from src.podcastnotes import readiness

    source = inspect.getsource(readiness.Readiness)

    assert "was_working_last_time" not in source
    assert readiness.SETTINGS_LAST_ALL_OK not in source


# --- an empty form ------------------------------------------------------------


def test_an_untouched_trip_form_counts_as_empty(qt_app):
    from src.ui.podcastnotes.intake_view import IntakeView

    view = IntakeView()
    try:
        assert view.has_input() is False
    finally:
        view.deleteLater()


@pytest.mark.parametrize("field", ["name", "description", "recording"])
def test_anything_typed_counts_as_input(qt_app, field, tmp_path):
    from src.ui.podcastnotes.intake_view import IntakeView

    view = IntakeView()
    try:
        if field == "name":
            view.name_input.setText("Ashford")
        elif field == "description":
            view.description_input.setPlainText("We met the team")
        else:
            view.recordings.add(str(tmp_path / "a.m4a"))

        assert view.has_input() is True
    finally:
        view.deleteLater()
