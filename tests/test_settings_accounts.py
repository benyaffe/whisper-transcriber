"""
Tests for the account fields in Settings.

The interesting one is the project picker. It shows a friendly label and
carries the project id underneath, so reading the wrong one saves a display
string as somebody's billing project and every Claude call then fails with
what looks like a permissions error.

Run with: python -m pytest tests/test_settings_accounts.py -v
"""

import pytest


@pytest.fixture
def fake_keychain(monkeypatch):
    """A dict standing in for the Keychain.

    Without this these tests read and write the developer's real credentials,
    and a failure mid-test leaves one of them clobbered.
    """
    store = {}
    import keyring

    monkeypatch.setattr(keyring, "get_password", lambda s, k: store.get((s, k)))
    monkeypatch.setattr(
        keyring, "set_password", lambda s, k, v: store.__setitem__((s, k), v)
    )
    monkeypatch.setattr(keyring, "delete_password", lambda s, k: store.pop((s, k), None))
    return store


@pytest.fixture
def dialog(qt_app, scoped_settings, fake_keychain):
    from src.ui.settings_dialog import SettingsDialog

    view = SettingsDialog()
    yield view
    view.deleteLater()


# --- the project picker -------------------------------------------------------


def test_picking_from_the_list_saves_the_id_not_the_label(dialog):
    """The failure this guards: saving "My Team (my-team-123)" as the project."""
    dialog._on_projects_found(
        [{"id": "my-team-123", "name": "My Team"}], ""
    )
    dialog.project_combo.setCurrentIndex(0)

    assert dialog.project_combo.currentText() == "My Team (my-team-123)"
    assert dialog._chosen_project() == "my-team-123"


def test_a_typed_project_id_is_taken_as_written(dialog):
    dialog.project_combo.setCurrentText("typed-project-9")

    assert dialog._chosen_project() == "typed-project-9"


def test_a_project_whose_name_is_its_id_is_not_shown_twice(dialog):
    dialog._on_projects_found([{"id": "acme-vertex", "name": "acme-vertex"}], "")

    assert dialog.project_combo.itemText(0) == "acme-vertex"


def test_the_saved_project_stays_selected_when_the_list_arrives(dialog):
    """Otherwise clicking Find my projects silently changes what you are billed
    to, which is the one thing this field must never do quietly.

    The selection is asserted, not just the value. Qt happens to keep the typed
    text across clear() while a placeholder is set, so checking the value alone
    passes even with the re-selection deleted; the guarantee would then rest on
    a placeholder that somebody could remove for unrelated reasons. Requiring
    the row to be highlighted is what was meant anyway.
    """
    dialog.project_combo.setCurrentText("my-team-123")

    dialog._on_projects_found(
        [
            {"id": "other-project", "name": "Other"},
            {"id": "my-team-123", "name": "My Team"},
        ],
        "",
    )

    index = dialog.project_combo.currentIndex()
    assert index >= 0, "the saved project is not selected in the list"
    assert dialog.project_combo.itemData(index) == "my-team-123"
    assert dialog._chosen_project() == "my-team-123"


def test_an_account_with_no_projects_says_who_to_ask(dialog):
    dialog._on_projects_found([], "")

    assert "ask" in dialog.project_status.text().lower()


def test_a_failed_lookup_is_reported_and_can_be_retried(dialog, monkeypatch):
    """Going through the real click, so the button is genuinely disabled first.

    Asserting isEnabled on a button nothing ever disabled proves nothing.
    """
    from src.podcastnotes import auth

    monkeypatch.setattr(auth, "is_configured", lambda: True)
    monkeypatch.setattr(auth, "stored_credentials", lambda: object())
    monkeypatch.setattr(auth, "list_projects", lambda c: (_ for _ in ()).throw(
        RuntimeError("quota exceeded")))

    dialog._find_projects()
    assert dialog.find_projects_btn.isEnabled() is False, "nothing was disabled"
    dialog._projects_worker.wait(5000)
    dialog._on_projects_found([], "quota exceeded")

    assert "quota exceeded" in dialog.project_status.text()
    assert dialog.find_projects_btn.isEnabled(), "no way to try again"


def test_finding_projects_signed_out_says_to_sign_in_first(dialog, monkeypatch):
    from src.podcastnotes import auth

    monkeypatch.setattr(auth, "is_configured", lambda: True)
    monkeypatch.setattr(auth, "stored_credentials", lambda: None)

    dialog._find_projects()

    assert "sign in" in dialog.project_status.text().lower()


# --- saving -------------------------------------------------------------------


def test_the_account_fields_round_trip(dialog, scoped_settings, fake_keychain):
    from src.podcastnotes import auth, glean

    dialog.project_combo.setCurrentText("team-alpha")
    dialog.glean_instance_input.setText("https://acme.glean.com/search")
    dialog.glean_token_input.setText("  glean-token-1  ")

    dialog._save_account_settings()

    assert auth.project_id() == "team-alpha"
    assert glean.instance() == "acme", "the pasted URL should be reduced to the instance"
    assert glean.get_token() == "glean-token-1", "whitespace should be stripped"


def test_opening_settings_shows_what_is_already_saved(
    qt_app, scoped_settings, fake_keychain
):
    from src.podcastnotes import auth, glean
    from src.ui.settings_dialog import SettingsDialog

    auth.set_project_id("saved-project")
    glean.set_instance("acme")
    glean.save_token("saved-token")

    view = SettingsDialog()
    try:
        assert view._chosen_project() == "saved-project"
        assert view.glean_instance_input.text() == "acme"
        assert view.glean_token_input.text() == "saved-token"
    finally:
        view.deleteLater()


def test_the_glean_token_is_not_shown_on_screen(dialog):
    from PyQt6.QtWidgets import QLineEdit

    assert dialog.glean_token_input.echoMode() == QLineEdit.EchoMode.Password


def test_clearing_the_glean_token_removes_it(dialog, fake_keychain):
    from src.podcastnotes import glean

    glean.save_token("to-be-removed")
    dialog.glean_token_input.setText("")

    dialog._save_account_settings()

    assert glean.get_token() == ""


# --- the two screens have to agree about naming speakers ------------------------


def test_naming_speakers_is_on_by_default(scoped_settings):
    """Found by a cold-start rehearsal. With this False, the setup checklist
    read "Off, so transcripts will not name speakers" and went green, and the
    same person then hit "Naming speakers needs a HuggingFace token" on the
    trip screen and could not start. A checklist that says everything is fine
    and then a screen that refuses is the failure the checklist exists to
    prevent."""
    from src.core.config import is_speaker_id_enabled

    assert is_speaker_id_enabled() is True


def test_the_intake_screen_agrees_with_the_stored_default(qt_app, scoped_settings):
    """These are two separate defaults in two files and they disagreed. The
    checklist reads one and the trip screen shows the other."""
    from src.core.config import is_speaker_id_enabled
    from src.ui.podcastnotes.intake_view import IntakeView

    view = IntakeView()
    try:
        assert view.speakers_checkbox.isChecked() == is_speaker_id_enabled()
    finally:
        view.close()


def test_a_cold_machine_is_told_the_token_is_missing(scoped_settings, monkeypatch):
    """Rather than being told it is fine because the feature is off."""
    from src.core.config import KEYRING_SERVICE
    from src.podcastnotes.checks_local import check_huggingface

    monkeypatch.setattr("src.core.config.get_hf_token", lambda: "")

    result = check_huggingface()

    assert not result.ok
    assert "not set up" in result.detail.lower()
    assert result.skip_action, "there has to be a way past it; it is optional"
