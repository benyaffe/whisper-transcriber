"""
Tests for the Google sign-in and the Vertex client.

None of these touch the network. What is under test is the decision-making
around the calls: which model string is sent, which project, what a person is
told when it fails, and whether the "not set up yet" case is distinguished from
the "you are signed out" case. Those are different problems with different
owners, and conflating them sends somebody to retry something that cannot work.

Run with: python -m pytest tests/test_auth.py -v
"""

import json

import pytest

from src.podcastnotes import auth
from src.podcastnotes.llm import client as llm
from src.podcastnotes.readiness import State


@pytest.fixture(autouse=True)
def no_ambient_client(monkeypatch):
    """Stop a developer's own env vars leaking into the tests."""
    monkeypatch.delenv(auth.CLIENT_ID_ENV, raising=False)
    monkeypatch.delenv(auth.CLIENT_SECRET_ENV, raising=False)
    monkeypatch.delenv(auth.CLIENT_FILE_ENV, raising=False)


# --- the OAuth client, which an admin has to create ---------------------------


def test_no_oauth_client_is_not_the_same_as_being_signed_out(monkeypatch):
    """The distinction the whole rollout depends on.

    Nobody can sign in until an administrator creates the client once, for
    everybody. Reporting that as "sign in failed" sends every colleague to
    click a button that cannot work.
    """
    monkeypatch.setattr(auth, "_bundled_client_path", lambda: "")

    with pytest.raises(auth.NotConfigured) as caught:
        auth.client_config()
    assert auth.is_configured() is False

    # And it has to say which of the two it is. "Not set up yet" is an admin's
    # job; "could not be read" is a corrupted install. Same exception, opposite
    # remedies, so the message is the only thing distinguishing them.
    message = str(caught.value).lower()
    assert "set up" in message
    assert "could not be read" not in message, "this reads as a broken file"
    assert "errno" not in message


def test_env_vars_supply_a_client_in_development(monkeypatch):
    monkeypatch.setenv(auth.CLIENT_ID_ENV, "id-123.apps.googleusercontent.com")
    monkeypatch.setenv(auth.CLIENT_SECRET_ENV, "secret-abc")

    config = auth.client_config()

    assert config["client_id"] == "id-123.apps.googleusercontent.com"
    assert config["client_secret"] == "secret-abc"


def test_a_client_file_is_read_whichever_shape_google_used(tmp_path, monkeypatch):
    """Google wraps these in "installed" or "web" depending on the client type."""
    for wrapper in ("installed", "web"):
        path = tmp_path / f"{wrapper}.json"
        path.write_text(json.dumps({wrapper: {"client_id": "x", "client_secret": "y"}}))
        monkeypatch.setenv(auth.CLIENT_FILE_ENV, str(path))

        assert auth.client_config() == {"client_id": "x", "client_secret": "y"}


def test_a_truncated_client_file_is_reported_as_not_configured(tmp_path, monkeypatch):
    path = tmp_path / "broken.json"
    path.write_text(json.dumps({"installed": {"client_id": "x"}}))  # no secret
    monkeypatch.setenv(auth.CLIENT_FILE_ENV, str(path))

    with pytest.raises(auth.NotConfigured):
        auth.client_config()


def test_unreadable_json_does_not_escape_as_a_json_error(tmp_path, monkeypatch):
    path = tmp_path / "garbage.json"
    path.write_text("not json at all")
    monkeypatch.setenv(auth.CLIENT_FILE_ENV, str(path))

    with pytest.raises(auth.NotConfigured):
        auth.client_config()


# --- credentials --------------------------------------------------------------


def test_signed_out_means_no_credentials(monkeypatch):
    monkeypatch.setenv(auth.CLIENT_ID_ENV, "id")
    monkeypatch.setenv(auth.CLIENT_SECRET_ENV, "secret")
    monkeypatch.setattr(auth, "get_refresh_token", lambda: "")

    assert auth.stored_credentials() is None


def test_credentials_carry_both_scopes(monkeypatch):
    """One consent screen has to cover calling Claude and publishing the doc.

    If these ever diverge, publishing fails after the expensive part has
    already run.
    """
    monkeypatch.setenv(auth.CLIENT_ID_ENV, "id")
    monkeypatch.setenv(auth.CLIENT_SECRET_ENV, "secret")
    monkeypatch.setattr(auth, "get_refresh_token", lambda: "refresh-xyz")

    credentials = auth.stored_credentials()

    assert "https://www.googleapis.com/auth/cloud-platform" in credentials.scopes
    assert "https://www.googleapis.com/auth/drive.file" in credentials.scopes


def test_no_access_token_is_cached(monkeypatch):
    """It would be stale within the hour, and both clients refresh on demand."""
    monkeypatch.setenv(auth.CLIENT_ID_ENV, "id")
    monkeypatch.setenv(auth.CLIENT_SECRET_ENV, "secret")
    monkeypatch.setattr(auth, "get_refresh_token", lambda: "refresh-xyz")

    assert auth.stored_credentials().token is None


def test_the_refresh_token_shares_the_existing_keychain_service():
    """One entry in Keychain Access to find, and one place to revoke."""
    from src.core.config import KEYRING_SERVICE

    assert auth.KEYRING_SERVICE == KEYRING_SERVICE == "WhisperTranscriber"


# --- model and endpoint choices -----------------------------------------------


def test_vertex_model_ids_follow_the_vertex_convention():
    """A wrong model string 404s, and a 404 here is reported as "Claude is not
    enabled in Model Garden". Getting this wrong tells people a lie and sends
    them to a console page that will not help.

    Models from the 4.6 generation on carry no date. Older ones do, and on
    Vertex the separator is "@" where the Claude API uses "-".
    """
    assert llm.MODEL_BEST == "claude-opus-5"
    assert "@" not in llm.MODEL_BEST, "4.6-generation models take no date suffix"

    assert llm.MODEL_FAST.startswith("claude-haiku-4-5")
    assert "@" in llm.MODEL_FAST, "pre-4.6 models need a dated snapshot on Vertex"
    assert "-2025" not in llm.MODEL_FAST, "that is the Claude API form, not Vertex"


def test_the_default_endpoint_is_global(monkeypatch):
    """Cheaper, fails over on capacity, and does not care which region a
    colleague's project happens to live in."""
    assert llm.DEFAULT_REGION == "global"


def test_a_region_can_be_pinned_for_data_residency(monkeypatch):
    """Anybody who does need residency has to be able to say so."""
    monkeypatch.setattr(
        "src.core.config._settings",
        lambda: _FakeSettings({llm.SETTINGS_VERTEX_REGION: "europe-west1"}),
    )

    assert llm.configured_region() == "europe-west1"


def test_an_empty_saved_region_falls_back_to_global(monkeypatch):
    """QSettings returns "" rather than the default for a key written empty."""
    monkeypatch.setattr(
        "src.core.config._settings",
        lambda: _FakeSettings({llm.SETTINGS_VERTEX_REGION: ""}),
    )

    assert llm.configured_region() == "global"


class _FakeSettings:
    """Enough of QSettings for the region lookup."""

    def __init__(self, values):
        self._values = values

    def value(self, key, default=None, type=None):
        return self._values.get(key, default)


# --- the probe's error mapping ------------------------------------------------
#
# The checks above stub probe() out, so the mapping from an API error to
# something a person can act on is only exercised here.


def _status_error(cls, message, status):
    import httpx2

    return cls(
        message,
        response=httpx2.Response(
            status, request=httpx2.Request("POST", "https://example.com")
        ),
        body=None,
    )


def _probe_raising(monkeypatch, error):
    class FakeMessages:
        def create(self, **kwargs):
            raise error

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr(llm, "build_client", lambda project="", region="": FakeClient())


def test_a_404_means_claude_was_never_enabled_here(monkeypatch):
    """The model plainly exists, so a 404 is Model Garden, not a typo."""
    import anthropic

    _probe_raising(
        monkeypatch,
        _status_error(anthropic.NotFoundError, "Publisher Model not found", 404),
    )

    with pytest.raises(llm.ClaudeNotEnabled) as caught:
        llm.probe(project="proj-x")

    assert caught.value.project == "proj-x"
    assert "proj-x" in caught.value.url


def test_a_403_about_a_model_is_also_model_garden(monkeypatch):
    import anthropic

    _probe_raising(
        monkeypatch,
        _status_error(
            anthropic.PermissionDeniedError,
            "Permission denied on publisher model",
            403,
        ),
    )

    with pytest.raises(llm.ClaudeNotEnabled):
        llm.probe(project="proj-x")


def test_a_403_about_anything_else_is_not_dressed_up_as_model_garden(monkeypatch):
    """The narrow rule. Sending somebody to enable a model they already enabled,
    when the real problem is that their account lacks the Vertex User role,
    wastes their afternoon."""
    import anthropic

    _probe_raising(
        monkeypatch,
        _status_error(
            anthropic.PermissionDeniedError,
            "Caller does not have required permission aiplatform.endpoints.predict",
            403,
        ),
    )

    with pytest.raises(anthropic.PermissionDeniedError):
        llm.probe(project="proj-x")


def test_a_rate_limit_is_not_swallowed(monkeypatch):
    """A throttled probe reporting "Claude is not enabled" would be a lie that
    sends somebody to a console page instead of telling them to wait."""
    import anthropic

    _probe_raising(
        monkeypatch, _status_error(anthropic.RateLimitError, "Too many requests", 429)
    )

    with pytest.raises(anthropic.RateLimitError):
        llm.probe(project="proj-x")


def test_the_probe_is_cheap(monkeypatch):
    """It runs on every launch, on somebody's own bill."""
    seen = {}

    class FakeMessages:
        def create(self, **kwargs):
            seen.update(kwargs)
            from types import SimpleNamespace

            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="ready")]
            )

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr(llm, "build_client", lambda project="", region="": FakeClient())

    assert llm.probe(project="proj-x") == "ready"
    assert seen["model"] == llm.MODEL_FAST, "the probe should not use Opus"
    assert seen["max_tokens"] <= 32


def test_building_a_client_without_a_project_refuses_rather_than_guessing(monkeypatch):
    """Guessing would bill somebody else's cost center."""
    monkeypatch.setattr(auth, "stored_credentials", lambda: object())
    monkeypatch.setattr(auth, "project_id", lambda: "")

    with pytest.raises(llm.NoProjectChosen):
        llm.build_client()


def test_building_a_client_signed_out_says_so(monkeypatch):
    monkeypatch.setattr(auth, "stored_credentials", lambda: None)

    with pytest.raises(llm.NotSignedIn):
        llm.build_client(project="proj-x")


# --- the sign-in flow itself --------------------------------------------------


def test_sign_in_asks_for_a_refresh_token_every_time(monkeypatch):
    """Without access_type=offline and prompt=consent, Google returns no
    refresh token on a repeat sign-in. The app then works for one hour and
    stops, which is about the worst failure shape available.
    """
    import google_auth_oauthlib.flow as flow_module

    seen = {}

    class FakeCredentials:
        refresh_token = "refresh-abc"
        id_token = None

    class FakeFlow:
        def run_local_server(self, **kwargs):
            seen.update(kwargs)
            return FakeCredentials()

    monkeypatch.setattr(
        flow_module.InstalledAppFlow,
        "from_client_config",
        classmethod(lambda cls, config, scopes=None: FakeFlow()),
    )
    monkeypatch.setattr(auth, "client_config", lambda: {"client_id": "i", "client_secret": "s"})
    monkeypatch.setattr(auth, "save_refresh_token", lambda t: seen.update(saved=t))
    monkeypatch.setattr(auth, "_remember_account", lambda c: None)

    auth.run_sign_in(open_browser=False)

    assert seen["access_type"] == "offline"
    assert seen["prompt"] == "consent"
    assert seen["saved"] == "refresh-abc"


def test_sign_in_lets_the_os_pick_a_port(monkeypatch):
    """A fixed port fails when it is already taken, and the sign-in wedges."""
    import google_auth_oauthlib.flow as flow_module

    seen = {}

    class FakeFlow:
        def run_local_server(self, **kwargs):
            seen.update(kwargs)
            from types import SimpleNamespace

            return SimpleNamespace(refresh_token="r", id_token=None)

    monkeypatch.setattr(
        flow_module.InstalledAppFlow,
        "from_client_config",
        classmethod(lambda cls, config, scopes=None: FakeFlow()),
    )
    monkeypatch.setattr(auth, "client_config", lambda: {"client_id": "i", "client_secret": "s"})
    monkeypatch.setattr(auth, "save_refresh_token", lambda t: None)
    monkeypatch.setattr(auth, "_remember_account", lambda c: None)

    auth.run_sign_in(open_browser=False)

    assert seen["port"] == 0


# --- what a person is told ----------------------------------------------------


def test_no_project_chosen_is_explained_in_terms_of_cost_centers(monkeypatch):
    """Colleagues are on different projects, so this is not a bug to report."""
    from src.podcastnotes import checks_account

    monkeypatch.setattr(auth, "project_id", lambda: "")

    result = checks_account.check_claude()

    assert result.state is State.FAILED
    assert "project" in result.detail.lower()
    assert "own" in result.remedy.lower() or "team" in result.remedy.lower()


def test_claude_not_enabled_deep_links_to_that_persons_project(monkeypatch):
    """The one step that cannot be automated. A generic error sends people
    hunting through a console they have never opened."""
    from src.podcastnotes import checks_account

    monkeypatch.setattr(auth, "project_id", lambda: "someones-own-project")

    def refuse(project="", region=""):
        raise llm.ClaudeNotEnabled("someones-own-project")

    monkeypatch.setattr(llm, "probe", refuse)

    result = checks_account.check_claude()

    assert result.state is State.FAILED
    assert "someones-own-project" in result.url
    assert "model-garden" in result.url
    assert "someones-own-project" in result.detail


def test_a_working_claude_names_the_project_being_billed(monkeypatch):
    from src.podcastnotes import checks_account

    monkeypatch.setattr(auth, "project_id", lambda: "team-alpha-vertex")
    monkeypatch.setattr(llm, "probe", lambda project="", region="": "ready")

    result = checks_account.check_claude()

    assert result.state is State.OK
    assert "team-alpha-vertex" in result.detail


def test_an_unconfigured_org_is_not_blamed_on_the_user(monkeypatch):
    from src.podcastnotes import checks_account

    monkeypatch.setattr(auth, "is_configured", lambda: False)

    result = checks_account.check_google()

    assert result.state is State.FAILED
    assert "administrator" in result.remedy.lower()
    assert "sign in" not in result.remedy.lower(), "this is not theirs to fix"
    assert result.fixable is False, "there is no button that could help"


def test_being_signed_out_does_tell_them_to_sign_in(monkeypatch):
    from src.podcastnotes import checks_account

    monkeypatch.setattr(auth, "is_configured", lambda: True)
    monkeypatch.setattr(auth, "stored_credentials", lambda: None)

    result = checks_account.check_google()

    assert result.state is State.FAILED
    assert "sign in" in result.remedy.lower()


def test_a_revoked_refresh_token_reads_as_sign_in_again(monkeypatch):
    """The failure the whole module exists for: it looks fine until it is used."""
    from src.podcastnotes import checks_account

    monkeypatch.setattr(auth, "is_configured", lambda: True)
    monkeypatch.setattr(auth, "stored_credentials", lambda: object())

    def revoked(credentials):
        raise RuntimeError("invalid_grant: Token has been expired or revoked.")

    monkeypatch.setattr(auth, "ensure_fresh", revoked)

    result = checks_account.check_google()

    assert result.state is State.FAILED
    assert "again" in result.remedy.lower()


# --- the Drive check does the real thing --------------------------------------


def test_the_drive_check_creates_and_deletes_a_real_document(monkeypatch):
    """`drive.file` being narrow and `drive.file` being sufficient are two
    different claims. Only one of them can be assumed."""
    from src.podcastnotes import checks_account

    calls = {"created": None, "deleted": None}

    class FakeFiles:
        def create(self, body=None, media_body=None, fields=None):
            calls["created"] = body
            calls["media"] = media_body
            return _Execute({"id": "doc-1"})

        def delete(self, fileId=None):
            calls["deleted"] = fileId
            return _Execute({})

    _install_fake_drive(monkeypatch, FakeFiles())

    result = checks_account.check_drive()

    assert result.state is State.OK
    assert calls["created"]["mimeType"] == "application/vnd.google-apps.document"
    assert calls["deleted"] == "doc-1", "the check left a document behind"


def test_the_drive_check_uploads_markdown_like_publishing_does(monkeypatch):
    """Checking with an empty file would prove less than the check appears to:
    publishing relies on Drive converting Markdown server-side."""
    from src.podcastnotes import checks_account

    seen = {}

    class FakeFiles:
        def create(self, body=None, media_body=None, fields=None):
            seen["mimetype"] = media_body.mimetype()
            return _Execute({"id": "doc-1"})

        def delete(self, fileId=None):
            return _Execute({})

    _install_fake_drive(monkeypatch, FakeFiles())

    checks_account.check_drive()

    assert seen["mimetype"] == "text/markdown"


def test_a_failed_cleanup_is_untidy_not_broken(monkeypatch):
    """Creating worked, which is the thing publishing needs."""
    from src.podcastnotes import checks_account

    class FakeFiles:
        def create(self, body=None, media_body=None, fields=None):
            return _Execute({"id": "doc-1"})

        def delete(self, fileId=None):
            raise RuntimeError("insufficient permissions")

    _install_fake_drive(monkeypatch, FakeFiles())

    result = checks_account.check_drive()

    assert result.state is State.OK
    assert "left behind" in result.detail


def test_the_probe_document_says_it_is_safe_to_delete():
    """Somebody will find one in their Drive. It should explain itself."""
    from src.podcastnotes.checks_account import PROBE_DOC_NAME

    assert "delete" in PROBE_DOC_NAME.lower()


class _Execute:
    def __init__(self, value):
        self._value = value

    def execute(self):
        return self._value


def _install_fake_drive(monkeypatch, files):
    import googleapiclient.discovery

    from src.podcastnotes import auth as auth_mod

    monkeypatch.setattr(auth_mod, "stored_credentials", lambda: object())
    monkeypatch.setattr(auth_mod, "ensure_fresh", lambda c: c)

    class FakeService:
        def files(self):
            return files

    monkeypatch.setattr(
        googleapiclient.discovery, "build", lambda *a, **k: FakeService()
    )


# --- the shape of the list ----------------------------------------------------


def test_claude_and_drive_both_wait_on_the_google_sign_in():
    """Neither can be judged before there are credentials, and reporting them
    as broken would send somebody to fix the wrong row."""
    from src.podcastnotes.checks_account import ACCOUNT_CHECKS

    by_key = {c.key: c for c in ACCOUNT_CHECKS}

    assert by_key["claude"].requires == ["google"]
    assert by_key["drive"].requires == ["google"]
    assert by_key["google"].requires == []
