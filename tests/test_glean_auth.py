"""
Tests for signing in to Glean without an administrator.

Glean publishes a registration endpoint, so the app registers itself and runs
an ordinary browser sign-in. That removes the only part of Glean setup that
needed somebody else, and it is worth pinning: the fallback is pasted tokens,
which is what this replaced.

Nothing here touches the network. The authorization server is a stub.

Run with: python -m pytest tests/test_glean_auth.py -v
"""

import json
import urllib.error

import pytest

from src.podcastnotes import glean_auth


METADATA = {
    "issuer": "https://acme-be.glean.com/oauth",
    "authorization_endpoint": "https://acme-be.glean.com/oauth/authorize",
    "token_endpoint": "https://acme-be.glean.com/oauth/token",
    "registration_endpoint": "https://acme-be.glean.com/oauth/register",
    "code_challenge_methods_supported": ["S256"],
}


@pytest.fixture
def fake_keychain(monkeypatch):
    store = {}
    import keyring

    monkeypatch.setattr(keyring, "get_password", lambda s, k: store.get((s, k)))
    monkeypatch.setattr(keyring, "set_password", lambda s, k, v: store.__setitem__((s, k), v))
    monkeypatch.setattr(keyring, "delete_password", lambda s, k: store.pop((s, k), None))
    return store


# --- what the app asks Glean for ----------------------------------------------


def test_it_only_asks_for_what_it_uses():
    """A sign-in prompt listing admin or write access would be alarming, and
    would deserve to be."""
    assert "search" in glean_auth.SCOPES
    for forbidden in ("admin", "content_hiding", "data_governance", "auth_token_creator"):
        assert forbidden not in glean_auth.SCOPES


def test_it_asks_for_a_refresh_token():
    """Without offline_access the sign-in lasts an hour and then stops, which
    is a worse failure than never working."""
    assert "offline_access" in glean_auth.SCOPES


def test_registration_is_a_public_client_using_pkce(monkeypatch):
    """A secret shipped on somebody's laptop is not a secret."""
    sent = {}
    _fake_http(monkeypatch, on_post=lambda url, body, headers: sent.update(
        url=url, body=json.loads(body)) or {"client_id": "generated-123"})

    client_id = glean_auth.register(METADATA, "http://127.0.0.1:5000/oauth/callback")

    assert client_id == "generated-123"
    assert sent["body"]["token_endpoint_auth_method"] == "none"
    assert "client_secret" not in sent["body"]
    assert sent["body"]["redirect_uris"] == ["http://127.0.0.1:5000/oauth/callback"]
    assert "refresh_token" in sent["body"]["grant_types"]


def test_registration_that_returns_no_client_id_is_an_error(monkeypatch):
    _fake_http(monkeypatch, on_post=lambda *a: {"ok": True})

    with pytest.raises(glean_auth.SignInFailed):
        glean_auth.register(METADATA, "http://127.0.0.1:5000/oauth/callback")


# --- discovery ----------------------------------------------------------------


def test_endpoints_are_discovered_rather_than_assumed(monkeypatch):
    """The paths are conventional, not guaranteed, and an instance on another
    Glean release would move them."""
    _fake_http(monkeypatch, on_get=lambda url: METADATA)

    assert glean_auth.discover("acme")["token_endpoint"].endswith("/oauth/token")


def test_an_instance_missing_a_token_endpoint_is_rejected(monkeypatch):
    _fake_http(monkeypatch, on_get=lambda url: {"authorization_endpoint": "x"})

    with pytest.raises(glean_auth.SignInFailed, match="token_endpoint"):
        glean_auth.discover("acme")


def test_self_registration_is_detected_not_assumed():
    """An instance without it needs a pasted token, and the checklist has to
    ask for the right thing."""
    assert glean_auth.supports_self_registration(METADATA) is True
    assert glean_auth.supports_self_registration({"token_endpoint": "x"}) is False


def test_the_discovery_url_matches_the_instance(monkeypatch):
    seen = {}
    _fake_http(monkeypatch, on_get=lambda url: seen.update(url=url) or METADATA)

    glean_auth.discover("acme")

    assert seen["url"] == "https://acme-be.glean.com/.well-known/oauth-authorization-server"


# --- PKCE ---------------------------------------------------------------------


def test_the_pkce_challenge_is_a_sha256_of_the_verifier():
    import base64
    import hashlib

    verifier, challenge = glean_auth._pkce_pair()

    expected = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).decode().rstrip("=")
    assert challenge == expected
    assert "=" not in challenge, "padding breaks the URL encoding"


def test_every_sign_in_uses_a_fresh_verifier():
    assert glean_auth._pkce_pair()[0] != glean_auth._pkce_pair()[0]


# --- the token exchange -------------------------------------------------------


def test_a_rotated_refresh_token_replaces_the_old_one(monkeypatch, fake_keychain):
    """Glean may hand back a new one on each use. Dropping it logs the person
    out at some unpredictable point later."""
    glean_auth.save_client_id("client-1")
    glean_auth.save_refresh_token("original")

    _fake_http(
        monkeypatch,
        on_get=lambda url: METADATA,
        on_post=lambda url, body, headers: {
            "access_token": "access-1",
            "refresh_token": "rotated",
        },
    )

    assert glean_auth.access_token("acme") == "access-1"
    assert glean_auth.get_refresh_token() == "rotated"


def test_a_response_without_a_new_refresh_token_keeps_the_old_one(monkeypatch, fake_keychain):
    glean_auth.save_client_id("client-1")
    glean_auth.save_refresh_token("original")

    _fake_http(
        monkeypatch,
        on_get=lambda url: METADATA,
        on_post=lambda url, body, headers: {"access_token": "access-1"},
    )

    glean_auth.access_token("acme")

    assert glean_auth.get_refresh_token() == "original"


def test_no_saved_sign_in_means_no_token_and_no_network(monkeypatch, fake_keychain):
    called = []
    _fake_http(monkeypatch, on_get=lambda url: called.append(url) or METADATA)

    assert glean_auth.access_token("acme") == ""
    assert called == [], "it went to the network with nothing to send"


def test_a_rejected_refresh_says_what_glean_said(monkeypatch, fake_keychain):
    glean_auth.save_client_id("client-1")
    glean_auth.save_refresh_token("revoked")

    def refuse(url, body, headers):
        raise urllib.error.HTTPError(url, 400, "Bad Request", {}, None)

    _fake_http(monkeypatch, on_get=lambda url: METADATA, on_post=refuse)

    with pytest.raises(glean_auth.SignInFailed, match="400"):
        glean_auth.access_token("acme")


# --- what is stored -----------------------------------------------------------


def test_signed_in_needs_both_halves(fake_keychain):
    assert glean_auth.signed_in() is False

    glean_auth.save_refresh_token("r")
    assert glean_auth.signed_in() is False, "a token with no client id is unusable"

    glean_auth.save_client_id("c")
    assert glean_auth.signed_in() is True


def test_signing_out_removes_both(fake_keychain):
    glean_auth.save_refresh_token("r")
    glean_auth.save_client_id("c")

    glean_auth.sign_out()

    assert fake_keychain == {}


def test_it_shares_the_keychain_service_with_everything_else():
    from src.core.config import KEYRING_SERVICE

    assert glean_auth.KEYRING_SERVICE == KEYRING_SERVICE


# --- how the rest of the app reaches it ---------------------------------------


def test_a_browser_sign_in_is_preferred_over_a_pasted_token(monkeypatch, fake_keychain):
    from src.podcastnotes import glean

    monkeypatch.setattr(glean, "instance", lambda: "acme")
    monkeypatch.setattr(glean, "get_token", lambda: "pasted-token")
    monkeypatch.setattr(glean_auth, "signed_in", lambda: True)
    monkeypatch.setattr(glean_auth, "access_token", lambda i: "oauth-token")

    assert glean.credential() == "oauth-token"


def test_a_pasted_token_still_works_where_registration_is_not_allowed(
    monkeypatch, fake_keychain
):
    """Not every instance permits self-registration, and those users are not
    left without a way in."""
    from src.podcastnotes import glean

    monkeypatch.setattr(glean, "get_token", lambda: "pasted-token")
    monkeypatch.setattr(glean_auth, "signed_in", lambda: False)

    assert glean.credential() == "pasted-token"
    assert glean.has_credential() is True


def test_neither_means_no_credential(monkeypatch, fake_keychain):
    from src.podcastnotes import glean

    monkeypatch.setattr(glean, "get_token", lambda: "")
    monkeypatch.setattr(glean_auth, "signed_in", lambda: False)

    assert glean.has_credential() is False


def test_the_client_resolves_the_token_per_request(monkeypatch, fake_keychain):
    """An OAuth access token lasts about an hour. Capturing one at construction
    is how a long job dies in the middle."""
    import glean.api_client as sdk

    from src.podcastnotes import glean as glean_mod

    seen = {}
    monkeypatch.setattr(sdk, "Glean", lambda **kw: seen.update(kw))
    monkeypatch.setattr(glean_mod, "instance", lambda: "acme")
    monkeypatch.setattr(glean_mod, "has_credential", lambda: True)

    glean_mod.build_client()

    assert seen["api_token"] is glean_mod.credential, "the token was captured, not deferred"


# --- helpers ------------------------------------------------------------------


def _fake_http(monkeypatch, on_get=None, on_post=None):
    """Stands in for urllib, dispatching on whether there is a body."""
    import urllib.request

    class FakeResponse:
        def __init__(self, payload):
            self._payload = json.dumps(payload).encode()

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(request, timeout=None):
        if isinstance(request, str):
            return FakeResponse(on_get(request))
        if request.data:
            return FakeResponse(on_post(request.full_url, request.data, request.headers))
        return FakeResponse(on_get(request.full_url))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
