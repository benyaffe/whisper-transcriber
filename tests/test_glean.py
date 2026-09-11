"""
Tests for the Glean connection.

Two things matter here and neither is the happy path. The credential has to be
the person's own, because Glean is permission-aware and a shared one would
leak documents between colleagues into a published Google Doc. And a token
that authenticates but sees nothing has to read as a failure, because the
alternative is a green tick and a summary full of invented names.

Run with: python -m pytest tests/test_glean.py -v
"""

import pytest

from src.podcastnotes import glean, glean_auth
from src.podcastnotes.readiness import State


@pytest.fixture(autouse=True)
def no_ambient_sign_in(monkeypatch):
    """Ignore whatever this machine happens to be signed in to.

    A real Glean sign-in lives in the developer's Keychain, so without this
    every test of the signed-out path passes here and fails on a machine that
    has never signed in, or the reverse. Tests that want a sign-in say so.
    """
    monkeypatch.setattr(glean_auth, "signed_in", lambda: False)


# --- the address somebody pastes ----------------------------------------------


@pytest.mark.parametrize(
    "pasted",
    [
        "acme",
        "acme.glean.com",
        "https://acme.glean.com",
        "https://acme.glean.com/",
        "https://acme.glean.com/search?q=hello",
        "acme-be.glean.com",
        "http://acme-be.glean.com",
        "  acme  ",
    ],
)
def test_any_shape_of_glean_address_resolves_to_the_instance(pasted):
    """People paste whatever is in the address bar.

    Being strict produces a setup step that fails for a reason nobody can see,
    on the one screen whose whole job is explaining failures.
    """
    assert glean._normalise_instance(pasted) == "acme"


def test_an_empty_address_stays_empty():
    assert glean._normalise_instance("") == ""
    assert glean._normalise_instance("   ") == ""


def test_what_is_stored_is_already_canonical(scoped_settings):
    """Tidying happens once, on the way in.

    Doing it on the way out as well would leave one of the two never
    exercised, and the stored setting would hold whatever was pasted.
    """
    glean.set_instance("https://acme.glean.com/search?q=hi")

    assert scoped_settings.value(glean.SETTINGS_GLEAN_INSTANCE) == "acme"
    assert glean.instance() == "acme"


# --- the credential is per person ---------------------------------------------


def test_the_token_is_read_at_request_time_not_captured(monkeypatch):
    """Passing a callable is the seam OAuth needs: a refreshing credential
    slots in without changing any call site. Capturing the string now would
    close that off and pin a stale token for the process lifetime."""
    import glean.api_client as sdk

    seen = {}

    class FakeGlean:
        def __init__(self, api_token=None, instance=None, timeout_ms=None):
            seen["api_token"] = api_token
            seen["instance"] = instance

    monkeypatch.setattr(sdk, "Glean", FakeGlean)
    monkeypatch.setattr(glean, "instance", lambda: "acme")
    monkeypatch.setattr(glean, "get_token", lambda: "glean-token-1")

    glean.build_client()

    assert callable(seen["api_token"]), "the token should be resolved lazily"
    assert seen["api_token"]() == "glean-token-1"
    assert seen["instance"] == "acme"


def test_a_browser_sign_in_is_used_ahead_of_a_pasted_token(monkeypatch):
    monkeypatch.setattr(glean_auth, "signed_in", lambda: True)
    monkeypatch.setattr(glean_auth, "access_token", lambda where: "oauth-token")
    monkeypatch.setattr(glean, "instance", lambda: "acme")
    monkeypatch.setattr(glean, "get_token", lambda: "pasted-token")

    assert glean.credential() == "oauth-token"


def test_a_field_containing_only_spaces_clears_the_token(monkeypatch):
    """"No token" and "a token that is the empty string" mean the same thing
    and must not be two different stored states."""
    store = {}
    import keyring

    monkeypatch.setattr(keyring, "get_password", lambda s, k: store.get((s, k)))
    monkeypatch.setattr(keyring, "set_password", lambda s, k, v: store.__setitem__((s, k), v))
    monkeypatch.setattr(keyring, "delete_password", lambda s, k: store.pop((s, k), None))

    glean.save_token("real-token")
    glean.save_token("   ")

    assert store == {}, "a blank field left an entry behind"
    assert glean.get_token() == ""


def test_the_token_lives_in_the_same_keychain_service_as_the_others():
    from src.core.config import KEYRING_SERVICE

    assert glean.KEYRING_SERVICE == KEYRING_SERVICE


def test_no_instance_refuses_before_calling(monkeypatch):
    """Both values are pinned deliberately.

    Leaving the token to the real Keychain made this pass on a machine with no
    Glean token and fail to test anything at all: the token guard fired first
    and the instance guard could have been deleted unnoticed.
    """
    monkeypatch.setattr(glean, "instance", lambda: "")
    monkeypatch.setattr(glean, "get_token", lambda: "a-perfectly-good-token")

    with pytest.raises(glean.NotConfigured, match="instance"):
        glean.build_client()


def test_no_token_refuses_before_calling(monkeypatch):
    monkeypatch.setattr(glean, "instance", lambda: "acme")
    monkeypatch.setattr(glean, "get_token", lambda: "")

    with pytest.raises(glean.NotConfigured, match="credential"):
        glean.build_client()


# --- results ------------------------------------------------------------------


def test_results_come_back_as_plain_dicts(monkeypatch):
    """Nothing downstream should import the Glean SDK to read a search result."""
    _fake_search(monkeypatch, [
        _result("Ashford trip notes", "https://d/1", "gdrive", "We met the team"),
    ])

    results = glean.search("ashford")

    assert results == [
        {
            "title": "Ashford trip notes",
            "url": "https://d/1",
            "source": "gdrive",
            "snippet": "We met the team",
        }
    ]


def test_a_result_with_no_snippet_is_still_usable(monkeypatch):
    _fake_search(monkeypatch, [_result("Title only", "https://d/2", "slack", None)])

    assert glean.search("x")[0]["snippet"] == ""


def test_no_results_is_an_empty_list_not_a_crash(monkeypatch):
    _fake_search(monkeypatch, [])

    assert glean.search("nothing at all") == []


# --- research: what Claude reads ----------------------------------------------


def test_research_brings_back_the_body_not_just_a_snippet(monkeypatch):
    """The whole reason this exists. A snippet cannot establish that a garbled
    name belongs to a real person; the body of somebody's working doc can."""
    _fake_research(monkeypatch, [
        _rich("Working Doc", "https://d/1", "gdrive", full_text="Dr. Miles Nadeau, our consultant."),
    ])

    assert glean.research("nadeau")[0]["body"] == "Dr. Miles Nadeau, our consultant."


def test_research_reports_dates_because_the_caller_cannot_ask_later(monkeypatch):
    _fake_research(monkeypatch, [
        _rich("Plan", "https://d/1", "gdrive", full_text="x",
              created="2026-05-11 18:38:27+00:00", updated="2026-08-14 00:43:57+00:00"),
    ])

    hit = glean.research("plan")[0]

    assert hit["created"] == "2026-05-11"
    assert hit["updated"] == "2026-08-14"


def test_a_slack_thread_is_joined_rather_than_dropped(monkeypatch):
    """Slack arrives as one entry per message. Reading only the first would
    lose the reply that usually carries the answer."""
    _fake_research(monkeypatch, [
        _rich("thread", "https://s/1", "slack", full_text_list=["I'm in Merrow", "with Miles Nadeau"]),
    ])

    assert glean.research("merrow")[0]["body"] == "I'm in Merrow\nwith Miles Nadeau"


def test_a_datasource_with_no_body_falls_back_to_snippets(monkeypatch):
    _fake_research(monkeypatch, [_rich("Title only", "https://d/2", "jira", snippet="a snippet")])

    assert glean.research("x")[0]["body"] == "a snippet"


def test_a_body_longer_than_the_budget_is_truncated(monkeypatch):
    _fake_research(monkeypatch, [_rich("Long", "https://d/3", "gdrive", full_text="z" * 20000)])

    assert len(glean.research("x")[0]["body"]) == glean.BODY_CHARS


def test_llm_content_is_requested_with_a_size_or_glean_rejects_it(monkeypatch):
    """Both of these were 400s before they were arguments. `max_snippet_size`
    is required whenever LLM content is asked for, and `facet_bucket_size` is
    required by the options model even with no facets wanted."""
    seen = {}
    _fake_research(monkeypatch, [], seen=seen)

    glean.research("anything")

    assert seen["max_snippet_size"] == glean.BODY_CHARS
    assert seen["options"].return_llm_content_over_snippets is True
    assert seen["options"].facet_bucket_size is not None


def test_a_result_carrying_no_document_is_skipped_not_crashed_on(monkeypatch):
    _fake_research(monkeypatch, [_rich("ok", "https://d/1", "gdrive", full_text="fine"), _headless()])

    assert [h["title"] for h in glean.research("x")] == ["ok"]


def test_research_needs_a_credential_like_everything_else(monkeypatch):
    monkeypatch.setattr(glean, "instance", lambda: "acme")
    monkeypatch.setattr(glean, "get_token", lambda: "")

    with pytest.raises(glean.NotConfigured):
        glean.research("x")


# --- the check ----------------------------------------------------------------


def test_a_credential_that_sees_nothing_is_not_a_pass(monkeypatch):
    """The failure that would otherwise show a green tick and then produce a
    document full of names Claude has never seen."""
    from src.podcastnotes import checks_account

    monkeypatch.setattr(glean, "instance", lambda: "acme")
    monkeypatch.setattr(glean, "get_token", lambda: "token")
    monkeypatch.setattr(glean, "search", lambda q, page_size=10: [])

    result = checks_account.check_glean()

    assert result.state is State.FAILED
    assert "scoped" in result.remedy.lower() or "own account" in result.remedy.lower()


def test_a_working_glean_names_the_instance(monkeypatch):
    from src.podcastnotes import checks_account

    monkeypatch.setattr(glean, "instance", lambda: "acme")
    monkeypatch.setattr(glean, "get_token", lambda: "token")
    monkeypatch.setattr(
        glean, "search", lambda q, page_size=10: [{"title": "a", "url": "", "source": "", "snippet": ""}]
    )

    result = checks_account.check_glean()

    assert result.state is State.OK
    assert "acme" in result.detail


def test_a_missing_address_is_asked_for_in_plain_terms(monkeypatch):
    from src.podcastnotes import checks_account

    monkeypatch.setattr(glean, "instance", lambda: "")

    result = checks_account.check_glean()

    assert result.state is State.FAILED
    assert "browser" in result.remedy.lower(), "tell them where to find it"


def test_the_probe_query_is_generic(monkeypatch):
    """A query returning nothing would tell us about the query, not the
    connection, and would fail the check for a working setup."""
    from src.podcastnotes.checks_account import GLEAN_PROBE_QUERY

    assert GLEAN_PROBE_QUERY.isalpha(), "no operators, no filters, no quoting"
    assert len(GLEAN_PROBE_QUERY.split()) == 1


def test_glean_does_not_wait_on_the_google_sign_in():
    """They are unrelated, and blocking Glean behind Google would hide a real
    Glean problem until after Google is fixed."""
    from src.podcastnotes.checks_account import ACCOUNT_CHECKS

    by_key = {c.key: c for c in ACCOUNT_CHECKS}

    assert by_key["glean"].requires == []


# --- helpers ------------------------------------------------------------------


class _Doc:
    def __init__(self, title, url, source):
        self.title = title
        self.url = url
        self.datasource = source


class _Snippet:
    def __init__(self, text):
        self.text = text


class _SearchResult:
    def __init__(self, title, url, source, snippet):
        self.document = _Doc(title, url, source)
        self.title = title
        self.url = url
        self.snippets = [_Snippet(snippet)] if snippet else []


def _result(title, url, source, snippet):
    return _SearchResult(title, url, source, snippet)


class _Meta:
    def __init__(self, created, updated, owner):
        self.create_time = created
        self.update_time = updated
        self.owner = _Named(owner) if owner else None


class _Named:
    def __init__(self, name):
        self.name = name


class _RichDoc(_Doc):
    def __init__(self, title, url, source, created, updated, owner):
        super().__init__(title, url, source)
        self.metadata = _Meta(created, updated, owner)


class _RichResult:
    def __init__(self, document, full_text, full_text_list, snippet):
        self.document = document
        self.full_text = full_text
        self.full_text_list = full_text_list
        self.snippets = [_Snippet(snippet)] if snippet else []


def _rich(title, url, source, full_text=None, full_text_list=None, snippet=None,
          created=None, updated=None, owner=None):
    return _RichResult(
        _RichDoc(title, url, source, created, updated, owner),
        full_text, full_text_list, snippet,
    )


def _headless():
    """A result the SDK returned with no document attached, which happens."""
    return _RichResult(None, None, None, None)


def _fake_research(monkeypatch, results, seen=None):
    class FakeResponse:
        def __init__(self):
            self.results = results

    class FakeSearch:
        def query(self, query=None, page_size=None, request_options=None, max_snippet_size=None):
            if seen is not None:
                seen["query"] = query
                seen["options"] = request_options
                seen["max_snippet_size"] = max_snippet_size
            return FakeResponse()

    class FakeInner:
        search = FakeSearch()

    class FakeClient:
        client = FakeInner()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(glean, "build_client", lambda: FakeClient())


def _fake_search(monkeypatch, results):
    class FakeResponse:
        def __init__(self):
            self.results = results

    class FakeSearch:
        def query(self, query=None, page_size=None):
            return FakeResponse()

    class FakeInner:
        search = FakeSearch()

    class FakeClient:
        client = FakeInner()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(glean, "build_client", lambda: FakeClient())
