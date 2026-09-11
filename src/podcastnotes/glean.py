"""
Glean: what the trip was actually about.

Without it the corrections, the attributions and the summary are Claude
guessing at names it has never seen, so this is required rather than optional.
Every colleague has Glean access, which is what makes it viable to depend on.

**The credential is per person, never shared.** Glean's Client API is
permission-aware: a user-scoped credential returns only what that person is
allowed to see. A single shared token would let one colleague's trip surface
documents another colleague cannot open, and those names would then be
published into a shared Google Doc. That is a confidentiality property, not a
convenience, and it is why `api_token` here is always somebody's own.

OAuth is the intended end state and needs an app registered for the
organisation. Until that exists, a personal API token does the same job with
the same permission scoping; it is just less pleasant to obtain.

This module shares a name with the SDK it wraps, which is safe but looks like
a bug: `from glean.api_client import Glean` below resolves to the installed
package, not to this file, because Python 3 imports are absolute and this
module's real name is src.podcastnotes.glean.

No Qt.
"""

from src.core.config import KEYRING_SERVICE

KEYRING_GLEAN_TOKEN = "glean_api_token"
SETTINGS_GLEAN_INSTANCE = "glean_instance"

# How long to wait before deciding Glean is unreachable. The checklist runs at
# launch, so it cannot sit there for the SDK's default.
CHECK_TIMEOUT_MS = 15_000


class NotConfigured(Exception):
    """No instance or no credential, so there is nothing to call."""


def instance() -> str:
    """The organisation's Glean instance, already canonical.

    set_instance does the tidying, so what is stored is what is returned.
    Normalising on both sides would leave one of the two never exercised.
    """
    from src.core.config import _settings

    return (_settings().value(SETTINGS_GLEAN_INSTANCE, "", type=str) or "").strip()


def _normalise_instance(raw: str) -> str:
    if not raw:
        return ""
    value = raw.strip()
    for prefix in ("https://", "http://"):
        if value.startswith(prefix):
            value = value[len(prefix):]
    value = value.split("/")[0]
    # app.glean.com and <name>-be.glean.com both appear in the wild; the SDK
    # wants the bare name.
    for suffix in ("-be.glean.com", ".glean.com"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value.strip().strip(".")


def set_instance(value: str):
    """Store the instance, tidied.

    People paste whatever is in their address bar. Being strict about it would
    produce a setup step that fails for a reason nobody can see, on the one
    screen whose entire job is explaining failures.
    """
    from src.core.config import _settings

    _settings().setValue(SETTINGS_GLEAN_INSTANCE, _normalise_instance(value))


def get_token() -> str:
    import keyring

    try:
        return keyring.get_password(KEYRING_SERVICE, KEYRING_GLEAN_TOKEN) or ""
    except Exception:
        return ""


def save_token(token: str):
    import keyring

    # Normalised before the decision, not after. Branching on the raw value and
    # stripping inside meant a field holding only spaces stored an empty string
    # rather than removing the entry, so "no token" and "a token that is the
    # empty string" became two different states with one meaning.
    token = (token or "").strip()
    try:
        if token:
            keyring.set_password(KEYRING_SERVICE, KEYRING_GLEAN_TOKEN, token)
        else:
            try:
                keyring.delete_password(KEYRING_SERVICE, KEYRING_GLEAN_TOKEN)
            except keyring.errors.PasswordDeleteError:
                pass
    except Exception as e:
        raise RuntimeError(f"Could not save the Glean token: {e}")


def credential() -> str:
    """A usable bearer token for this person, however they signed in.

    Browser sign-in first, because it needs nothing from anybody. A pasted
    personal token is the fallback for an instance that does not allow
    applications to register themselves.
    """
    from src.podcastnotes import glean_auth

    if glean_auth.signed_in():
        token = glean_auth.access_token(instance())
        if token:
            return token
    return get_token()


def has_credential() -> bool:
    """Whether there is something to try, without spending a network call."""
    from src.podcastnotes import glean_auth

    return glean_auth.signed_in() or bool(get_token())


def build_client():
    """A Glean client scoped to this person.

    The token is passed as a callable so it is resolved at request time. That
    is what lets an OAuth access token, which lasts about an hour, be minted
    fresh rather than captured once and then quietly expire mid-job.
    """
    from glean.api_client import Glean

    where = instance()
    if not where:
        raise NotConfigured("No Glean instance set.")
    if not has_credential():
        raise NotConfigured("No Glean credential.")

    return Glean(api_token=credential, instance=where, timeout_ms=CHECK_TIMEOUT_MS)


def search(query: str, page_size: int = 10) -> list[dict]:
    """One search, returned as plain dicts so nothing downstream imports the SDK."""
    with build_client() as client:
        response = client.client.search.query(query=query, page_size=page_size)

    results = []
    for result in getattr(response, "results", None) or []:
        document = getattr(result, "document", None)
        results.append(
            {
                "title": getattr(document, "title", "") or getattr(result, "title", ""),
                "url": getattr(document, "url", "") or "",
                "source": getattr(document, "datasource", "") or "",
                "snippet": _first_snippet(result),
            }
        )
    return results


def _first_snippet(result) -> str:
    for snippet in getattr(result, "snippets", None) or []:
        text = getattr(snippet, "text", "") or getattr(snippet, "snippet", "")
        if text:
            return text
    return ""


# How much of a document to bring back when Claude is the reader, and how many
# documents. Both are set to favour recall over brevity.
#
# Ten thousand characters is Glean's own ceiling for this parameter, so there
# is nothing above it to reach for. An earlier and much lower value was chosen
# to keep the token count down, which is the wrong trade here: the evidence
# that resolves a garbled name is often a single line partway through somebody
# else's document, and truncating is how you lose exactly that line.
BODY_CHARS = 10000
RESULTS_PER_SEARCH = 16


def research(query: str, page_size: int = RESULTS_PER_SEARCH) -> list[dict]:
    """One search, with document bodies and dates, for Claude rather than a person.

    `search` above answers "did the credential work"; a snippet is plenty for
    that. Building a context map is a different job: a title and one snippet
    cannot tell you that "Miles Nadoe" is Dr. Miles Nadeau, because the
    evidence for that is in the body of somebody's working doc.

    Two details are not obvious from the SDK signature and were both found by
    being rejected. `return_llm_content_over_snippets` is refused unless
    `max_snippet_size` is also set to something between 1 and 10000, and
    `SearchRequestOptions` requires `facet_bucket_size` even when no facets are
    wanted. Neither has a usable default.

    Dates come back too, because the caller cannot ask for them afterwards and
    a context map that cannot say when something happened is much less useful.
    """
    from glean.api_client.models.searchrequestoptions import SearchRequestOptions

    options = SearchRequestOptions(
        facet_bucket_size=0,
        return_llm_content_over_snippets=True,
    )
    with build_client() as client:
        response = client.client.search.query(
            query=query,
            page_size=page_size,
            request_options=options,
            max_snippet_size=BODY_CHARS,
        )

    results = []
    for result in getattr(response, "results", None) or []:
        document = getattr(result, "document", None)
        if document is None:
            continue
        metadata = getattr(document, "metadata", None)
        results.append(
            {
                "title": getattr(document, "title", "") or "",
                "url": getattr(document, "url", "") or "",
                "source": getattr(document, "datasource", "") or "",
                "created": _date(getattr(metadata, "create_time", None)),
                "updated": _date(getattr(metadata, "update_time", None)),
                "owner": _person(getattr(metadata, "owner", None)),
                "body": _body(result)[:BODY_CHARS],
            }
        )
    return results


def _body(result) -> str:
    """Whatever the richest available text is, in preference order.

    Slack conversations arrive as `full_text_list`, one entry per message,
    while documents arrive as `full_text`. Falling back to snippets matters for
    datasources that return neither, which would otherwise contribute a title
    and nothing to reason from.
    """
    parts = getattr(result, "full_text_list", None)
    if parts:
        return "\n".join(p for p in parts if p)
    whole = getattr(result, "full_text", None)
    if whole:
        return whole
    return "\n".join(
        (getattr(s, "text", "") or "") for s in (getattr(result, "snippets", None) or [])
    ).strip()


def _date(value) -> str:
    """Just the day. The time of day has never mattered and adds noise."""
    return str(value)[:10] if value else ""


def _person(value) -> str:
    return getattr(value, "name", "") or "" if value is not None else ""
