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


def build_client():
    """A Glean client scoped to this person.

    The token is passed as a callable so it is read at request time. That costs
    nothing now and is the seam OAuth needs later: a refreshing credential
    slots in without changing any call site.
    """
    from glean.api_client import Glean

    where = instance()
    if not where:
        raise NotConfigured("No Glean instance set.")
    if not get_token():
        raise NotConfigured("No Glean credential.")

    return Glean(api_token=get_token, instance=where, timeout_ms=CHECK_TIMEOUT_MS)


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
