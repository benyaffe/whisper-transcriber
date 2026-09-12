"""
The checks that involve somebody's accounts: Google, Claude, Glean.

Each one does the real thing rather than reading a stored setting.

There is no Drive check because there is no Drive. Publishing goes through the
clipboard into a blank Google Doc, which needs no credential, no scope and
nobody's permission. See src/podcastnotes/publish.py.
"""

from src.podcastnotes import auth
from src.podcastnotes.readiness import Check, failed, ok


def check_google():
    """Signed in, with credentials that still work right now.

    Organisations that put Google Cloud behind a session policy expire these
    every day or two by design, so this failing is routine rather than
    alarming and the wording says so.
    """
    credentials = auth.stored_credentials()

    if credentials is None:
        if not auth.is_configured():
            return failed(
                "Google sign-in has not been set up for this app yet.",
                remedy="Somebody with access to your Google Cloud console "
                       "creates the sign-in once, for everybody. It takes "
                       "about ten minutes and never needs doing again.",
                # Nothing to press. Until the OAuth client exists there is no
                # sign-in to start, and a button that cannot work is worse
                # than no button.
                fixable=False,
            )
        return failed(
            "Not signed in.",
            remedy="Sign in with your work Google account.",
            action="Sign in",
        )

    try:
        auth.ensure_fresh(credentials)
    except Exception:
        return failed(
            "Your Google sign-in has expired.",
            remedy="This is normal: your organisation expires these every "
                   "day or two. Signing in again takes a few seconds.",
            action="Sign in again",
            fixable=auth.is_configured(),
        )

    if auth.credentials_source() == auth.SOURCE_GCLOUD:
        return ok("Using your gcloud sign-in")

    who = auth.account_email()
    return ok(f"Signed in as {who}" if who else "Signed in")


def check_claude():
    """One real completion, on the user's own project.

    Everything short of calling can be true while the call still fails: the
    project exists, the credentials are valid, the API is enabled, and nobody
    ever accepted the terms in Model Garden.
    """
    from src.podcastnotes.llm import client as llm

    project = auth.project_id() or _guess_project()
    if not project:
        return failed(
            "No Google Cloud project chosen.",
            remedy="Pick the project your team is billed to. Everyone uses "
                   "their own, so ask your team which one.",
            action="Choose project",
        )

    try:
        reply = llm.probe(project=project)
    except llm.UnsupportedRegion as e:
        return failed(
            f"The Claude location is set to {e.region}, which does not carry "
            f"the model this app uses.",
            remedy="Set it to global, or to us or eu if your data has to stay "
                   "in one of those.",
        )
    except llm.ClaudeNotEnabled as e:
        return failed(
            f"Claude is not switched on in project {e.project}.",
            remedy="Open the page below, click Enable, and accept the terms. "
                   "It is a one-off.",
            url=e.url,
        )
    except llm.NotSignedIn:
        return failed("Not signed in to Google.", remedy="Sign in first.")

    if not reply:
        return failed(
            "Claude replied with nothing.",
            remedy="Try again. If it keeps happening, copy the diagnostics.",
        )

    # Only remembered once it has been shown to work. Saving a guess before
    # proving it would leave somebody with a silently wrong billing project
    # and no reason to look at the setting again.
    if project != auth.project_id():
        auth.set_project_id(project)

    return ok(f"Working, on project {project}")


def _guess_project() -> str:
    """A sensible default so most people never see the project picker.

    Only the project the machine is already pointed at. Anything cleverer
    risks billing somebody's work to a cost center they do not own, which is
    a mistake they would not notice until an invoice arrived.
    """
    return auth.default_project()


def check_glean():
    """One real search, returning at least one document this person can see.

    A credential that authenticates but returns nothing is not a working
    setup: the whole point of Glean here is context, and a token scoped to
    nothing produces a summary full of names Claude has never seen while
    every indicator says green.
    """
    from src.podcastnotes import glean

    if not glean.instance():
        return failed(
            "No Glean address set.",
            remedy="Enter your company's Glean address. It is the one in your "
                   "browser when you use Glean.",
            action="Add address",
        )
    if not glean.has_credential():
        return failed(
            "Not signed in to Glean.",
            remedy="Sign in with your normal Glean account. It opens your "
                   "browser and takes a few seconds.",
            action="Sign in",
        )

    results = glean.search(GLEAN_PROBE_QUERY, page_size=1)
    if not results:
        return failed(
            "Glean answered, but found nothing you can see.",
            remedy="The credential works but is not scoped to any content. "
                   "Check it was made for your own account.",
        )

    return ok(f"Working, on {glean.instance()}")


# Something every indexed workspace has. A query that returns nothing tells us
# about the query, not the connection, so it needs to be genuinely generic.
GLEAN_PROBE_QUERY = "meeting"


ACCOUNT_CHECKS = [
    Check(
        key="google",
        title="Google account",
        purpose="Letting the app write the notes for you",
        run=check_google,
    ),
    Check(
        key="claude",
        title="Claude",
        purpose="Correcting names and writing the summary",
        run=check_claude,
        requires=["google"],
    ),
    Check(
        key="glean",
        title="Glean",
        purpose="Looking up who and what the trip involved",
        run=check_glean,
    ),
]
