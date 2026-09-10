"""
The checks that involve somebody's accounts: Google, Claude, Drive.

Each one does the real thing. The Drive check genuinely creates a document and
deletes it again, because `drive.file` being the narrow scope we want and
`drive.file` being sufficient to publish are two different claims and only one
of them can be assumed.
"""

from src.podcastnotes import auth
from src.podcastnotes.readiness import Check, failed, ok

# A Google Doc converted from Markdown, which is exactly what publishing does.
# Checking with an empty file would prove less than the check appears to.
PROBE_DOC_NAME = "PodcastNotesWT connection check (safe to delete)"
PROBE_DOC_BODY = "# Connection check\n\nCreated and deleted automatically.\n"


def check_google():
    """Signed in, with a refresh token that still works."""
    if not auth.is_configured():
        return failed(
            "Google sign-in has not been set up for your organisation yet.",
            remedy="An administrator needs to create the sign-in once, for "
                   "everybody. Send them the setup notes.",
            # Deliberately not "fixable": no amount of clicking helps.
        )

    credentials = auth.stored_credentials()
    if credentials is None:
        return failed(
            "Not signed in.",
            remedy="Sign in with your work Google account.",
        )

    try:
        auth.ensure_fresh(credentials)
    except Exception as e:
        return failed(
            f"The saved sign-in is no longer valid ({type(e).__name__}).",
            remedy="Sign in again.",
        )

    who = auth.account_email()
    return ok(f"Signed in as {who}" if who else "Signed in")


def check_claude():
    """One real completion, on the user's own project.

    Everything short of calling can be true while the call still fails: the
    project exists, the credentials are valid, the API is enabled, and nobody
    ever accepted the terms in Model Garden.
    """
    from src.podcastnotes.llm import client as llm

    project = auth.project_id()
    if not project:
        return failed(
            "No Google Cloud project chosen.",
            remedy="Pick the project your team is billed to. Everyone uses "
                   "their own, so ask your team which one.",
        )

    try:
        reply = llm.probe(project=project)
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
    return ok(f"Working, on project {project}")


def check_drive():
    """Create a document and delete it again.

    `drive.file` only grants access to files this app created, which is the
    narrow scope we want. Whether that is *enough* to publish is a separate
    question, and the only honest way to answer it is to publish something.
    """
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaInMemoryUpload

    credentials = auth.stored_credentials()
    if credentials is None:
        return failed("Not signed in to Google.", remedy="Sign in first.")

    auth.ensure_fresh(credentials)
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)

    created = service.files().create(
        body={"name": PROBE_DOC_NAME, "mimeType": "application/vnd.google-apps.document"},
        media_body=MediaInMemoryUpload(
            PROBE_DOC_BODY.encode("utf-8"), mimetype="text/markdown"
        ),
        fields="id",
    ).execute()

    file_id = created.get("id", "")
    if not file_id:
        return failed(
            "Drive accepted the document but returned no id.",
            remedy="Try again, then copy the diagnostics.",
        )

    try:
        service.files().delete(fileId=file_id).execute()
    except Exception as e:
        # Creating worked, which is what publishing needs. Leaving a stray
        # document is untidy, not broken, and it is named so it can be found.
        return ok(f"Working (a test document was left behind: {e})")

    return ok("Can create documents")


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
        )
    if not glean.get_token():
        return failed(
            "No Glean credential.",
            remedy="Create a personal Glean token and paste it in Settings.",
            url="https://app.glean.com/admin/platform/tokenManagement",
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
        purpose="Writing the notes, and saving them",
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
        key="drive",
        title="Google Docs",
        purpose="Publishing the finished document",
        run=check_drive,
        requires=["google"],
    ),
    Check(
        key="glean",
        title="Glean",
        purpose="Looking up who and what the trip involved",
        run=check_glean,
    ),
]
