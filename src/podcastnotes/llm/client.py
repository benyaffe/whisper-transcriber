"""
Talking to Claude on whichever Vertex project the user is billed to.

Colleagues are on different cost centers and therefore different Vertex
projects, so nothing here is hardcoded. The project comes from the user's
settings and the credentials come from the in-app Google sign-in, which is
what removes the gcloud dependency: AnthropicVertex takes an explicit
`credentials` object, so no `gcloud auth application-default login` and no
Terminal.

No Qt, no network at import time.
"""

from src.podcastnotes import auth

# Anything a person reads gets Opus. Bulk mechanical passes get Haiku, where
# the quality difference does not land on the artefact.
#
# Two naming conventions, and mixing them up produces a 404 that looks exactly
# like a permissions problem. Models from the 4.6 generation onwards have no
# date suffix. Older ones keep a dated snapshot, and on Vertex the separator is
# "@" where the Claude API uses "-".
MODEL_BEST = "claude-opus-5"
MODEL_FAST = "claude-haiku-4-5@20251001"

# The global endpoint, not a region, and this is a correctness requirement
# rather than a preference. Specific regional endpoints such as us-east5 carry
# Sonnet 4.6 and earlier; Opus 5 exists only on the global and multi-region
# endpoints. Pinning a region therefore does not merely cost 10% more, it makes
# the model this product depends on unavailable.
#
# Data residency is served by the multi-region endpoints "us" and "eu", which
# keep traffic inside that geography and do carry current models. Those are the
# only alternatives to global that work.
DEFAULT_REGION = "global"
SUPPORTED_REGIONS = ("global", "us", "eu")
SETTINGS_VERTEX_REGION = "vertex_region"

MODEL_GARDEN_URL = (
    "https://console.cloud.google.com/vertex-ai/publishers/anthropic/"
    "model-garden/claude-opus-5?project={project}"
)


class NotSignedIn(Exception):
    """No Google credentials, so there is nothing to build a client from."""


class NoProjectChosen(Exception):
    """Signed in, but no Vertex project picked yet."""


class UnsupportedRegion(Exception):
    """A single region was pinned, and current models are not served there.

    Kept apart from ClaudeNotEnabled because the remedies are opposites: one
    says go and enable something, the other says the thing you enabled is fine
    and the endpoint is wrong.
    """

    def __init__(self, region: str):
        self.region = region
        super().__init__(
            f"{region} is a single-region endpoint, which does not carry the "
            f"current models. Use global, us or eu."
        )


class ClaudeNotEnabled(Exception):
    """The project is real but Claude has not been enabled in Model Garden.

    The one manual step that cannot be automated. Carries the deep link to the
    right page for *this* project, because a generic error here sends people
    hunting through a console they have never opened.
    """

    def __init__(self, project: str, detail: str = ""):
        self.project = project
        self.url = MODEL_GARDEN_URL.format(project=project)
        super().__init__(detail or f"Claude is not enabled in project {project}.")


def configured_region() -> str:
    """The endpoint to use, defaulting to global."""
    from src.core.config import _settings

    value = _settings().value(SETTINGS_VERTEX_REGION, DEFAULT_REGION, type=str)
    return value or DEFAULT_REGION


def build_client(project: str = "", region: str = ""):
    """An AnthropicVertex bound to the user's own project and credentials."""
    from anthropic import AnthropicVertex

    region = region or configured_region()
    credentials = auth.stored_credentials()
    if credentials is None:
        raise NotSignedIn("Sign in with Google first.")

    project = project or auth.project_id()
    if not project:
        raise NoProjectChosen("Choose which Google Cloud project to bill this to.")

    auth.ensure_fresh(credentials)
    return AnthropicVertex(
        project_id=project, region=region, credentials=credentials
    )


def probe(project: str = "", region: str = "") -> str:
    """One real, cheap completion on the model real work uses. Returns the reply.

    A real call rather than a permissions inspection, because every way of
    asking "would this work?" short of asking is wrong in some case that
    matters: the project can exist, the credentials can be valid, the API can
    be enabled, and the call can still fail because nobody accepted the terms
    in Model Garden.

    It uses MODEL_BEST specifically, not the cheaper model. Haiku 4.5 is
    available on regional endpoints and Opus 5 is not, so probing with Haiku
    would report a green tick on a pinned region and then fail on the first
    real request. Sixteen output tokens of Opus costs a fraction of a penny;
    a check that proves less than it appears to costs a great deal more.
    """
    import anthropic

    project = project or auth.project_id()
    region = region or configured_region()
    client = build_client(project=project, region=region)

    try:
        message = client.messages.create(
            model=MODEL_BEST,
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        )
    except anthropic.NotFoundError as e:
        if region not in SUPPORTED_REGIONS:
            # Not a Model Garden problem, and saying so would send somebody to
            # enable a model that is already enabled. Current models are simply
            # not served from single-region endpoints.
            raise UnsupportedRegion(region)
        raise ClaudeNotEnabled(project, str(e))
    except anthropic.PermissionDeniedError as e:
        text = str(e)
        if "model" in text.lower() or "publisher" in text.lower():
            raise ClaudeNotEnabled(project, text)
        raise

    return "".join(
        block.text for block in message.content if getattr(block, "type", "") == "text"
    ).strip()
