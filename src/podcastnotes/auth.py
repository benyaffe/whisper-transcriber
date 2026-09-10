"""
Signing in to Google, once, for two things at the same time.

The same consent screen carries `cloud-platform` (so Claude can be called on
the user's own Vertex project) and `drive.file` (so the finished document can
be published). One sign-in, two capabilities, and critically **no gcloud**:
nobody installs a 500MB SDK or opens a Terminal.

Each colleague sits on a different Vertex project because they are on different
cost centers, so the project is a per-user setting rather than a constant. That
is the whole reason this module exists instead of a hardcoded project id.

`run_sign_in` blocks until the browser comes back. Call it off the GUI thread.

No Qt. Heavy imports stay inside the functions, matching src/core/config.py.
"""

import os

from src.core.config import KEYRING_SERVICE

# Both scopes on one consent screen. cloud-platform is broad and sensitive, but
# the OAuth client is Internal to the Workspace, which exempts it from Google's
# verification review. drive.file is deliberately narrow: it grants access only
# to files this app itself created, so it can publish without being able to
# read anything else in the user's Drive.
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/drive.file",
]

TOKEN_URI = "https://oauth2.googleapis.com/token"

KEYRING_GOOGLE_REFRESH = "google_refresh_token"
SETTINGS_GOOGLE_PROJECT = "google_project_id"
SETTINGS_GOOGLE_EMAIL = "google_account_email"

# Where an admin drops the OAuth client. For an installed app Google does not
# treat the "secret" as confidential, so shipping it in the bundle is the
# documented arrangement rather than a shortcut.
CLIENT_FILE_ENV = "PODCASTNOTES_GOOGLE_CLIENT_FILE"
CLIENT_ID_ENV = "PODCASTNOTES_GOOGLE_CLIENT_ID"
CLIENT_SECRET_ENV = "PODCASTNOTES_GOOGLE_CLIENT_SECRET"
BUNDLED_CLIENT_FILE = "google_oauth_client.json"

CONSOLE_CREDENTIALS_URL = "https://console.cloud.google.com/apis/credentials"


class NotConfigured(Exception):
    """No OAuth client is present, so signing in is not yet possible.

    Distinct from a failed sign-in. This one is somebody else's job: an admin
    has to create the client once, for everybody. Saying "sign-in failed" here
    would send the user to retry something that cannot work.
    """


def client_config() -> dict:
    """The OAuth client, from the environment in development or the bundle.

    Raises NotConfigured when there is none, which is the state of the world
    until an admin creates it.
    """
    import json

    client_id = os.environ.get(CLIENT_ID_ENV, "").strip()
    client_secret = os.environ.get(CLIENT_SECRET_ENV, "").strip()
    if client_id and client_secret:
        return {"client_id": client_id, "client_secret": client_secret}

    path = os.environ.get(CLIENT_FILE_ENV, "").strip() or _bundled_client_path()
    if not path or not os.path.exists(path):
        raise NotConfigured(
            "No Google sign-in has been set up for this organisation yet."
        )

    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, ValueError) as e:
        raise NotConfigured(f"The Google sign-in file could not be read: {e}")

    # Google hands these out wrapped in "installed" or "web".
    body = data.get("installed") or data.get("web") or data
    if not body.get("client_id") or not body.get("client_secret"):
        raise NotConfigured("The Google sign-in file is missing its client id.")
    return {"client_id": body["client_id"], "client_secret": body["client_secret"]}


def is_configured() -> bool:
    try:
        client_config()
        return True
    except NotConfigured:
        return False


def _bundled_client_path() -> str:
    from src.utils.file_utils import get_resource_path

    try:
        return get_resource_path(BUNDLED_CLIENT_FILE)
    except Exception:
        return ""


# --- signing in ---------------------------------------------------------------


def run_sign_in(open_browser: bool = True):
    """Loopback OAuth in the user's own browser. Blocks until they come back.

    Returns the credentials and stores the refresh token in the Keychain, under
    the same service as the HuggingFace token so there is one place to revoke.
    """
    from google_auth_oauthlib.flow import InstalledAppFlow

    config = client_config()
    flow = InstalledAppFlow.from_client_config(
        {
            "installed": {
                "client_id": config["client_id"],
                "client_secret": config["client_secret"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": TOKEN_URI,
            }
        },
        scopes=GOOGLE_SCOPES,
    )
    # port=0 lets the OS pick, so two people on one machine, or a port already
    # taken, does not wedge the sign-in.
    credentials = flow.run_local_server(
        port=0,
        open_browser=open_browser,
        # Without these Google returns no refresh token on a repeat sign-in,
        # and the app silently stops working an hour later.
        access_type="offline",
        prompt="consent",
        success_message="Signed in. You can close this tab and go back to the app.",
    )
    save_refresh_token(credentials.refresh_token or "")
    _remember_account(credentials)
    return credentials


SOURCE_APP = "app"
SOURCE_GCLOUD = "gcloud"


def stored_credentials():
    """Whatever Google credentials this machine has, or None.

    The app's own sign-in first, because it is the one that covers Drive as
    well. Application Default Credentials second, which is what somebody with
    gcloud already has: it carries cloud-platform and therefore Claude, but
    not drive.file, so publishing still needs the app's own sign-in.

    The access token is deliberately not cached. Both the Anthropic client and
    google-api-python-client refresh on demand, and a token in the Keychain
    would be stale within the hour.
    """
    from google.oauth2.credentials import Credentials

    refresh_token = get_refresh_token()
    if refresh_token:
        config = client_config()
        return Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri=TOKEN_URI,
            client_id=config["client_id"],
            client_secret=config["client_secret"],
            scopes=GOOGLE_SCOPES,
        )

    return application_default_credentials()


def application_default_credentials():
    """What `gcloud auth application-default login` leaves behind, if anything.

    Nobody is asked to install gcloud. This exists because people who already
    have it should not have to sign in twice, and because it makes the Claude
    path testable before the OAuth client exists.
    """
    try:
        import google.auth

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        return credentials
    except Exception:
        return None


def credentials_source() -> str:
    """Which of the two is in use, since they do not grant the same things."""
    if get_refresh_token():
        return SOURCE_APP
    if application_default_credentials() is not None:
        return SOURCE_GCLOUD
    return ""


def covers_drive() -> bool:
    """Whether the current credentials can create a document.

    gcloud's default credentials carry cloud-platform but not drive.file, so
    Claude works and publishing does not. Saying so up front beats discovering
    it after a trip has finished running.
    """
    return credentials_source() == SOURCE_APP


def ensure_fresh(credentials):
    """Mint an access token if there is not a usable one."""
    from google.auth.transport.requests import Request

    if not credentials.token or credentials.expired:
        credentials.refresh(Request())
    return credentials


def sign_out():
    save_refresh_token("")
    _settings().remove(SETTINGS_GOOGLE_EMAIL)


# --- what is remembered -------------------------------------------------------


def get_refresh_token() -> str:
    import keyring

    try:
        return keyring.get_password(KEYRING_SERVICE, KEYRING_GOOGLE_REFRESH) or ""
    except Exception:
        return ""


def save_refresh_token(token: str):
    import keyring

    try:
        if token:
            keyring.set_password(KEYRING_SERVICE, KEYRING_GOOGLE_REFRESH, token)
        else:
            try:
                keyring.delete_password(KEYRING_SERVICE, KEYRING_GOOGLE_REFRESH)
            except keyring.errors.PasswordDeleteError:
                pass
    except Exception as e:
        raise RuntimeError(f"Could not save the Google sign-in: {e}")


def _settings():
    from src.core.config import _settings as settings

    return settings()


def account_email() -> str:
    return _settings().value(SETTINGS_GOOGLE_EMAIL, "", type=str)


def _remember_account(credentials):
    """Record which account signed in, so the list can show it.

    Best effort. Failing to learn the address is not a failed sign-in.
    """
    try:
        from google.oauth2 import id_token
        from google.auth.transport.requests import Request

        info = id_token.verify_oauth2_token(
            credentials.id_token, Request(), audience=None
        )
        email = info.get("email", "")
    except Exception:
        email = ""
    if email:
        _settings().setValue(SETTINGS_GOOGLE_EMAIL, email)


def project_id() -> str:
    return _settings().value(SETTINGS_GOOGLE_PROJECT, "", type=str)


def set_project_id(value: str):
    _settings().setValue(SETTINGS_GOOGLE_PROJECT, value or "")


def default_project() -> str:
    """The project gcloud is already pointed at, if any.

    Worth having on its own because listing every project needs the Cloud
    Resource Manager API enabled, which it often is not, while this needs
    nothing. Somebody who already uses Vertex is almost always pointed at the
    project they want.
    """
    try:
        import google.auth

        _, project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        return project or ""
    except Exception:
        return ""


class ProjectListUnavailable(Exception):
    """The project list cannot be read, which is not the same as having none.

    Cloud Resource Manager is frequently left disabled. Reporting that as "you
    have no projects" would be wrong and would send somebody to ask for access
    they already have.
    """


def list_projects(credentials) -> list[dict]:
    """The projects this person can use, for the picker.

    Returns dicts of id and name, active projects only, so the picker does not
    offer something that will fail on first use.
    """
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    service = build(
        "cloudresourcemanager", "v1", credentials=credentials, cache_discovery=False
    )
    projects, request = [], service.projects().list()
    while request is not None:
        try:
            response = request.execute()
        except HttpError as e:
            if e.status_code in (403, 404):
                raise ProjectListUnavailable(str(e))
            raise
        for project in response.get("projects", []):
            if project.get("lifecycleState") != "ACTIVE":
                continue
            projects.append(
                {
                    "id": project["projectId"],
                    "name": project.get("name", project["projectId"]),
                }
            )
        request = service.projects().list_next(request, response)
    return sorted(projects, key=lambda p: p["name"].lower())
