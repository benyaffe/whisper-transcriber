"""
Signing in to Glean with nothing to paste and nobody to ask.

Glean publishes a `registration_endpoint`, which means the app can register
itself at runtime and then run an ordinary browser sign-in. No administrator
creates anything, no personal token is pasted, and the credential is the
person's own so the API stays permission-aware.

This is how the Glean MCP server already authenticates on this machine: it
registered a client called "MCP CLI Proxy" for itself and cached the tokens.
The same mechanism is open to us.

`run_sign_in` blocks until the browser comes back. Call it off the GUI thread.

No Qt.
"""

import base64
import hashlib
import json
import os
import secrets
import threading
import urllib.parse
import urllib.request

from src.core.config import KEYRING_SERVICE

CLIENT_NAME = "PodcastNotesWT"

# Only what the app actually does. "search" reads the index and "offline_access"
# is what makes Glean issue a refresh token, without which the sign-in lasts an
# hour. Nothing here grants admin, chat, or write access to anything.
SCOPES = ["search", "documents", "people", "offline_access"]

KEYRING_GLEAN_REFRESH = "glean_refresh_token"
KEYRING_GLEAN_CLIENT = "glean_client_id"

DISCOVERY_PATH = "/.well-known/oauth-authorization-server"
HTTP_TIMEOUT = 20


class SignInFailed(Exception):
    """The browser round trip did not produce a usable credential."""


class SignInCancelled(SignInFailed):
    """The person gave up waiting, which is not a failure to report as one.

    A subclass, so anything catching SignInFailed still catches this, but the
    screen can tell "you stopped" from "something went wrong" and stay quiet
    about the first.
    """


def _wait_for(done, cancel, timeout: float) -> bool:
    """Wait for the browser, or for the person to give up, whichever first.

    Polled rather than a single blocking wait, because two Events cannot be
    waited on together without a third to signal both. A quarter of a second
    is imperceptible against a sign-in and costs nothing over ten minutes.
    """
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if done.wait(0.25):
            return True
        if cancel is not None and cancel.is_set():
            return False
    return done.is_set()


def _base_url(instance: str) -> str:
    return f"https://{instance}-be.glean.com"


def discover(instance: str) -> dict:
    """Ask Glean where its OAuth endpoints are, rather than assuming.

    The paths are conventional but not guaranteed, and an instance on a
    different Glean release would move them.
    """
    url = _base_url(instance) + DISCOVERY_PATH
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as response:
        metadata = json.load(response)

    for required in ("authorization_endpoint", "token_endpoint"):
        if required not in metadata:
            raise SignInFailed(f"Glean did not advertise a {required}.")
    return metadata


def supports_self_registration(metadata: dict) -> bool:
    """Whether this instance lets the app register itself.

    When it does not, somebody has to paste a personal token instead, so the
    checklist needs to know which of the two it is asking for.
    """
    return bool(metadata.get("registration_endpoint"))


def register(metadata: dict, redirect_uri: str) -> str:
    """Register this app with Glean and return the client id.

    A public client: no secret, PKCE instead. That is the correct shape for
    something running on somebody's laptop, where a secret would not be one.
    """
    endpoint = metadata["registration_endpoint"]
    body = json.dumps(
        {
            "client_name": CLIENT_NAME,
            "redirect_uris": [redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
            "scope": " ".join(SCOPES),
        }
    ).encode()

    request = urllib.request.Request(
        endpoint, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
        registration = json.load(response)

    client_id = registration.get("client_id")
    if not client_id:
        raise SignInFailed("Glean registered the app but returned no client id.")
    return client_id


# --- the browser round trip ---------------------------------------------------


def _pkce_pair() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(os.urandom(64)).decode().rstrip("=")
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .decode()
        .rstrip("=")
    )
    return verifier, challenge


CALLBACK_PATH = "/oauth/callback"


def _build_callback_server(holder: dict, done: threading.Event):
    """A loopback server that waits for the redirect and ignores everything else.

    It stays up until the callback actually arrives. An earlier version served
    exactly one request and then closed, which failed in the field: browsers
    open speculative connections and ask for /favicon.ico, so the single
    request was routinely spent on something that was not the callback and the
    real redirect got "connection refused" with the authorisation code sitting
    in the address bar.
    """
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != CALLBACK_PATH:
                self.send_error(404)
                return

            query = urllib.parse.parse_qs(parsed.query)
            holder["code"] = query.get("code", [""])[0]
            holder["state"] = query.get("state", [""])[0]
            holder["error"] = query.get("error", [""])[0]

            body = (
                b"<html><body style='font-family:system-ui;padding:3em'>"
                b"<h2>Signed in to Glean.</h2>"
                b"<p>You can close this tab and go back to the app.</p>"
                b"</body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            done.set()

        def log_message(self, *args):
            pass  # the default implementation writes to stderr

    # Port 0 lets the OS choose, so a busy port cannot wedge the sign-in.
    return HTTPServer(("127.0.0.1", 0), Handler)


def run_sign_in(instance: str, open_browser: bool = True, timeout: int = 600,
                on_url=None, cancel: "threading.Event" = None) -> str:
    """Register, sign in through the browser, and save the result.

    Returns the account's refresh token. Blocks until the browser comes back or
    the timeout expires, so it does not belong on the GUI thread.

    Ten minutes rather than three. Signing in can mean a redirect through an
    identity provider and a prompt on a phone, and a timeout that expires
    mid-approval closes the port while the authorisation code is still in
    flight, which produces "connection refused" at the worst possible moment.

    `on_url` is handed the authorisation address before the wait begins. Ten
    minutes is a long time to be stuck, and the way people get stuck is the
    browser opening on the wrong profile: without the address there is nothing
    to paste into the right one, and the only way out is to wait it out.

    `cancel` is an Event that ends the wait early, so somebody who has given up
    can say so instead of staring at a screen that will not change for ten
    minutes.
    """
    import webbrowser

    metadata = discover(instance)
    if not supports_self_registration(metadata):
        raise SignInFailed(
            "This Glean instance does not let applications register themselves."
        )

    holder: dict = {}
    done = threading.Event()
    server = _build_callback_server(holder, done)
    redirect_uri = f"http://127.0.0.1:{server.server_port}{CALLBACK_PATH}"

    client_id = register(metadata, redirect_uri)
    verifier, challenge = _pkce_pair()
    state = secrets.token_urlsafe(24)

    url = metadata["authorization_endpoint"] + "?" + urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(SCOPES),
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
    )

    listener = threading.Thread(target=server.serve_forever, daemon=True)
    listener.start()
    try:
        if on_url:
            on_url(url)
        if open_browser:
            webbrowser.open(url)
        _wait_for(done, cancel, timeout)
    finally:
        server.shutdown()
        server.server_close()

    if cancel is not None and cancel.is_set() and not done.is_set():
        raise SignInCancelled("Sign-in cancelled.")

    if holder.get("error"):
        raise SignInFailed(f"Glean refused the sign-in: {holder['error']}")
    if not holder.get("code"):
        raise SignInFailed("The sign-in did not finish.")
    # Without this a malicious page could feed us somebody else's code.
    if holder.get("state") != state:
        raise SignInFailed("The sign-in came back mismatched and was discarded.")

    tokens = _exchange(
        metadata["token_endpoint"],
        {
            "grant_type": "authorization_code",
            "code": holder["code"],
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "code_verifier": verifier,
        },
    )

    refresh_token = tokens.get("refresh_token", "")
    if not refresh_token:
        raise SignInFailed(
            "Glean did not issue a refresh token, so the sign-in would expire "
            "within the hour."
        )

    save_client_id(client_id)
    save_refresh_token(refresh_token)
    return refresh_token


def _exchange(token_endpoint: str, form: dict) -> dict:
    body = urllib.parse.urlencode(form).encode()
    request = urllib.request.Request(
        token_endpoint,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:200]
        raise SignInFailed(f"Glean rejected the request ({e.code}): {detail}")


def access_token(instance: str) -> str:
    """A usable access token, minted from the saved refresh token.

    Called at request time rather than cached, because an access token is good
    for about an hour and a cached one is the reason things stop working in the
    middle of a long job.
    """
    refresh_token = get_refresh_token()
    client_id = get_client_id()
    if not refresh_token or not client_id:
        return ""

    metadata = discover(instance)
    tokens = _exchange(
        metadata["token_endpoint"],
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        },
    )
    # Glean may rotate the refresh token on use. Dropping the new one logs the
    # person out at some unpredictable point later.
    if tokens.get("refresh_token"):
        save_refresh_token(tokens["refresh_token"])
    return tokens.get("access_token", "")


# --- what is remembered -------------------------------------------------------


def _get(key: str) -> str:
    import keyring

    try:
        return keyring.get_password(KEYRING_SERVICE, key) or ""
    except Exception:
        return ""


def _set(key: str, value: str):
    import keyring

    value = (value or "").strip()
    try:
        if value:
            keyring.set_password(KEYRING_SERVICE, key, value)
        else:
            try:
                keyring.delete_password(KEYRING_SERVICE, key)
            except keyring.errors.PasswordDeleteError:
                pass
    except Exception as e:
        raise RuntimeError(f"Could not save the Glean sign-in: {e}")


def get_refresh_token() -> str:
    return _get(KEYRING_GLEAN_REFRESH)


def save_refresh_token(token: str):
    _set(KEYRING_GLEAN_REFRESH, token)


def get_client_id() -> str:
    return _get(KEYRING_GLEAN_CLIENT)


def save_client_id(client_id: str):
    _set(KEYRING_GLEAN_CLIENT, client_id)


def signed_in() -> bool:
    return bool(get_refresh_token() and get_client_id())


def sign_out():
    save_refresh_token("")
    save_client_id("")
