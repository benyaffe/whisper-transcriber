# Setting up PodcastNotesWT for an organisation

Two things need doing once, by somebody with admin rights, and they unblock
everybody. Neither is per-person. Until they exist, the app's setup screen
correctly reports that sign-in "has not been set up for your organisation yet"
and offers nobody a button, because there is nothing anyone can press.

Everything else a colleague needs is a sign-in or a single click.

---

## 1. A Google OAuth client

This is what lets the app sign people in from inside itself. Without it nobody
can use Claude or publish a document.

**It does not decide who pays for anything.** The OAuth client identifies the
*application*; each colleague separately chooses which Google Cloud project
their Claude usage bills to. So this client lives in one project of your
choosing and serves everyone, including colleagues on completely different
cost centers. That is the whole design.

### Steps

1. Pick any project in the Workspace organisation to host it. It does not
   matter which, and it will not be billed for model usage.
2. **APIs & Services → OAuth consent screen**
   - User type: **Internal**. This matters. Internal means the app is limited
     to your Workspace, which exempts it from Google's app verification review
     even though `cloud-platform` is a sensitive scope. Choosing External
     instead means a review that takes weeks.
   - App name: PodcastNotesWT
3. **Scopes** — add exactly these two:
   - `https://www.googleapis.com/auth/cloud-platform` — calling Claude on the
     user's own Vertex project
   - `https://www.googleapis.com/auth/drive.file` — creating the finished
     document. This is the narrow Drive scope: it grants access only to files
     the app itself created, not to anything else in anyone's Drive.
4. **Credentials → Create credentials → OAuth client ID**
   - Application type: **Desktop app**
5. Download the JSON.

### Where the file goes

Save it as `resources/google_oauth_client.json` in the app bundle, or set
`PODCASTNOTES_GOOGLE_CLIENT_FILE` to its path.

The JSON contains a "client secret". For an installed application Google does
not treat this as confidential — it ships inside every desktop app that does
browser sign-in, and the security comes from the loopback redirect rather than
from the secret. Shipping it in the bundle is the documented arrangement, not
a shortcut.

---

## 2. Glean access

Each colleague needs their own Glean credential. **Not a shared one.** Glean's
API is permission-aware, so a user's own credential returns only what that user
is allowed to see. A single shared token would let one person's trip surface
documents another person cannot open, and those names would then be published
into a shared Google Doc.

Two options, in order of preference:

1. **A Glean OAuth application for the organisation**, so colleagues sign in
   rather than pasting anything. This is the better end state.
2. **Personal API tokens.** Each colleague creates their own at
   `https://app.glean.com/admin/platform/tokenManagement` and pastes it into
   Settings. Works today, same permission scoping, just less pleasant.

The app supports option 2 now and is structured so option 1 drops in without
changing anything else.

---

## What each colleague does themselves

Nothing here needs admin help, and the app's setup screen walks through it.

| Step | Where | Notes |
|---|---|---|
| Sign in with Google | Setup screen, one button | Opens their browser |
| Choose their Cloud project | Settings, dropdown | Theirs, not yours. This is what their Claude usage bills to. |
| Enable Claude in that project | One click, once | The app deep-links to the right page for *their* project. See below. |
| Paste a Glean token | Settings | Until Glean OAuth exists |
| Paste a HuggingFace token | Settings | Free account, three model licences to accept |

### The one manual step that cannot be automated

Claude has to be enabled in each person's own Vertex project through Model
Garden, which means clicking Enable and accepting terms once. There is no API
for it.

The app detects this by making a cheap real call, and when it fails it links
straight to the Model Garden page for that person's project rather than
printing an error. They click Enable, come back, and press Check again.

Their account also needs the Vertex AI API enabled in that project and a role
that permits prediction (`roles/aiplatform.user` is sufficient). If they
already use Vertex for anything, both are already true.

### If your organisation has data residency rules

The app calls Claude through Google's global endpoint by default. If traffic
has to stay inside one geography, set the location to `us` or `eu`, which are
Google's multi-region endpoints.

Do not set a single region such as `us-east5`. Google serves only older models
from single-region endpoints, so pinning one makes the model this app depends
on unavailable. The app detects that case and says so, rather than reporting it
as a permissions problem.

---

## How to tell it worked

Open the app. The setup screen lists seven things and every one should be
green. It makes real calls rather than reading saved settings, so a green tick
means the thing genuinely works right now, not that a value is filled in.

| Row | Proved by |
|---|---|
| Audio tools | Running the bundled ffmpeg |
| Speech models | Checking the download cache |
| HuggingFace | Validating the token against HuggingFace |
| Google account | Refreshing the saved credential |
| Claude | One real, cheap completion |
| Google Docs | Creating a document and deleting it again |
| Glean | One real search that returns something the user can see |

A row that says "Waiting on ..." is not broken. It is downstream of something
that is, and it will resolve itself once the thing above it is green.
