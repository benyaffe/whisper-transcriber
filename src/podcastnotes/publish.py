"""
Getting the finished write-up out of the app.

Deliberately not the Drive API. Publishing through Google would need an OAuth
client that only an administrator can create, and it would fail at the very
end of a long job for reasons the person running it cannot do anything about.
Google Docs reads Markdown from the clipboard, so the same result is three
steps with no credential, no scope and nobody to ask.

Degrades in the right order, because the failure that matters is somebody
spending an hour and having nothing to show for it:

  1. The Markdown file is written to disk first, always. Everything after
     this point is convenience.
  2. Copying to the clipboard can fail. The file is still there.
  3. Opening the browser can fail. The clipboard still holds the text, and
     the file is still there.

No Qt: the clipboard is reached through pbcopy so this stays importable from
anywhere, including a script with no QApplication.
"""

import os
import subprocess
import webbrowser
from dataclasses import dataclass

NEW_DOC_URL = "https://docs.new/"

PASTE_INSTRUCTIONS = (
    "In the new document, right-click and choose Paste from Markdown. "
    "If that item is missing, turn it on once under Tools, Preferences, "
    "Enable Markdown."
)


@dataclass
class PublishResult:
    """What actually happened, step by step, rather than one pass or fail."""

    markdown_path: str = ""
    copied: bool = False
    browser_opened: bool = False
    problems: tuple = ()

    @property
    def saved(self) -> bool:
        """The only part that really matters."""
        return bool(self.markdown_path) and os.path.exists(self.markdown_path)

    def summary(self) -> str:
        if not self.saved:
            return "The write-up could not be saved."
        where = os.path.basename(self.markdown_path)
        if self.copied and self.browser_opened:
            return f"Saved as {where}, copied, and a blank document is open. "
        if self.copied:
            return f"Saved as {where} and copied. {NEW_DOC_URL} opens a blank document."
        return f"Saved as {where}. Open it and copy the text yourself."


def write_markdown(text: str, path: str) -> str:
    """Write the document to disk. This happens before anything else can fail."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def copy_to_clipboard(text: str) -> bool:
    """Put the Markdown on the clipboard, or report that it could not."""
    try:
        process = subprocess.run(
            ["pbcopy"], input=text.encode("utf-8"), timeout=10, check=False
        )
        return process.returncode == 0
    except Exception:
        return False


def open_new_google_doc() -> bool:
    """Open a blank Google Doc in the browser."""
    try:
        return bool(webbrowser.open(NEW_DOC_URL))
    except Exception:
        return False


def publish(text: str, path: str, open_browser: bool = True) -> PublishResult:
    """Save, copy, and open a blank document, in that order.

    The order is the point. Saving first means a clipboard or browser problem
    costs a convenience rather than the work.
    """
    problems = []

    try:
        markdown_path = write_markdown(text, path)
    except OSError as e:
        return PublishResult(problems=(f"Could not write the file: {e}",))

    copied = copy_to_clipboard(text)
    if not copied:
        problems.append("Could not reach the clipboard, so copy from the file.")

    opened = False
    if open_browser:
        opened = open_new_google_doc()
        if not opened:
            problems.append(f"Could not open a browser. Go to {NEW_DOC_URL}.")

    return PublishResult(
        markdown_path=markdown_path,
        copied=copied,
        browser_opened=opened,
        problems=tuple(problems),
    )
