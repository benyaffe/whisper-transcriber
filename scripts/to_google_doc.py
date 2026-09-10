#!/usr/bin/env python3
"""
Put a Markdown file into a new Google Doc.

    venv/bin/python scripts/to_google_doc.py path/to/notes.md

Copies the file to the clipboard and opens a blank document. Then right-click
in the document and choose Paste from Markdown.

No Google credential is involved. This is the same code path the app uses, so
running it here proves the real thing rather than a rehearsal of it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.podcastnotes import publish


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"No such file: {path}")
        return 1

    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    if not publish.copy_to_clipboard(text):
        print("Could not reach the clipboard. Open the file and copy it yourself.")
        return 1
    print(f"Copied {len(text):,} characters.")

    if not publish.open_new_google_doc():
        print(f"Could not open a browser. Go to {publish.NEW_DOC_URL}")
        return 1

    print(f"Opened {publish.NEW_DOC_URL}")
    print()
    print(publish.PASTE_INSTRUCTIONS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
