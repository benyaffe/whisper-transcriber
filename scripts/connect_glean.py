#!/usr/bin/env python3
"""
Connect this machine to Glean, and prove it worked.

    venv/bin/python scripts/connect_glean.py acme

Opens a browser, you sign in the way you always do, and it runs a real search
to show the connection is live. Nothing to paste, nothing to ask anybody for.

Exists because the app is not built into a clickable bundle yet. Once it is,
this is the Sign in button on the setup screen and this script goes away.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.podcastnotes import glean, glean_auth


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Which Glean? Pass the bit before .glean.com in your browser's "
              "address bar.")
        return 2

    instance = sys.argv[1].strip()
    glean.set_instance(instance)
    instance = glean.instance()
    print(f"Glean instance: {instance}")

    print("Checking what this instance allows...", flush=True)
    metadata = glean_auth.discover(instance)
    if not glean_auth.supports_self_registration(metadata):
        print("\nThis instance does not let applications register themselves.")
        print("A personal API token is needed instead. Stopping here rather "
              "than pretending otherwise.")
        return 1
    print("  Self-registration is allowed, so no administrator is involved.")

    print("\nOpening your browser. Sign in there, then come back.", flush=True)
    glean_auth.run_sign_in(instance)
    print("  Signed in.")

    print("\nProving it works with a real search...", flush=True)
    results = glean.search("meeting", page_size=3)
    if not results:
        print("  Signed in, but the search found nothing you can see.")
        print("  That is a permissions problem rather than a connection one.")
        return 1

    print(f"  {len(results)} result(s), so the connection is live:")
    for result in results:
        print(f"    - {result['title'][:70]}  [{result['source']}]")

    print("\nDone. The app will use this from now on; you will not be asked again.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
