"""The trips this machine has worked on, so a half-finished one can be reopened.

`WriteUp.resume` has always worked. Nothing could reach it: it was only ever
called straight after a fresh transcription, so a trip closed part way through
was finished or abandoned, and there was no third option. Fifteen minutes of
Glean research sat on disk with no way back to it.

A trip's folder is wherever its recordings live, under `_podcastnotes/`, which
means there is no one directory to list. So the paths are recorded as they are
used. That is a cache and is treated as one: an entry whose folder has gone is
dropped on read rather than repaired, because the usual reason a trip folder
disappears is that somebody deleted the recordings on purpose.

Newest first, capped, and de-duplicated on the path, so reopening a trip moves
it to the top rather than adding a second row for it.

No Qt.
"""

import json
import os

from src.podcastnotes.project import STATE_FILENAME, TripProject
from src.podcastnotes.speaker_library import DEFAULT_DIR

FILENAME = "recent-trips.json"

# Enough to cover the trips somebody is actually between, and short enough that
# the list is a list rather than a search problem. Past this, Open... is the
# right tool.
LIMIT = 12


def path(where: str = "") -> str:
    return where or os.path.join(DEFAULT_DIR, FILENAME)


def remember(work_dir: str, where: str = "") -> list:
    """Record a trip as the most recent, and give back the list as it now is."""
    work_dir = os.path.abspath(work_dir or "")
    if not work_dir:
        return load(where)

    kept = [p for p in _read(where) if p != work_dir]
    _write([work_dir] + kept, where)
    return load(where)


def forget(work_dir: str, where: str = "") -> list:
    """Drop one trip from the list. The folder itself is left alone.

    Deleting somebody's recordings because they tidied a menu would be a
    spectacular overreach, so this only forgets.
    """
    work_dir = os.path.abspath(work_dir or "")
    _write([p for p in _read(where) if p != work_dir], where)
    return load(where)


def load(where: str = "") -> list:
    """The trips still on disk, newest first, each with enough to show a row.

    Returns dicts rather than TripProjects: this is for drawing a menu, and a
    trip whose state file is corrupt should still be listed with its folder
    name rather than taking the whole list down with it.
    """
    out = []
    for work_dir in _read(where):
        if not os.path.isfile(os.path.join(work_dir, STATE_FILENAME)):
            continue
        out.append({"work_dir": work_dir, **_describe(work_dir)})
    return out


def _describe(work_dir: str) -> dict:
    try:
        trip = TripProject.load(work_dir)
    except Exception:
        # A half-written state file, from a crash mid-save. The folder is still
        # openable and the failure belongs on the screen that opens it, not
        # here in the middle of drawing a list.
        return {"name": os.path.basename(work_dir), "created_at": "", "readable": False}
    return {
        "name": trip.name or os.path.basename(work_dir),
        "created_at": trip.created_at,
        "readable": True,
    }


def _read(where: str = "") -> list:
    try:
        with open(path(where), encoding="utf-8") as handle:
            saved = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(saved, list):
        return []
    return [str(p) for p in saved if isinstance(p, str) and p.strip()]


def _write(paths: list, where: str = ""):
    target = path(where)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(paths[:LIMIT], handle, indent=1)
