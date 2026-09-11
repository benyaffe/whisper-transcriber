"""
Remembering voices between trips, so the second one pre-fills the first's names.

Naming four speakers is a small chore once and an irritating one every week,
and the same handful of colleagues are on most trips. This keeps the confirmed
voice profiles so the attribution screen can arrive already filled in.

**It suggests. It never decides.** A match pre-fills a name that a person then
confirms, exactly as an unmatched voice would be confirmed. The failure this
avoids is a wrong name applied silently on the strength of a cosine score,
which is the one mistake in this pipeline that nothing downstream can catch.

## The threshold, measured rather than chosen

The original plan proposed 0.75, which would have matched nobody. Measured on
the real Ashford recording, where the diarizer split one person across two
labels and so provided a known same-person pair:

    same person   (Anya vs the split label)      0.624
    different people                     0.029, 0.029, 0.038,
                                         0.078, 0.129, 0.244

That is a gap of 0.38 between the highest different-person score and the only
same-person score, so the boundary sits comfortably in between.

One caveat is deliberately allowed for. The same-person pair comes from within
a single recording, sharing a microphone, a room and a session, so it is
almost certainly an optimistic figure for matching the same person across two
trips recorded months apart. Rather than pick one threshold and hope, there
are two: above MATCH the name is pre-filled, and between CONSIDER and MATCH
the person is offered as a candidate without being filled in. A cross-session
score that has degraded still surfaces the right name for a human to accept.

## Several embeddings per person, matched best-of

Not a running average. An average of a person recorded in a quiet room and in
a busy ED is a vector describing neither, and it gets worse with every trip
added. Keeping the samples and taking the best match means a new recording
only has to resemble one previous occasion.

No Qt. No network.
"""

import json
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

# Application Support rather than Documents: this is app state, not something
# anybody wants to see in a folder of their own files. It is one file so that
# deleting it is a complete and obvious way to start again.
DEFAULT_DIR = os.path.join(
    os.path.expanduser("~"), "Library", "Application Support", "PodcastNotesWT"
)
DB_NAME = "speakers.db"

# Pre-fill the name. Comfortably above the highest different-person score
# measured on Ashford (0.244) and below the same-person score (0.624).
MATCH = 0.45

# Offer as a candidate without filling it in. Sits above the measured
# different-person range with a margin, and catches a same-person match
# degraded by a different room, microphone or session.
CONSIDER = 0.30

SCHEMA = """
CREATE TABLE IF NOT EXISTS voices (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    trip        TEXT    NOT NULL DEFAULT '',
    recorded_at TEXT    NOT NULL,
    dimension   INTEGER NOT NULL,
    embedding   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS voices_by_name ON voices(name);
"""


@dataclass
class Match:
    """One remembered person, and how well this voice matches them."""

    name: str
    score: float
    samples: int

    @property
    def confident(self) -> bool:
        """Whether to pre-fill the name rather than merely offer it."""
        return self.score >= MATCH


def cosine(a, b) -> float:
    """Similarity of two embeddings, 0.0 when either has no magnitude.

    A zero vector has no direction to compare, and the usual formula divides
    by zero for it. pyannote does occasionally return one for a speaker it
    heard for a fraction of a second.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    left = math.sqrt(sum(x * x for x in a))
    right = math.sqrt(sum(y * y for y in b))
    if not left or not right:
        return 0.0
    return dot / (left * right)


class Library:
    """The stored voices. Cheap to open, so callers need not hold one open."""

    def __init__(self, path: str = ""):
        self.path = path or os.path.join(DEFAULT_DIR, DB_NAME)
        if self.path != ":memory:":
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.executescript(SCHEMA)
        self._db.commit()

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def remember(self, name: str, embedding, trip: str = "", when=None) -> bool:
        """Store one confirmed voice sample. Returns whether it was stored.

        An empty name or an embedding with no magnitude is refused rather than
        stored, because both would sit in the library matching things
        arbitrarily well or not at all, and neither can be noticed later.
        """
        name = (name or "").strip()
        values = [float(x) for x in (embedding or [])]
        if not name or not values or not any(values):
            return False

        stamp = (when or datetime.now(timezone.utc)).isoformat()
        self._db.execute(
            "INSERT INTO voices (name, trip, recorded_at, dimension, embedding) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, trip or "", stamp, len(values), json.dumps(values)),
        )
        self._db.commit()
        return True

    def identify(self, embedding, limit: int = 3) -> list:
        """Who this voice might be, best match per person, strongest first.

        Only samples of the same dimension are considered. A different
        diarization model produces vectors of a different width whose
        similarity to these is not merely wrong but meaningless, and comparing
        them would return a confident number with nothing behind it.
        """
        values = [float(x) for x in (embedding or [])]
        if not values or not any(values):
            return []

        rows = self._db.execute(
            "SELECT name, embedding FROM voices WHERE dimension = ?", (len(values),)
        ).fetchall()

        best = {}
        counts = {}
        for name, stored in rows:
            counts[name] = counts.get(name, 0) + 1
            score = cosine(values, json.loads(stored))
            # Best-of, never a mean: see the module docstring.
            if score > best.get(name, -1.0):
                best[name] = score

        found = [
            Match(name=name, score=round(score, 4), samples=counts[name])
            for name, score in best.items()
            if score >= CONSIDER
        ]
        found.sort(key=lambda m: m.score, reverse=True)
        return found[:limit]

    def names(self) -> list:
        """Everyone remembered, with how many samples each has."""
        rows = self._db.execute(
            "SELECT name, COUNT(*) FROM voices GROUP BY name ORDER BY name"
        ).fetchall()
        return [(name, count) for name, count in rows]

    def forget(self, name: str) -> int:
        """Remove every sample of one person. Returns how many were removed."""
        cursor = self._db.execute("DELETE FROM voices WHERE name = ?", ((name or "").strip(),))
        self._db.commit()
        return cursor.rowcount


def remember_trip(payload: dict, names: dict, trip: str = "", path: str = "") -> int:
    """Store the confirmed speakers of one trip. Returns how many were stored.

    Takes the same decisions dict the attribution stage applies, so only
    voices a person actually confirmed are remembered. Labels left unconfirmed
    are exactly the ones whose identity is uncertain, and remembering those
    would teach the library a wrong name that it then suggests forever.
    """
    profiles = {
        p.get("id"): p for p in (payload.get("speakers") or []) if p.get("embedding")
    }
    stored = 0
    with Library(path) as library:
        for label, name in (names or {}).items():
            profile = profiles.get(label)
            if profile and library.remember(name, profile.get("embedding"), trip):
                stored += 1
    return stored


def suggest_names(payload: dict, path: str = "") -> dict:
    """The library's opinion on each speaker label in a trip.

    Returns a label-to-matches mapping for the attribution screen to pre-fill
    from. Nothing is applied; that still needs a person.
    """
    out = {}
    with Library(path) as library:
        for profile in payload.get("speakers") or []:
            label = profile.get("id")
            if not label:
                continue
            out[label] = library.identify(profile.get("embedding"))
    return out
