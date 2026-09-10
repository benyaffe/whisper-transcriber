"""
A trip: several recordings from one outing, and everywhere the work goes.

The working folder sits beside the recordings, in `_podcastnotes/<slug>/`,
because that is already where this project's `.vtt` and `.txt` outputs land. A
trip therefore stays in one place on disk rather than being split between the
audio and some other directory.

**Nothing here ever writes to a source recording.** They are the only artefact
in the pipeline that cannot be regenerated. Everything derived goes in the
working folder.

No Qt, same as src/core/runner.py.
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.utils.file_utils import get_file_info

WORK_DIR_NAME = "_podcastnotes"
STATE_FILENAME = "project.json"
STATE_VERSION = 1


@dataclass
class SourceRecording:
    """One recording, and enough about it to notice if it goes missing.

    `path` is absolute, so moving a recording after starting a trip breaks the
    link. `name` and `duration` are kept so that when it does break we can say
    which recording is gone and how long it was, rather than printing a dead
    path and stopping.
    """

    path: str
    name: str
    duration: float = 0.0

    @classmethod
    def from_path(cls, path: str) -> "SourceRecording":
        absolute = os.path.abspath(path)
        info = get_file_info(absolute)
        return cls(
            path=absolute,
            name=os.path.basename(absolute),
            duration=float(info.get("duration") or 0.0),
        )

    @property
    def exists(self) -> bool:
        return os.path.isfile(self.path)


def slugify(name: str) -> str:
    """A filesystem-safe folder name. Voice-memo trips get named freely."""
    slug = re.sub(r"[^\w\s-]", "", name.strip().lower())
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug or "trip"


def _common_parent(paths: list[str]) -> str:
    """Where the recordings live.

    os.path.commonpath on a single file would give the file itself, and on
    recordings spread across sibling folders it gives their shared parent,
    which is the sensible place for the trip either way.
    """
    directories = [os.path.dirname(os.path.abspath(p)) for p in paths]
    if not directories:
        raise ValueError("a trip needs at least one recording")
    if len(directories) == 1:
        return directories[0]
    return os.path.commonpath(directories)


def resolve_work_dir(sources: list[str], slug: str, taken: Optional[set] = None) -> str:
    """`<where the recordings live>/_podcastnotes/<slug>/`, avoiding collisions.

    Two trips named the same thing in one folder get `-2`, `-3` and so on,
    rather than the second silently writing over the first.
    """
    base = os.path.join(_common_parent(sources), WORK_DIR_NAME)
    taken = taken if taken is not None else set()

    candidate = slug
    n = 1
    while os.path.exists(os.path.join(base, candidate)) or candidate in taken:
        n += 1
        candidate = f"{slug}-{n}"
    return os.path.join(base, candidate)


@dataclass
class TripProject:
    """One trip's worth of recordings and the work derived from them."""

    name: str
    description: str = ""
    sources: list[SourceRecording] = field(default_factory=list)
    work_dir: str = ""
    created_at: str = ""

    @classmethod
    def create(cls, name: str, source_paths: list[str], description: str = "") -> "TripProject":
        if not source_paths:
            raise ValueError("a trip needs at least one recording")

        sources = [SourceRecording.from_path(p) for p in source_paths]
        return cls(
            name=name,
            description=description,
            sources=sources,
            work_dir=resolve_work_dir(source_paths, slugify(name)),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    # --- paths ----------------------------------------------------------------

    @property
    def state_path(self) -> str:
        return os.path.join(self.work_dir, STATE_FILENAME)

    def path_for(self, filename: str) -> str:
        """A path inside the working folder. Never touches the recordings."""
        return os.path.join(self.work_dir, filename)

    @property
    def combined_audio_path(self) -> str:
        return self.path_for("combined.m4a")

    @property
    def total_duration(self) -> float:
        return sum(s.duration for s in self.sources)

    # --- integrity ------------------------------------------------------------

    def missing_sources(self) -> list[SourceRecording]:
        """Recordings that have moved or been deleted since the trip started."""
        return [s for s in self.sources if not s.exists]

    # --- persistence ----------------------------------------------------------

    def save(self) -> str:
        os.makedirs(self.work_dir, exist_ok=True)
        payload = {
            "version": STATE_VERSION,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "sources": [asdict(s) for s in self.sources],
        }
        # work_dir is deliberately not stored: it is where this file already is,
        # so a trip folder that gets moved still opens.
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        return self.state_path

    @classmethod
    def load(cls, work_dir: str) -> "TripProject":
        state = os.path.join(work_dir, STATE_FILENAME)
        with open(state, encoding="utf-8") as f:
            payload = json.load(f)

        return cls(
            name=payload["name"],
            description=payload.get("description", ""),
            sources=[SourceRecording(**s) for s in payload.get("sources", [])],
            work_dir=os.path.abspath(work_dir),
            created_at=payload.get("created_at", ""),
        )
