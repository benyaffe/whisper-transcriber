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

from src.utils.file_utils import get_file_info, is_url

WORK_DIR_NAME = "_podcastnotes"
STATE_FILENAME = "project.json"
STATE_VERSION = 1

# Where a trip goes when every one of its sources is a URL and there is
# therefore nowhere on disk to sit beside.
REMOTE_TRIPS_DIR = os.path.join(
    os.path.expanduser("~"), "Documents", "PodcastNotesWT"
)

# How long a trip can be. Speaker identification holds the whole recording in
# memory at once, so cost grows with length until a laptop starts swapping.
WARN_MINUTES = 90
REFUSE_MINUTES = 180

# Measured, not guessed: diarizing the 42-minute Ashford recording peaked at
# 1750 MB above baseline, which is 41.7 MB per minute. Note that this is an
# order of magnitude more than the raw audio (a minute of 16kHz mono float32 is
# under 4 MB) because pyannote's segmentation and embedding tensors dominate.
# Estimating from the audio size alone understates it about elevenfold.
DIARIZATION_MB_PER_MINUTE = 42


@dataclass
class SourceRecording:
    """One recording, and enough about it to notice if it goes missing.

    For a local file `path` is absolute, so moving the recording after starting
    a trip breaks the link. `name` and `duration` are kept so that when it does
    break we can say which recording is gone and how long it was, rather than
    printing a dead path and stopping.

    A source can also be a URL that has not been fetched yet. Those have no
    duration and no local existence until the trip runs.
    """

    path: str
    name: str
    duration: float = 0.0
    remote: bool = False

    @classmethod
    def from_path(cls, path: str) -> "SourceRecording":
        if is_url(path):
            return cls.from_url(path)
        absolute = os.path.abspath(path)
        info = get_file_info(absolute)
        return cls(
            path=absolute,
            name=os.path.basename(absolute),
            duration=float(info.get("duration") or 0.0),
        )

    @classmethod
    def from_url(cls, url: str) -> "SourceRecording":
        """A recording that still has to be fetched.

        Deliberately does not go near os.path.abspath, which would turn
        "https://example.com/x" into "<cwd>/https:/example.com/x".
        """
        tail = url.rstrip("/").rsplit("/", 1)[-1] or url
        return cls(path=url, name=tail, duration=0.0, remote=True)

    @property
    def exists(self) -> bool:
        """A remote source cannot be checked without fetching it."""
        return True if self.remote else os.path.isfile(self.path)


def check_duration(total_seconds: float, estimated: bool = False) -> tuple[bool, str]:
    """Is a trip of this length sensible? Returns (allowed, what to tell them).

    Both thresholds produce a message; only the upper one blocks. An empty
    message means there is nothing worth saying.
    """
    minutes = total_seconds / 60
    if minutes <= 0:
        return True, ""

    maybe = "at least " if estimated else ""
    needs_gb = minutes * DIARIZATION_MB_PER_MINUTE / 1024

    if minutes > REFUSE_MINUTES:
        return False, (
            f"That is {maybe}{minutes:.0f} minutes of audio, over the "
            f"{REFUSE_MINUTES}-minute limit. Working out who is speaking would need "
            f"around {needs_gb:.0f}GB of memory for a recording this long. Split the "
            f"trip into shorter ones, or untick Multiple speakers."
        )
    if minutes > WARN_MINUTES:
        return True, (
            f"That is {maybe}{minutes:.0f} minutes of audio. It will work, but "
            f"working out who is speaking will need around {needs_gb:.0f}GB of memory "
            f"and {_free_gb():.0f}GB is free right now."
        )
    return True, ""


def _free_gb() -> float:
    """Memory available now. Never raises; a warning is not worth failing over."""
    try:
        import psutil

        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        return 0.0


def slugify(name: str) -> str:
    """A filesystem-safe folder name. Voice-memo trips get named freely."""
    slug = re.sub(r"[^\w\s-]", "", name.strip().lower())
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug or "trip"


def _common_parent(paths: list[str]) -> str:
    """Where the recordings live.

    URLs are skipped: they are nowhere on disk. A trip made entirely of URLs
    has no natural home beside its sources, so it falls back to
    ~/Documents/PodcastNotesWT.

    os.path.commonpath on a single file would give the file itself, and on
    recordings spread across sibling folders it gives their shared parent,
    which is the sensible place for the trip either way.
    """
    local = [p for p in paths if not is_url(p)]
    if not paths:
        raise ValueError("a trip needs at least one recording")
    if not local:
        return REMOTE_TRIPS_DIR

    directories = [os.path.dirname(os.path.abspath(p)) for p in local]
    if len(directories) == 1:
        return directories[0]
    return os.path.commonpath(directories)


def resolve_work_dir(sources: list[str], slug: str, taken: Optional[set] = None) -> str:
    """`<where the recordings live>/_podcastnotes/<slug>/`, avoiding collisions.

    Two trips named the same thing in one folder get `-2`, `-3` and so on,
    rather than the second silently writing over the first.
    """
    parent = _common_parent(sources)
    # A trip of URLs already lives in a dedicated folder; nesting _podcastnotes
    # inside it would just add a pointless level.
    base = parent if parent == REMOTE_TRIPS_DIR else os.path.join(parent, WORK_DIR_NAME)
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

    @property
    def duration_is_estimated(self) -> bool:
        """True when a source has not been fetched yet, so the total is a floor.

        A URL's length is unknown until it is downloaded, so a trip containing
        one could turn out longer than it looks.
        """
        return any(s.remote for s in self.sources)

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
