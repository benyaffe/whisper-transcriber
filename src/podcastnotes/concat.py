"""
Joining a trip's recordings into one file, and remembering where the seams are.

Two things matter here beyond "make one file out of several".

The first is not re-encoding when we do not have to. Phone voice memos from one
device share codec parameters, so ffmpeg's concat demuxer can copy the streams
straight through: instant, and lossless. Recordings from *different* devices do
not, and copying mismatched streams produces a file that is silently corrupt
after the first one. So the inputs get probed first.

The second is the boundary record. Once three recordings become one, "which
meeting was this said in" has no answer, and the summary stage needs it. That is
what boundaries.json is for.

No Qt, same as src/core/runner.py.
"""

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from src.utils.file_utils import get_bundled_binary, get_file_info

BOUNDARIES_FILENAME = "boundaries.json"

# What mismatched inputs get re-encoded to. Matches what the phone recordings
# already are, so the common path stays a stream copy and the fallback does not
# upsample anything.
REENCODE_CODEC = "aac"
REENCODE_SAMPLE_RATE = 48000
REENCODE_CHANNELS = 1
REENCODE_BITRATE = "96k"

# Joining AAC loses a fraction of a second of accounting per seam, because of
# frame alignment. Measured at roughly 0.3s per file on real recordings. Warn
# above this per file; it does not affect attributing a passage to a meeting,
# but accumulating it silently would.
DRIFT_TOLERANCE_PER_FILE = 1.0


class ConcatError(Exception):
    """The recordings could not be joined."""


@dataclass
class AudioParams:
    """The three things that have to match for a stream copy to be safe."""

    codec: str
    sample_rate: str
    channels: str

    @classmethod
    def probe(cls, path: str) -> "AudioParams":
        out = subprocess.run(
            [
                get_bundled_binary("ffprobe"), "-v", "quiet",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_name,sample_rate,channels",
                "-of", "default=nw=1:nk=1", path,
            ],
            capture_output=True, text=True, timeout=60,
        ).stdout.split()
        if len(out) < 3:
            raise ConcatError(f"No audio stream found in {os.path.basename(path)}")
        return cls(codec=out[0], sample_rate=out[1], channels=out[2])

    def describe(self) -> str:
        return f"{self.codec} {self.sample_rate}Hz {self.channels}ch"


@dataclass
class Boundary:
    """Where one recording sits inside the joined file."""

    source_name: str
    source_path: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ConcatResult:
    output_path: str
    boundaries: list[Boundary] = field(default_factory=list)
    stream_copied: bool = True
    drift: float = 0.0
    reencode_reason: str = ""

    @property
    def drift_is_acceptable(self) -> bool:
        allowed = DRIFT_TOLERANCE_PER_FILE * max(len(self.boundaries), 1)
        return abs(self.drift) <= allowed


def probe_compatibility(paths: list[str]) -> tuple[bool, str]:
    """Can these be joined without re-encoding?

    Returns (compatible, reason). The reason names the offending file, because
    "your recordings don't match" is not actionable and "Lakeside WP.m4a is
    44100Hz, the others are 48000Hz" is.
    """
    if not paths:
        raise ConcatError("nothing to join")
    if len(paths) == 1:
        return True, ""

    first = AudioParams.probe(paths[0])
    for path in paths[1:]:
        params = AudioParams.probe(path)
        if params != first:
            return False, (
                f"{os.path.basename(path)} is {params.describe()}, "
                f"but {os.path.basename(paths[0])} is {first.describe()}"
            )
    return True, ""


def _escape_for_list_file(path: str) -> str:
    r"""Quote a path for ffmpeg's concat list file.

    The format is `file '<path>'`, so a literal quote has to be closed,
    backslash-escaped and reopened. Voice memo filenames routinely contain
    apostrophes, and a newline would inject a whole extra directive.
    """
    if "\n" in path or "\r" in path:
        raise ConcatError(f"Newline in filename, cannot be joined: {path!r}")
    return "file '" + path.replace("'", r"'\''") + "'"


def _write_list_file(paths: list[str], list_path: str):
    with open(list_path, "w", encoding="utf-8") as f:
        for path in paths:
            f.write(_escape_for_list_file(os.path.abspath(path)) + "\n")


def compute_boundaries(paths: list[str]) -> list[Boundary]:
    """Where each recording will land in the joined file, back to back."""
    boundaries = []
    cursor = 0.0
    for path in paths:
        duration = float(get_file_info(path).get("duration") or 0.0)
        boundaries.append(
            Boundary(
                source_name=os.path.basename(path),
                source_path=os.path.abspath(path),
                start=round(cursor, 3),
                end=round(cursor + duration, 3),
            )
        )
        cursor += duration
    return boundaries


def write_boundaries(result: ConcatResult, out_dir: str) -> str:
    """Persist the seam record beside the joined audio.

    This is what makes the running order recoverable later. The recordings are
    joined in whatever order they were added, because the person adding them
    should not have to know, and because diarization wants one continuous file
    so a speaker keeps the same identity throughout.

    Each source's span in the combined audio is recorded here, so once there
    is a transcript it can be cut back into per-recording pieces and those
    pieces put in the right order by reading what was said. The audio is never
    rejoined; only the write-up is arranged.

    File timestamps are not a shortcut for this. On the Ashford recordings all
    three report a creation time within four seconds of each other, because
    that is when they were copied off the device.
    """
    path = os.path.join(out_dir, BOUNDARIES_FILENAME)
    payload = {
        "version": 1,
        "combined_audio": os.path.basename(result.output_path),
        "stream_copied": result.stream_copied,
        "reencode_reason": result.reencode_reason,
        "drift": round(result.drift, 3),
        "sources": [asdict(b) for b in result.boundaries],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    return path


def concat(
    paths: list[str],
    out_path: str,
    on_status: Optional[Callable[[str], None]] = None,
) -> ConcatResult:
    """Join recordings, in the order given, into out_path."""
    if not paths:
        raise ConcatError("nothing to join")
    say = on_status or (lambda _msg: None)

    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise ConcatError(
            "Cannot find: " + ", ".join(os.path.basename(p) for p in missing)
        )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    compatible, reason = probe_compatibility(paths)
    if compatible:
        say(f"[Joining {len(paths)} recordings without re-encoding]")
    else:
        say(f"[Recordings differ, re-encoding: {reason}]")

    boundaries = compute_boundaries(paths)
    list_path = out_path + ".concat.txt"
    _write_list_file(paths, list_path)

    cmd = [
        get_bundled_binary("ffmpeg"), "-nostdin", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", list_path,
        "-vn",
    ]
    if compatible:
        cmd += ["-c", "copy"]
    else:
        cmd += [
            "-c:a", REENCODE_CODEC,
            "-ar", str(REENCODE_SAMPLE_RATE),
            "-ac", str(REENCODE_CHANNELS),
            "-b:a", REENCODE_BITRATE,
        ]
    cmd.append(out_path)

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=3600, check=False)
    finally:
        # The list file is scratch; never leave it beside the output.
        if os.path.exists(list_path):
            os.remove(list_path)

    if proc.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        if os.path.exists(out_path):
            os.remove(out_path)
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise ConcatError(f"Could not join recordings: {stderr or 'no output'}")

    expected = boundaries[-1].end if boundaries else 0.0
    actual = float(get_file_info(out_path).get("duration") or 0.0)
    result = ConcatResult(
        output_path=out_path,
        boundaries=boundaries,
        stream_copied=compatible,
        drift=expected - actual,
        reencode_reason="" if compatible else reason,
    )

    if not result.drift_is_acceptable:
        say(
            f"[Warning: joined audio is {abs(result.drift):.1f}s off the sum of its "
            f"parts; meeting boundaries may be approximate]"
        )
    say(f"[Joined: {actual / 60:.0f} minutes from {len(paths)} recordings]")
    return result
