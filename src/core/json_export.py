"""
Machine-readable transcript export.

The VTT and TXT outputs are for people. This one is for the next program: it
keeps word-level timings, both diarization timelines, and the per-speaker voice
embeddings that let a later run recognise the same person on a different trip.

Everything here must survive json.dumps, which rules out the numpy scalars
pyannote hands back.
"""

import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional

SCHEMA_VERSION = 1

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _app_version() -> str:
    """Read the release version from the VERSION file.

    Same single source of truth build.sh and the PyInstaller spec read. In a
    frozen bundle the file is unpacked under sys._MEIPASS instead of the repo
    root, so try both. Never raises: an unknown version in a metadata block is
    not worth failing an export over.
    """
    candidates = [os.path.join(_REPO_ROOT, "VERSION")]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.insert(0, os.path.join(meipass, "VERSION"))

    for candidate in candidates:
        try:
            with open(candidate, encoding="utf-8") as f:
                version = f.read().strip()
            if version:
                return version
        except OSError:
            continue
    return "unknown"


def _turn_dicts(turns) -> list[dict]:
    return [
        {"start": round(t.start, 3), "end": round(t.end, 3), "speaker": t.speaker}
        for t in turns
    ]


def _speaker_entries(speaker_map: dict, diarization, turns) -> list[dict]:
    """One entry per speaker, carrying the label mapping and the voice profile.

    `pyannote_label` is not decoration. assign_speakers_to_segments numbers
    speakers by first appearance in the transcript, while embedding rows are
    ordered by pyannote's labels(). The two orders differ, so anything deriving
    an embedding index from the digit in "Speaker 3" gets the wrong person.
    """
    speech_by_label: dict[str, float] = {}
    for turn in turns:
        speech_by_label[turn.speaker] = (
            speech_by_label.get(turn.speaker, 0.0) + (turn.end - turn.start)
        )

    embeddings = diarization.embeddings if diarization else {}

    entries = []
    for raw_label, display in sorted(speaker_map.items(), key=lambda kv: kv[1]):
        entries.append(
            {
                "id": display,
                "pyannote_label": raw_label,
                "total_speech_s": round(speech_by_label.get(raw_label, 0.0), 3),
                # Absent rather than null when pyannote gave us nothing usable,
                # so a consumer cannot mistake a padded row for a real profile.
                **(
                    {"embedding": embeddings[raw_label]}
                    if raw_label in embeddings
                    else {}
                ),
            }
        )
    return entries


def build_payload(
    *,
    source_path: str,
    segments: list,
    duration: float,
    model: str,
    language: Optional[str],
    device: str,
    speaker_id_used: bool,
    speaker_map: Optional[dict] = None,
    diarization=None,
) -> dict:
    """Assemble the export. Pure: no file access beyond reading VERSION."""
    speaker_map = speaker_map or {}

    payload = {
        "version": SCHEMA_VERSION,
        "metadata": {
            "source_file": os.path.basename(source_path),
            "duration": round(duration, 3),
            "model": model,
            "language": language,
            "device": device,
            "app_version": _app_version(),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "speaker_id_used": speaker_id_used,
        },
        "segments": [
            {
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text,
                "confidence": round(s.confidence, 4),
                "speaker": s.speaker if speaker_id_used else None,
                "words": [
                    {
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "word": w.word,
                        "probability": round(w.probability, 4),
                    }
                    for w in s.words
                ],
            }
            for s in segments
        ],
    }

    if speaker_id_used and diarization is not None:
        payload["speakers"] = _speaker_entries(speaker_map, diarization, diarization.turns)
        payload["diarization"] = {
            "turns": _turn_dicts(diarization.turns),
            "exclusive_turns": _turn_dicts(diarization.exclusive_turns),
            "embedding_dimension": diarization.embedding_dimension,
            "speakers_without_profile": diarization.dropped_speakers,
        }

    return payload


def write_json(path: str, payload: dict) -> str:
    """Write the payload and return the path it went to."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    return path
