"""
Checkpoint/resume support for long transcriptions.
Saves progress periodically to allow resuming after crashes.
"""

import os
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List
from datetime import datetime


CHECKPOINT_DIR = Path.home() / ".cache" / "whisper_transcriber" / "checkpoints"


@dataclass
class TranscriptionCheckpoint:
    """Checkpoint data for a transcription in progress."""
    file_hash: str  # Hash of source file for verification
    source_path: str
    model_size: str
    language: Optional[str]
    last_segment_end: float  # Timestamp of last completed segment
    segments_json: str  # JSON serialized segments
    created_at: str
    updated_at: str


def _file_hash(filepath: str) -> str:
    """Calculate a hash of the first 1MB of a file for identification."""
    h = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            chunk = f.read(1024 * 1024)  # First 1MB
            h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return "unknown"


def _checkpoint_path(file_hash: str) -> Path:
    """Get checkpoint file path for a given file hash."""
    return CHECKPOINT_DIR / f"checkpoint_{file_hash}.json"


def save_checkpoint(
    source_path: str,
    model_size: str,
    language: Optional[str],
    segments: List[dict],
    last_segment_end: float
):
    """
    Save a transcription checkpoint.

    Args:
        source_path: Path to source audio/video file
        model_size: Whisper model size being used
        language: Language code or None for auto
        segments: List of segment dicts (start, end, text, confidence)
        last_segment_end: End timestamp of last completed segment
    """
    try:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        file_hash = _file_hash(source_path)
        checkpoint = TranscriptionCheckpoint(
            file_hash=file_hash,
            source_path=source_path,
            model_size=model_size,
            language=language,
            last_segment_end=last_segment_end,
            segments_json=json.dumps(segments),
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

        path = _checkpoint_path(file_hash)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(checkpoint), f, indent=2)

    except Exception:
        pass  # Best effort - don't fail transcription if checkpoint fails


def load_checkpoint(source_path: str) -> Optional[tuple[float, List[dict]]]:
    """
    Load a checkpoint for a file if it exists and is valid.

    Returns:
        Tuple of (last_segment_end, segments_list) or None if no valid checkpoint
    """
    try:
        file_hash = _file_hash(source_path)
        path = _checkpoint_path(file_hash)

        if not path.exists():
            return None

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Verify hash matches
        if data.get('file_hash') != file_hash:
            return None

        segments = json.loads(data.get('segments_json', '[]'))
        last_end = data.get('last_segment_end', 0)

        if not segments or last_end <= 0:
            return None

        return (last_end, segments)

    except Exception:
        return None


def clear_checkpoint(source_path: str):
    """Remove checkpoint for a file after successful completion."""
    try:
        file_hash = _file_hash(source_path)
        path = _checkpoint_path(file_hash)
        if path.exists():
            path.unlink()
    except Exception:
        pass


def get_checkpoint_info(source_path: str) -> Optional[dict]:
    """
    Get info about an existing checkpoint for display to user.

    Returns:
        Dict with checkpoint info or None
    """
    try:
        file_hash = _file_hash(source_path)
        path = _checkpoint_path(file_hash)

        if not path.exists():
            return None

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if data.get('file_hash') != file_hash:
            return None

        segments = json.loads(data.get('segments_json', '[]'))
        return {
            'last_segment_end': data.get('last_segment_end', 0),
            'segment_count': len(segments),
            'updated_at': data.get('updated_at', 'unknown'),
            'model_size': data.get('model_size', 'unknown'),
        }

    except Exception:
        return None
