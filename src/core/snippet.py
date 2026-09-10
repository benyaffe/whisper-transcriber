"""
Short audio clips, one per speaker, for putting a name to a voice.

The speaker-attribution step shows one row per `Speaker N` and asks who it is.
Answering needs a few seconds of that person talking, so this finds the best few
seconds in the recording and cuts them out.

Reads the segment JSON that a transcription writes, rather than an in-memory
result, so it works equally on a run from ten minutes ago or last month.

No Qt, same as runner.py.
"""

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

from src.utils.file_utils import get_bundled_binary

# Long enough to recognise a voice, short enough to click through a list of
# them quickly.
DEFAULT_CLIP_SECONDS = 6.0

# Clips are re-encoded rather than stream-copied. A six-second cut from the
# middle of an m4a would otherwise start at the nearest keyframe rather than
# where asked, and this rate and format is what the audio player wants anyway.
CLIP_SAMPLE_RATE = 16000


@dataclass
class SpeakerClip:
    """Where one speaker can best be heard, and the clip once it is cut."""

    speaker: str            # display label, "Speaker 2"
    pyannote_label: str     # raw label. Not derivable from the display digit;
                            # see the JSON export for why that matters.
    start: float
    duration: float
    text: str = ""          # what is said in the window, to show beside the player
    path: Optional[str] = None

    @property
    def end(self) -> float:
        return self.start + self.duration


class ClipExtractionError(Exception):
    """ffmpeg could not produce the clip."""


def _turns_for_selection(payload: dict) -> list[dict]:
    """The timeline to pick windows from.

    exclusive_turns keeps only the dominant speaker in each frame. That is the
    whole point here: a window where two people talk at once is the worst
    possible thing to hand somebody trying to identify a voice. Falls back to
    the overlapping turns, because a pipeline that produced no exclusive view
    should still yield clips.
    """
    diarization = payload.get("diarization") or {}
    return diarization.get("exclusive_turns") or diarization.get("turns") or []


def _text_between(payload: dict, start: float, end: float) -> str:
    """Transcript text overlapping a time window."""
    spoken = [
        s["text"]
        for s in payload.get("segments", [])
        if s.get("start", 0.0) < end and s.get("end", 0.0) > start
    ]
    return " ".join(t.strip() for t in spoken if t).strip()


def pick_speaker_clips(
    payload: dict, max_seconds: float = DEFAULT_CLIP_SECONDS
) -> list[SpeakerClip]:
    """Choose the best window for each speaker. Pure: touches no audio.

    Each speaker gets their single longest uninterrupted turn, truncated to
    max_seconds. A speaker whose longest turn is shorter than that gets a
    shorter clip rather than nothing; someone who only ever says "mm-hmm" is
    exactly who is hardest to identify, so a brief clip still beats silence.
    """
    turns = _turns_for_selection(payload)
    speakers = payload.get("speakers") or []

    longest: dict[str, dict] = {}
    for turn in turns:
        label = turn.get("speaker")
        if label is None:
            continue
        duration = turn.get("end", 0.0) - turn.get("start", 0.0)
        if duration <= 0:
            continue
        best = longest.get(label)
        if best is None or duration > (best["end"] - best["start"]):
            longest[label] = turn

    clips = []
    for entry in speakers:
        raw = entry.get("pyannote_label")
        turn = longest.get(raw)
        if turn is None:
            # Present in the speaker list but never got a clean turn of their
            # own. No clip is the honest answer; do not invent one.
            continue
        start = turn["start"]
        duration = min(turn["end"] - start, max_seconds)
        clips.append(
            SpeakerClip(
                speaker=entry.get("id", raw),
                pyannote_label=raw,
                start=start,
                duration=duration,
                text=_text_between(payload, start, start + duration),
            )
        )
    return clips


def _clip_filename(clip: SpeakerClip) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in clip.speaker).strip("_")
    return f"{safe or clip.pyannote_label}.wav"


def extract_clip(audio_path: str, clip: SpeakerClip, out_dir: str) -> SpeakerClip:
    """Cut one clip out of the audio. Returns the clip with .path filled in."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, _clip_filename(clip))

    cmd = [
        get_bundled_binary('ffmpeg'),
        '-nostdin', '-loglevel', 'error', '-y',
        # -ss BEFORE -i is the fast seek: ffmpeg jumps to the offset instead of
        # decoding up to it. Measured at 0.29s versus 0.63s for a 588s offset.
        '-ss', f"{clip.start:.3f}",
        '-t', f"{clip.duration:.3f}",
        '-i', audio_path,
        '-vn', '-ac', '1', '-ar', str(CLIP_SAMPLE_RATE),
        '-c:a', 'pcm_s16le',
        out_path,
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=120, check=False)
    if result.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        # Never leave a truncated or empty file where a caller might play it.
        if os.path.exists(out_path):
            os.remove(out_path)
        stderr = result.stderr.decode('utf-8', errors='replace').strip()
        raise ClipExtractionError(
            f"Could not extract {clip.speaker} at {clip.start:.1f}s: {stderr or 'no output'}"
        )

    clip.path = out_path
    return clip


def extract_speaker_clips(
    json_path: str,
    audio_path: str,
    out_dir: str,
    max_seconds: float = DEFAULT_CLIP_SECONDS,
) -> list[SpeakerClip]:
    """Pick and cut a clip for every speaker in a transcription's JSON."""
    with open(json_path, encoding='utf-8') as f:
        payload = json.load(f)

    return [
        extract_clip(audio_path, clip, out_dir)
        for clip in pick_speaker_clips(payload, max_seconds=max_seconds)
    ]
