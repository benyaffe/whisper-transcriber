"""
Putting names to voices, and never doing it on its own authority.

The diarizer gives back "Speaker 1" through "Speaker 4". Turning those into
people is the one stage where a wrong answer is both easy to produce and
impossible to spot afterwards: a transcript that attributes the GM's opinion
to a hardware engineer reads perfectly well and is completely wrong. So
nothing here is applied without being confirmed. `suggest` proposes and
`apply` takes only the decisions it is handed.

The diarizer also invents people. On the real Ashford recording it produced a
fourth speaker with 22 seconds of speech who is not a fourth participant, and
every one of those lines belongs to somebody already in the room. Suggestions
can therefore say "this label is the same person as that one", which the
merging in `apply` then honours by giving both the same name.

**Why this is its own stage rather than something each document works out.**
The quality spike generated a corrected transcript and a summary
independently, and they disagreed about who the fourth speaker was, because
each re-derived it. One stage decides, once, and both documents read the
answer.

Two things are deliberately left alone. `pyannote_label` is the link back to
the raw diarizer output and the row order of the embeddings, so renaming must
never touch it. And `diarization.turns` is keyed by that label rather than by
the display name, so it needs no rewriting either.

No Qt.
"""

import copy
import json
from dataclasses import dataclass, field

from src.podcastnotes.llm import agent

# Enough of each voice for Claude to recognise a person from how they talk and
# what they talk about, without handing over the whole transcript per speaker.
SAMPLE_LINES = 12

# Below this, a label is likely to be the diarizer splitting somebody rather
# than a real additional participant. Not a rule, just something the prompt is
# told so it treats the label with suspicion.
PHANTOM_SECONDS = 60.0

SYSTEM = """You are identifying who each voice on a recorded work-trip debrief belongs to.

You are given the trip description, background context about the people involved, and for each
speaker label the diarizer produced: how long they speak, and a sample of their lines.

Work out which real person each label is. Use what people say about each other, who addresses
whom, what each person would plausibly know about, and the roles in the background context.
Somebody who talks about hardware and device siting is not the same person who talks about
commercial strategy.

Speech-to-text diarizers make two specific mistakes and you should expect both:
  - They invent a speaker. A label with very little total speech, whose lines all read as
    continuations of somebody else's turn, is usually not a real additional participant.
  - They split one person across two labels.
When you believe two labels are the same person, say so with "same_as".

Return ONLY a JSON object, no prose and no code fence:
  {"speakers": [{"speaker", "name", "confidence", "evidence", "same_as"}]}

One entry per label you were given, using the label exactly as it was given to you.
  - "name" is the person's real name, or "" if you genuinely cannot tell. An empty name is a
    good answer when the evidence is not there; a plausible guess is not, because a name here
    is published against somebody's words.
  - **If you are choosing between two people, you do not know.** Return "" for the name and
    say in the evidence who the candidates are and what would tell them apart. Naming your
    marginal preference is worse than saying nothing, because the person reviewing cannot see
    that it was marginal and will simply accept it.
  - "confidence" is high, medium or low.
  - "evidence" is one or two sentences saying what convinced you, quoting the recording where
    you can.
  - "same_as" is another speaker label when this label is the same person, otherwise "".
"""


@dataclass
class Voice:
    """One label the diarizer produced, as a person would need to judge it."""

    speaker: str
    pyannote_label: str = ""
    total_speech_s: float = 0.0
    share: float = 0.0
    lines: list = field(default_factory=list)
    # Where to start playing to hear this voice. The longest turn rather than
    # the first, for the same reason the sample lines are: an opening "yeah,
    # exactly" is not enough of anybody to recognise.
    first_heard: float = 0.0

    @property
    def is_slight(self) -> bool:
        """Little enough speech to be worth doubting as a real participant."""
        return self.total_speech_s < PHANTOM_SECONDS


@dataclass
class Suggestion:
    """What Claude thinks, which is not the same as what will be applied."""

    speaker: str
    name: str = ""
    confidence: str = ""
    evidence: str = ""
    same_as: str = ""

    @property
    def worth_filling_in(self) -> bool:
        """Whether this is good enough to put in the box for somebody.

        A low-confidence guess is not. Measured across three runs of the same
        recording, the 22-second phantom speaker came back as Anya, Anya and then
        Devan, from identical diarization: the answer flips between runs. Put
        in the box it looks exactly like the three confident ones above it and
        gets accepted with them, so the box stays empty and the row asks.
        """
        return bool(self.name) and self.confidence in ("high", "medium")


def voices(payload: dict, sample_lines: int = SAMPLE_LINES) -> list:
    """Every speaker label, with the material needed to identify it.

    Ordered by how much each speaks, so the person reviewing meets the main
    voices first and the doubtful fragments last. Sample lines are the longest
    ones rather than the first: an opening "yeah, exactly" identifies nobody,
    while a long turn carries both subject matter and cadence.
    """
    segments = payload.get("segments") or []
    profiles = {s.get("id"): s for s in (payload.get("speakers") or [])}

    by_speaker = {}
    for segment in segments:
        label = segment.get("speaker")
        if not label:
            continue
        by_speaker.setdefault(label, []).append(segment)

    spoken = {
        label: (profiles.get(label, {}).get("total_speech_s") or _spoken(rows))
        for label, rows in by_speaker.items()
    }
    total = sum(spoken.values()) or 1.0

    out = []
    for label, rows in by_speaker.items():
        longest = sorted(rows, key=lambda r: len(r.get("text") or ""), reverse=True)
        out.append(
            Voice(
                speaker=label,
                pyannote_label=profiles.get(label, {}).get("pyannote_label", ""),
                total_speech_s=round(spoken[label], 1),
                share=round(spoken[label] / total, 3),
                lines=[(r.get("text") or "").strip() for r in longest[:sample_lines]],
                first_heard=float(longest[0].get("start") or 0.0) if longest else 0.0,
            )
        )
    out.sort(key=lambda v: v.total_speech_s, reverse=True)
    return out


def _spoken(rows: list) -> float:
    """Fallback when the export carries no speaker profiles."""
    return sum(max((r.get("end") or 0) - (r.get("start") or 0), 0) for r in rows)


def suggest(payload: dict, context_map, description: str = "", client=None) -> list:
    """Ask Claude who each voice is. Proposes only; applies nothing."""
    found = voices(payload)
    if not found:
        return []

    answer = agent.ask(
        _prompt(found, context_map, description),
        system=SYSTEM,
        client=client,
        # One hard question rather than a research loop, which is the case
        # where the top of the effort range is worth paying for.
        effort="max",
    )
    return _read(answer.text, found)


def _prompt(found: list, context_map, description: str) -> str:
    people = getattr(context_map, "people", None) or []
    parts = [
        f"Trip description:\n{description or '(none given)'}\n",
        "People who may be involved, from the company's internal knowledge:",
        json.dumps(people, ensure_ascii=False, indent=1),
        "\nThe voices the diarizer produced:\n",
    ]
    for voice in found:
        note = "  (very little speech, treat with suspicion)" if voice.is_slight else ""
        parts.append(
            f"--- {voice.speaker}{note}\n"
            f"speaks for {voice.total_speech_s}s, {voice.share:.0%} of the recording\n"
            + "\n".join(f"  {line}" for line in voice.lines)
        )
    return "\n".join(parts)


def _read(text: str, found: list) -> list:
    """Parse the answer, keeping only labels that actually exist.

    A suggestion naming a speaker the recording does not contain cannot be
    confirmed or applied, and carrying it forward would put a row in the review
    screen that does nothing.
    """
    known = {v.speaker for v in found}
    body = text.strip()
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end <= start:
        return []
    try:
        parsed = json.loads(body[start : end + 1])
    except json.JSONDecodeError:
        return []

    out = []
    for row in parsed.get("speakers") or []:
        label = (row.get("speaker") or "").strip()
        if label not in known:
            continue
        same_as = (row.get("same_as") or "").strip()
        out.append(
            Suggestion(
                speaker=label,
                name=(row.get("name") or "").strip(),
                confidence=(row.get("confidence") or "").strip().lower(),
                evidence=(row.get("evidence") or "").strip(),
                same_as=same_as if same_as in known and same_as != label else "",
            )
        )
    return out


def apply(payload: dict, names: dict) -> dict:
    """Rename the confirmed speakers and nobody else.

    `names` maps a speaker label to the name a person confirmed. A label that
    is absent, or mapped to an empty name, keeps the label it had: an
    unreviewed voice stays visibly unreviewed rather than silently becoming
    somebody. Two labels may map to the same name, which is how a diarizer's
    invented speaker is merged back into the person it was split from.

    `pyannote_label` is left exactly as it was. It is the link to the raw
    diarizer output and to the row order of the speaker embeddings, so the
    display name is the only thing that may move.
    """
    renamed = copy.deepcopy(payload)
    confirmed = {
        label: name.strip()
        for label, name in (names or {}).items()
        if isinstance(name, str) and name.strip()
    }
    if not confirmed:
        return renamed

    for segment in renamed.get("segments") or []:
        label = segment.get("speaker")
        if label in confirmed:
            segment["speaker"] = confirmed[label]

    for profile in renamed.get("speakers") or []:
        label = profile.get("id")
        if label in confirmed:
            profile["id"] = confirmed[label]

    return renamed


def merges(suggestions: list) -> dict:
    """The label-to-label merges Claude proposed, resolved to one target each.

    Follows a chain so that three labels claiming to be each other settle on
    one, and refuses to follow a cycle, which would otherwise not terminate.
    """
    claim = {s.speaker: s.same_as for s in suggestions if s.same_as}
    resolved = {}
    for label in claim:
        seen = {label}
        target = claim[label]
        while target in claim and target not in seen:
            seen.add(target)
            target = claim[target]
        resolved[label] = target
    return resolved
