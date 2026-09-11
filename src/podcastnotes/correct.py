"""
Applying the context map to the transcript, without losing the clock.

The hard constraint here is word timings. Everything downstream that plays
audio against text depends on them: the six-second speaker clips, any future
click-a-line-to-hear-it, and the per-meeting split. So this stage does not ask
Claude to rewrite the transcript. Claude already did the difficult part when it
worked out that "Miles Nadoe" is Dr. Miles Nadeau; what is left is
substitution, and substitution can be done in place with the timings intact.

That is also why it is deterministic. A rewrite would be easier to write and
impossible to check: there would be no way to tell an improved sentence from a
quietly invented one. Here every change is a named substitution with a
recorded origin, and anything the map proposed that matched nothing is
reported rather than dropped.

**Confident renames apply. Unsure ones become `[?candidate]` inline.** An
uncertain correction that is applied silently is indistinguishable from a
certain one, and the reader has no way back to the original. The marker is the
whole value of the uncertain case.

When a substitution changes the number of words, the replacement words share
the original run's time span in proportion to their length. The span is what
matters, since it is what the audio actually occupies; the boundaries inside
it were always an estimate.

No Qt, no network. No Claude either: this stage spends nothing.
"""

import copy
import re
from dataclasses import dataclass, field

# Applied silently. Anything less certain is marked for a person to check.
CONFIDENT = "high"

# One and two character variants only. Three is the floor rather than the
# ceiling on purpose: the substitutions that matter most here are acronyms,
# OPS to OBS with CTA and EKG beside it, and excluding those to be careful
# would discard most of what this stage is for. Word boundaries already stop
# "OPS" firing inside "OPSEC", which is the risk that length was standing in
# for.
MIN_VARIANT = 3

# The context map writes alternatives as "trope / abnormal trope".
VARIANT_SPLIT = re.compile(r"\s*/\s*")


@dataclass
class Change:
    """One substitution, and everywhere it landed."""

    heard: str
    applied: str
    confidence: str
    marked: bool
    occurrences: int = 0
    segments: list = field(default_factory=list)


@dataclass
class Correction:
    """The corrected transcript, and an account of how it differs."""

    payload: dict
    changes: list = field(default_factory=list)
    unmatched: list = field(default_factory=list)

    @property
    def applied(self) -> int:
        return sum(c.occurrences for c in self.changes)


def apply(payload: dict, context_map) -> Correction:
    """Correct the segment JSON in place-preserving fashion.

    `payload` is the WT-3 export: segments with `text` and `words`. It is
    copied rather than mutated, because a caller that wants to show a
    before-and-after has no other way back to the original.
    """
    corrected = copy.deepcopy(payload)
    segments = corrected.get("segments") or []

    changes = []
    unmatched = []

    for proposal in _proposals(context_map):
        change = Change(
            heard=proposal["heard"],
            applied=proposal["replacement"],
            confidence=proposal["confidence"],
            marked=proposal["marked"],
        )
        for index, segment in enumerate(segments):
            hits = _apply_to_segment(segment, proposal)
            if hits:
                change.occurrences += hits
                change.segments.append(index)
        if change.occurrences:
            changes.append(change)
        else:
            unmatched.append(change)

    return Correction(corrected, changes, unmatched)


def _proposals(context_map) -> list:
    """One entry per usable variant, longest first.

    Longest first matters: "St. Bede's Lakesides" has to be tried before
    "St. Bede's", or the shorter match fires inside the longer phrase and the
    longer correction can then never apply.
    """
    out = []
    for row in getattr(context_map, "likely_errors", None) or []:
        heard = (row.get("heard") or "").strip()
        probably = (row.get("probably") or "").strip()
        if not heard or not probably:
            continue
        confidence = (row.get("confidence") or "").strip().lower()
        marked = confidence != CONFIDENT
        replacement = f"[?{probably}]" if marked else probably

        for variant in VARIANT_SPLIT.split(heard):
            variant = variant.strip()
            if len(variant) < MIN_VARIANT:
                continue
            out.append(
                {
                    "heard": variant,
                    "replacement": replacement,
                    "confidence": confidence,
                    "marked": marked,
                    "pattern": _pattern(variant),
                }
            )

    out.sort(key=lambda p: len(p["heard"]), reverse=True)
    return out


def _pattern(variant: str):
    """Case-insensitive, whole-word, tolerant of separators between tokens.

    The transcript writes "St. Bede's Fairmont" and the map may write
    "St Bede's Fairmont", so runs of non-alphanumerics match each other rather
    than having to be identical.

    Tolerance stops at the token. "Johns" is not made to match "John's",
    because the only way to do that is to allow punctuation between every pair
    of characters, and a pattern that loose starts substituting things nobody
    intended into a document that gets published. A correction that fails to
    match is reported as unmatched and someone sees it; a wrong one applied
    confidently is not noticed at all.
    """
    parts = [re.escape(p) for p in re.split(r"[^0-9A-Za-z]+", variant) if p]
    if not parts:
        return None
    return re.compile(
        r"(?<![0-9A-Za-z])" + r"[^0-9A-Za-z]+".join(parts) + r"(?![0-9A-Za-z])",
        re.IGNORECASE,
    )


def _apply_to_segment(segment: dict, proposal: dict) -> int:
    """Substitute in the segment text and keep its words in step. Returns hits."""
    pattern = proposal["pattern"]
    if pattern is None:
        return 0

    text = segment.get("text") or ""
    replaced, hits = pattern.subn(proposal["replacement"], text)
    if not hits:
        return 0

    segment["text"] = replaced
    words = segment.get("words")
    if words:
        segment["words"] = _retime(words, pattern, proposal["replacement"])
    return hits


def _retime(words: list, pattern, replacement: str) -> list:
    """Replace the matching run of words, sharing its span across the new ones.

    Word entries carry their own leading space from the recogniser, so the
    run is matched against the joined text rather than token by token. The
    span is preserved exactly; the boundaries within it are apportioned by
    character length, which is an estimate replacing an estimate.
    """
    joined = "".join(w.get("word", "") for w in words)
    match = pattern.search(joined)
    if match is None:
        return words

    # Which word entries the match covers.
    spans = []
    cursor = 0
    for word in words:
        length = len(word.get("word", ""))
        spans.append((cursor, cursor + length))
        cursor += length

    covered = [
        i for i, (begin, end) in enumerate(spans)
        if begin < match.end() and end > match.start()
    ]
    if not covered:
        return words

    first, last = covered[0], covered[-1]
    start = words[first].get("start")
    finish = words[last].get("end")

    # Keep whatever fell either side of the match inside the covered entries,
    # so "Ridgelane," keeps its comma when "Ridgelane" is replaced.
    prefix = joined[spans[first][0] : match.start()]
    suffix = joined[match.end() : spans[last][1]]
    rebuilt = prefix + replacement + suffix

    pieces = _split_keeping_spaces(rebuilt)
    if not pieces:
        return words[:first] + words[last + 1 :]

    probability = min(
        (w.get("probability", 1.0) or 1.0) for w in words[first : last + 1]
    )
    replacements = _apportion(pieces, start, finish, probability)
    return words[:first] + replacements + words[last + 1 :]


def _split_keeping_spaces(text: str) -> list:
    """Back into recogniser-shaped tokens, each carrying its leading space."""
    return [t for t in re.findall(r"\s*\S+", text) if t.strip()]


def _apportion(pieces: list, start, finish, probability) -> list:
    """Share one time span across several words, by length.

    If either end is missing the recogniser never gave a time for this run, so
    nothing is invented: the words go back without timings rather than with
    made-up ones.
    """
    if start is None or finish is None:
        return [{"word": p, "start": None, "end": None, "probability": probability}
                for p in pieces]

    total = sum(len(p.strip()) for p in pieces) or 1
    span = max(float(finish) - float(start), 0.0)

    out = []
    cursor = float(start)
    for index, piece in enumerate(pieces):
        share = span * (len(piece.strip()) / total)
        end = finish if index == len(pieces) - 1 else cursor + share
        out.append(
            {
                "word": piece,
                "start": round(cursor, 3),
                "end": round(float(end), 3),
                "probability": probability,
            }
        )
        cursor = float(end)
    return out
