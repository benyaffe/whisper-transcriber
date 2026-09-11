"""
Asking about the handful of things nothing could work out.

After the context map and the correction pass, a few things are still marked
`[?like this]`: a name nobody in the company's documents has written down, a
number that contradicts itself, a word the speaker could not parse on the day
either. Somebody who was on the trip can settle most of them in a minute.

**Four questions a round, three rounds at most.** Not a token budget; an
attention budget. The whole product exists to turn a two-hour chore into
twenty minutes, and a screen of nineteen questions is how that promise is
broken. Four is a screenful somebody answers; nineteen is a form they abandon,
and an abandoned form leaves every marker in place, which is worse than never
having asked.

Questions are ranked so the four that get asked are the four worth asking. A
marker appearing eleven times matters more than one appearing once, and a
person's name matters more than a room designation, because the name is
published against somebody's words.

Skipping is always allowed. An unanswered question leaves its marker exactly
as it was, which is an honest outcome and the reason the markers exist.

No Qt.
"""

import copy
import json
import re
from dataclasses import dataclass, field

from src.podcastnotes.llm import agent

MARKER = re.compile(r"\[\?([^\[\]]+)\]")

# An attention budget rather than a token budget. See the module docstring.
PER_ROUND = 4
MAX_ROUNDS = 3

SYSTEM = """You are writing a very short list of questions for the person assembling a write-up of
a recorded work trip, about the few things a machine transcript could not resolve.

**Write in the third person about everybody on the recording.** The person reading these
questions is often not on the tape: they are putting the document together from somebody else's
recording. "Anya says '75 beds, 60-hole' at the Westvale site, do you know what she meant?" is right.
"Anya, on the tape you say..." addresses the wrong person, and reads as though the tool has
confused who it is talking to.

They have limited patience and may have plenty of context. So:
  - Ask about things a transcript cannot settle but a person close to the trip can.
  - Quote the surrounding words, so the moment can be found again. "Anya says
    '75 beds, 60-hole' at the Westvale site" is answerable; "what is 60-hole" is not.
  - One question per thing. Do not bundle two uncertainties into one sentence.
  - Prefer the uncertainty that appears most often, and prefer people's names over room names
    and equipment, because a name gets published against somebody's words.
  - If a marker is obviously unimportant, leave it out. Asking fewer good questions is better
    than filling the quota.

Return ONLY a JSON object, no prose and no code fence:
  {"questions": [{"marker", "ask", "why_it_matters"}]}

"marker" is the exact marker text you are asking about, copied from the list you were given.
"ask" is the question itself, one or two sentences, written to be read by a colleague rather
than by a machine. "why_it_matters" is at most one short sentence, or "" when it is obvious."""

ABSORB_SYSTEM = """You are turning a colleague's answers to a few questions into corrections that
can be applied to a transcript by direct substitution.

For each answer, decide what it means for the transcript, and produce corrections in exactly the
form the correction stage consumes:
  - "heard" is the shortest run of words as it appears in the transcript, usually a single term
    or name, never a sentence, and never two errors combined.
  - "probably" is exactly what replaces it, so that swapping one for the other leaves a correct
    sentence and loses no words.
  - "confidence" is high when the colleague stated it plainly, medium when they hedged, and low
    when they guessed. They were there, so a plain statement is high confidence.

If an answer settles nothing, or says they do not know, produce no correction for it and record
it under "still_open" instead. Inventing a correction from "I'm not sure" is worse than leaving
the marker in place, because the marker is honest and the correction is not.

Return ONLY a JSON object, no prose and no code fence:
  {"settled": [marker], "corrections": [{"heard", "probably", "evidence", "confidence"}],
   "still_open": [string]}

"settled" lists the markers the colleague actually resolved, whether by correcting them or by
confirming what was already there. A marker they hedged on, guessed at, or could not recall does
NOT go in "settled", even though they replied: its uncertainty marker has to stay in the
transcript. Put those in "still_open" instead, saying what they offered.

"evidence" names the colleague's answer as the source, so the change log says where it came from.
"""


@dataclass
class Question:
    """One thing to ask, and what it is about."""

    marker: str
    ask: str
    why_it_matters: str = ""
    occurrences: int = 0


@dataclass
class Round:
    """What to ask this time, and what is left over."""

    questions: list = field(default_factory=list)
    remaining: int = 0
    number: int = 1

    @property
    def is_last(self) -> bool:
        return self.number >= MAX_ROUNDS


def outstanding(payload: dict) -> dict:
    """Every `[?marker]` still in the transcript, and how often each appears.

    Counted rather than listed, because frequency is most of what decides
    whether a question is worth one of the four slots.
    """
    found = {}
    for segment in payload.get("segments") or []:
        for marker in MARKER.findall(segment.get("text") or ""):
            marker = marker.strip()
            if marker:
                found[marker] = found.get(marker, 0) + 1
    return found


def excerpts(payload: dict, marker: str, limit: int = 2) -> list:
    """The lines a marker appears in, so a question can quote the moment."""
    needle = f"[?{marker}]"
    out = []
    for segment in payload.get("segments") or []:
        text = (segment.get("text") or "").strip()
        if needle in text:
            speaker = segment.get("speaker") or ""
            out.append(f"{speaker}: {text}" if speaker else text)
        if len(out) >= limit:
            break
    return out


def next_round(
    payload: dict,
    context_map,
    already_asked=(),
    number: int = 1,
    client=None,
) -> Round:
    """The next few questions, or an empty round when there is nothing to ask.

    `already_asked` is the markers put to the person in earlier rounds. They
    are excluded rather than re-ranked: asking the same thing twice because
    the first answer did not clear the marker is how a tool loses somebody's
    trust.
    """
    if number > MAX_ROUNDS:
        return Round(questions=[], remaining=0, number=number)

    counts = outstanding(payload)
    asked = {a.strip() for a in already_asked}
    open_markers = {m: n for m, n in counts.items() if m not in asked}
    if not open_markers:
        return Round(questions=[], remaining=0, number=number)

    answer = agent.ask(
        _prompt(payload, context_map, open_markers),
        system=SYSTEM,
        client=client,
        effort="max",
    )

    questions = []
    for row in _parse(answer.text).get("questions") or []:
        marker = _bare(row.get("marker"))
        ask = (row.get("ask") or "").strip()
        if marker not in open_markers or not ask:
            continue
        questions.append(
            Question(
                marker=marker,
                ask=ask,
                why_it_matters=(row.get("why_it_matters") or "").strip(),
                occurrences=open_markers[marker],
            )
        )
        if len(questions) >= PER_ROUND:
            break

    return Round(
        questions=questions,
        remaining=max(len(open_markers) - len(questions), 0),
        number=number,
    )


def _bare(marker) -> str:
    """A marker with its brackets off, however it came back.

    The markers are shown in the prompt the way they appear in the transcript,
    as `[?Simone]`, so that is the form Claude echoes. Matching that against
    the bare inner text silently discarded every question, which is a failure
    with no error: the round simply comes back empty and looks like Claude
    having nothing to ask.
    """
    text = (marker or "").strip()
    inner = MARKER.fullmatch(text)
    return inner.group(1).strip() if inner else text


def _prompt(payload: dict, context_map, open_markers: dict) -> str:
    parts = [
        "Background context already established, so do not ask about any of this:",
        json.dumps(
            {
                "people": getattr(context_map, "people", []),
                "terms": getattr(context_map, "terms", []),
                "organisations": getattr(context_map, "organisations", []),
            },
            ensure_ascii=False,
        ),
        f"\nUnresolved markers, with how often each appears and where. "
        f"Choose at most {PER_ROUND}:\n",
    ]
    for marker, count in sorted(open_markers.items(), key=lambda kv: -kv[1]):
        lines = "\n".join(f"    {line}" for line in excerpts(payload, marker))
        parts.append(f"--- [?{marker}]  appears {count}x\n{lines}")
    return "\n".join(parts)


def absorb(context_map, answers: dict, client=None):
    """Fold a person's answers back into the context map.

    Returns a new map with the corrections their answers imply added to
    `likely_errors`, so re-running the correction pass clears the markers.
    The original is not modified, because a caller showing what an answer
    changed needs both versions.

    An answer that settles nothing produces no correction. Manufacturing one
    from "I'm not sure" replaces an honest marker with a confident error.
    """
    given = {
        marker: text.strip()
        for marker, text in (answers or {}).items()
        if isinstance(text, str) and text.strip()
    }
    updated = copy.deepcopy(context_map)
    if not given:
        return updated

    answer = agent.ask(
        "A colleague who was on the trip answered these questions about the transcript.\n\n"
        + "\n\n".join(
            f"About the marker [?{marker}]:\n  they said: {text}"
            for marker, text in given.items()
        ),
        system=ABSORB_SYSTEM,
        client=client,
        effort="max",
    )

    parsed = _parse(answer.text)

    # Only a marker the colleague actually settled loses its original guess.
    # Replying "I'm not sure" is not settling it: dropping the guess there
    # would take the uncertainty marker out of the transcript and leave the
    # mis-heard words standing unflagged, which is the one outcome worse than
    # asking nothing.
    settled = {_bare(m) for m in (parsed.get("settled") or []) if str(m).strip()}
    settled &= set(given)
    updated.likely_errors = [
        row for row in updated.likely_errors if not _answers_it(row, settled)
    ]

    for row in parsed.get("corrections") or []:
        heard = (row.get("heard") or "").strip()
        probably = (row.get("probably") or "").strip()
        if not heard or not probably or heard == probably:
            # An answer that only confirms what was already there needs no
            # substitution. Dropping the uncertain entry above is what clears
            # the marker; an identity correction would just be noise in the
            # change log a person reads.
            continue
        updated.likely_errors.append(
            {
                "heard": heard,
                "probably": probably,
                "evidence": (row.get("evidence") or "answered by a colleague").strip(),
                "confidence": (row.get("confidence") or "high").strip().lower(),
            }
        )

    still_open = [q.strip() for q in (parsed.get("still_open") or []) if str(q).strip()]
    updated.open_questions = list(updated.open_questions) + still_open
    return updated


def _answers_it(row: dict, settled: set) -> bool:
    """Whether one of the settled markers is the guess this entry represents.

    The marker in the transcript is rendered from the entry's `probably`, so
    that is what a settled marker is matched against. Slash-separated
    alternatives are checked individually, because the correction stage pairs
    them off and renders whichever one applied.
    """
    candidate = (row.get("probably") or "").strip()
    # An empty candidate needs no guard of its own: it contributes only empty
    # parts, which the truthiness check below already rejects.
    parts = [candidate] + [p.strip() for p in candidate.split("/")]
    return any(part and part in settled for part in parts)


def _parse(text: str) -> dict:
    body = (text or "").strip()
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(body[start : end + 1])
    except json.JSONDecodeError:
        return {}
