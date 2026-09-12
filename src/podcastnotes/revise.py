"""Taking somebody at their word after they have read the first draft.

The corrections used to be reviewed before anything was written, which asked
for a judgement before there was anything to judge against. This is the other
way round: the documents arrive, the person reads them, and what they say next
is worth more than a tick, because a name they supply can be searched.

**A supplied name is a key, not just a correction.** "The coordinator is
Simone Vasari" should send the research back out after Simone Vasari, then
after the studies she runs, then after the people around her. That is why this
is a fresh agentic pass over Glean rather than a substitution, and it is the
whole argument for doing it here rather than in the correction stage.

**They were there and this was not.** Anything stated plainly comes back at
high confidence, which is what makes the rewritten documents drop the `[?...]`
markers around it: `correct.apply` has one rule, high applies silently, and
promoting at the source keeps that rule with no exception bolted onto it.

Two things the pass has to be told, because both are invisible from the
finished documents it is reacting to. Corrections are applied to the *raw*
transcript, so `heard` has to be the words as the recogniser produced them and
not as they read after substitution. And a request the recording cannot support
comes back through `impossible` rather than as an invented correction, because
doing nothing is indistinguishable from being ignored: they retype it and pay
for another pass.

No Qt.
"""

import json
from dataclasses import dataclass, field

from src.podcastnotes import context
from src.podcastnotes.context import ContextMap

# Smaller than the sixty a cold pass gets. That one starts from a description
# and has to find everything; this one starts from a name somebody handed over
# and a map that already exists, so it branches far less.
REVISE_SEARCHES = 20

SYSTEM = """A colleague has read the first draft of a trip write-up and told you what is wrong
with it, or what you did not know. Your job is to look into what they said and return the
corrections and background it implies.

**Believe them.** They were physically present and you were not. Anything they state plainly is
true: set "confidence" to "high" and do not hedge it. Only hedge when they hedge.

**A name they give you is a key, not just a spelling.** Search for it, then for the project or
study it belongs to, then for the people around it. Go several levels deep, exactly as you would
from a cold start. This is the reason they are talking to you rather than editing the document
themselves.

You are given the raw transcript, the background already established, and the questions still
open. Use the search tool as much as you need.

{substitution_rules}

Return ONLY a JSON object, no prose and no code fence, with keys:
  "settled": [the exact open questions their note answers, copied from the list you were given]
  "rejected": [the exact "heard" values of corrections that should not happen after all]
  "impossible": [things they asked for that the recording cannot support, one sentence each]
  "speakers": {"Speaker 3": "Simone Vasari"}
  "people": [{"name", "role", "why_relevant", "confidence"}]
  "organisations": [{"name", "what_it_is"}]
  "terms": [{"term", "expansion", "note"}]
  "projects": [{"name", "what_it_is", "dates"}]
  "likely_errors": [{"heard", "probably", "evidence", "confidence"}]
  "hard_dates": [{"date", "what"}]
  "open_questions": [anything their note raised that you could not settle]

"speakers" is for when they identify a voice rather than fix a word. "The coordinator is Simone
Vasari" much more often means that a speaker label is her than that those syllables are misspelt,
and naming the label is the useful reading. Use the labels exactly as they appear in the
transcript you were given.

"rejected" is for when they say a word was right as it stood. Copy the "heard" value from the
background you were given, exactly. Use this rather than a correction from the word to itself.

"impossible" is not a failure. If they ask you to say something nobody said, or to name somebody
the recording never mentions, say so there and change nothing for it. Inventing a substitution to
look helpful puts a fabricated claim in a document somebody publishes."""


@dataclass
class Revision:
    """What a person's notes changed, and what could not be honoured."""

    context_map: ContextMap = field(default_factory=ContextMap)
    settled: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    impossible: list = field(default_factory=list)
    speakers: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (
            self.context_map.likely_errors
            or self.context_map.people
            or self.speakers
            or self.impossible
            or self.rejected
        )


def from_notes(
    context_map: ContextMap,
    notes: str,
    transcript: str,
    description: str = "",
    speakers=(),
    max_searches: int = REVISE_SEARCHES,
    on_search=None,
    client=None,
    search=None,
) -> Revision:
    """Research what somebody said about the draft, and return what it changes."""
    text, _ = context.research(
        _prompt(context_map, notes, transcript, description, speakers),
        SYSTEM.replace("{substitution_rules}", context.SUBSTITUTION_RULES),
        max_searches=max_searches,
        on_search=on_search,
        client=client,
        search=search,
    )
    parsed = context._parse(text)

    found = ContextMap.from_dict(parsed)
    known = {str(label) for label in speakers}
    return Revision(
        context_map=found,
        settled=[str(q).strip() for q in (parsed.get("settled") or []) if str(q).strip()],
        rejected=[
            str(h).strip() for h in (parsed.get("rejected") or []) if str(h).strip()
        ],
        impossible=[
            str(i).strip() for i in (parsed.get("impossible") or []) if str(i).strip()
        ],
        speakers={
            label: str(name).strip()
            for label, name in (parsed.get("speakers") or {}).items()
            # Only labels the recording actually has. A rename for a speaker
            # that does not exist cannot be applied and would be a silent no-op.
            if label in known and str(name).strip()
        },
    )


def _prompt(context_map, notes, transcript, description, speakers) -> str:
    return (
        f"Trip description:\n{description or '(none given)'}\n\n"
        f"What the colleague said after reading the draft:\n{notes}\n\n"
        f"Speaker labels in the transcript: {', '.join(str(s) for s in speakers) or '(none)'}\n\n"
        f"Background already established:\n"
        f"{json.dumps({'people': context_map.people, 'terms': context_map.terms, 'organisations': context_map.organisations, 'likely_errors': context_map.likely_errors}, ensure_ascii=False)}\n\n"
        f"Questions still open:\n"
        + "\n".join(f"  - {q}" for q in context_map.open_questions)
        + f"\n\nThe raw transcript, before any correction was applied. Write `heard` to match "
        f"this, not the finished document:\n\n{transcript}"
    )
