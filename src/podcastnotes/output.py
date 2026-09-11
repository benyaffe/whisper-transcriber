"""
Writing the two documents somebody actually reads.

The polished transcript and the summary, produced from the same inputs and
published as one Google Doc. Everything before this exists to make these two
worth reading.

**They are written from a settled record, not each from scratch.** The quality
spike generated both independently and they contradicted each other about who
the fourth speaker was, because each worked it out again. By this point the
speakers are named, the terms are corrected and the open questions are known,
so both documents are describing a decided thing rather than deciding it.

**House style is enforced, not requested.** A prompt saying "never use an
em-dash" works nearly always, and nearly is not good enough for a rule
somebody cares about: the one that slips through lands in a published document
with their name on it. So the prompt asks and the code then checks, which
costs nothing and makes the rule true rather than likely.

**Uncertainty markers must survive.** `[?Errol Marchetti]` exists because nobody
could confirm it, and a writing pass that tidies it into "Errol Marchetti" turns
an honest flag into a fabricated fact. They are counted on the way in and on
the way out, and any that went missing are reported rather than assumed fine.

Per-meeting sections come from `boundaries.json`, which is why that was built
long before anything could use it. A trip of one recording is a trip with one
meeting and gets no artificial split.

No Qt.
"""

import re
from dataclasses import dataclass, field

from src.podcastnotes.llm import agent
from src.podcastnotes.qa import MARKER

# Em-dash, en-dash used as a dash, and the double hyphen people type instead.
# Replaced with a comma and a space, which is the substitution Ben makes by
# hand and the reason this rule exists at all.
DASHES = re.compile(r"\s*(?:—|-{2,}|(?<=\s)–(?=\s))\s*")

# A line of three or more hyphens is a Markdown horizontal rule, not somebody
# reaching for an em-dash.
#
# This is the fix for a real defect and not a hypothetical. `--` matched inside
# `---`, so every section break became the literal text ", -" and took the blank
# line after it along too. It happened four times in the last real transcript,
# and the symptom people reported was that the document had no line breaks and
# was hard to read, which sounds like a rendering problem and is not.
RULE_LINE = re.compile(r"^[ \t]*-{3,}[ \t]*$")


@dataclass
class HouseStyle:
    """The conventions these documents follow, as data rather than prose.

    Written down here because they are decisions somebody made once and should
    not have to make again, and because a rule in a prompt is a request while a
    rule here can be checked.
    """

    em_dashes: bool = False
    action_items_as_bullets: bool = True
    owner_column: bool = False

    def rules(self) -> str:
        """The style section of the prompt, generated from the settings."""
        lines = []
        if not self.em_dashes:
            lines.append(
                "Never use an em-dash or an en-dash. Where you would write one, write a "
                "comma followed by a space instead."
            )
        if self.action_items_as_bullets:
            lines.append("Write action items as bullets, never as a table.")
        if not self.owner_column:
            lines.append(
                "Do not attach an owner to each action item. Where the recording clearly "
                "assigns something to somebody, say so in the sentence instead."
            )
        lines.append(
            "Write complete sentences throughout. No sentence fragments and no clipped "
            "note-taking style, in headings as well as body text."
        )
        return "\n".join(f"  - {line}" for line in lines)


@dataclass
class Documents:
    """What came out, and what to check before publishing it."""

    transcript: str = ""
    summary: str = ""
    dropped_markers: list = field(default_factory=list)
    style_fixes: int = 0

    @property
    def combined(self) -> str:
        """Both documents, summary first, as one Markdown file to publish."""
        return f"{self.summary}\n\n---\n\n{self.transcript}".strip()


TRANSCRIPT_SYSTEM = """You are producing the readable transcript of a recorded work-trip debrief,
for a colleague who was there and wants to find what was said.

The speakers are already named and the terminology is already corrected. Do not re-decide either.
Your job is presentation:
  - Group the turns under the speaker names exactly as they are given to you.
  - Fix punctuation and paragraphing. Drop pure filler that carries no meaning.
  - Merge consecutive turns from the same speaker into continuous paragraphs.
  - Add section headings where the conversation moves to a new site or topic.
  - Open with a short header giving the trip, the sites in the order discussed, and who was on
    the recording.

Two things are absolute. Do not remove content: this is the whole transcript, reflowed, not a
summary. And copy every [?marker] through exactly as it appears, brackets and all. A marker means
nobody could confirm that word, so tidying it away turns an honest flag into an invented fact.

The header block describes the trip and never how the transcript was made. The background
context can contain notes from an earlier write-up of this same trip, because those get published
and then indexed alongside everything else, and it is easy to repeat one as though it belonged
here. Nobody wants a paragraph about transcription passes or speaker-label policy.

Output Markdown and nothing else."""

SUMMARY_SYSTEM = """You are writing the summary and action items that follow a work trip, for a
senior colleague who was on the trip and wants the decisions rather than a retelling.

Produce, in Markdown:
  - A short header: the trip, the sites in order, who was there, and the purpose.
  - "Top-line takeaways": the handful of things that change what the team does next. Lead each
    with the claim in bold, then the evidence for it.
  - A site-by-site or topic-by-topic summary, whichever the recording is actually organised by.
  - "Action items".
  - "Open questions" for whatever the recording genuinely left unresolved.

Be specific with numbers and names, and prefer a direct quotation to a paraphrase where the
wording matters. Do not invent an owner, a date or a figure that was not said.

Copy any [?marker] through exactly as it appears. It means nobody could confirm that word, and
stating it cleanly in a summary is how an uncertain name becomes a fact somebody acts on.

Write about the trip, never about how the transcript was made. The background context can
contain notes from an earlier write-up of this same trip, because those get published and then
indexed alongside everything else, and it is easy to repeat one as though it were a finding.
Nobody reading this wants a paragraph about transcription passes or speaker-label policy.

Output Markdown and nothing else."""


def meetings(payload: dict, boundaries: dict = None) -> list:
    """The recording split into meetings, as (name, segments) pairs.

    A trip of one recording is one meeting and gets no artificial split. The
    spans come from `boundaries.json` because file timestamps cannot supply
    them: on the real Ashford trip all three recordings report creation within
    four seconds of each other, since that is when they were copied off the
    device rather than when they were made.
    """
    segments = payload.get("segments") or []
    sources = (boundaries or {}).get("sources") or []
    if len(sources) < 2:
        return [("", segments)]

    out = []
    for source in sources:
        start = source.get("start") or 0.0
        end = source.get("end")
        inside = [
            s for s in segments
            if (s.get("start") or 0.0) >= start
            and (end is None or (s.get("start") or 0.0) < end)
        ]
        out.append((source.get("source_name") or "", inside))
    return out


def as_script(payload: dict, boundaries: dict = None) -> str:
    """The attributed transcript as plain text for Claude to work from."""
    parts = []
    for name, segments in meetings(payload, boundaries):
        if name:
            parts.append(f"\n===== recording: {name} =====")
        speaker = None
        for segment in segments:
            who = segment.get("speaker") or ""
            if who != speaker:
                parts.append(f"\n{who}:")
                speaker = who
            parts.append((segment.get("text") or "").strip())
    return "\n".join(parts).strip()


def write(
    payload: dict,
    context_map=None,
    description: str = "",
    boundaries: dict = None,
    style: HouseStyle = None,
    client=None,
) -> Documents:
    """Both documents, in house style, with the markers intact."""
    style = style or HouseStyle()
    script = as_script(payload, boundaries)
    expected = _markers(script)

    shared = _shared(description, context_map, script, style)

    transcript = agent.ask(
        shared + "\n\nProduce the reflowed transcript now.",
        system=TRANSCRIPT_SYSTEM,
        client=client,
        effort="max",
    ).text
    summary = agent.ask(
        shared + "\n\nProduce the summary and action items now.",
        system=SUMMARY_SYSTEM,
        client=client,
        effort="max",
    ).text

    transcript, fixed_a = enforce(transcript, style)
    summary, fixed_b = enforce(summary, style)

    return Documents(
        transcript=transcript,
        summary=summary,
        dropped_markers=sorted(expected - _markers(transcript)),
        style_fixes=fixed_a + fixed_b,
    )


def _shared(description: str, context_map, script: str, style: HouseStyle) -> str:
    open_questions = getattr(context_map, "open_questions", None) or []
    return (
        f"Trip description:\n{description or '(none given)'}\n\n"
        f"House style, which is not negotiable:\n{style.rules()}\n\n"
        f"Questions nobody could resolve, so do not present them as settled:\n"
        + "\n".join(f"  - {q}" for q in open_questions)
        + f"\n\nThe transcript. Speakers are already named and terms already corrected:\n\n{script}"
    )


def enforce(text: str, style: HouseStyle = None) -> tuple:
    """Apply the house rules the code can apply. Returns the text and a count.

    Only the rules that are mechanically checkable are enforced here. Asking
    for them in the prompt works nearly always, and the one that slips through
    lands in a published document with somebody's name on it.
    """
    style = style or HouseStyle()
    if style.em_dashes:
        return text, 0

    # Line by line, so a horizontal rule can be left alone. Substituting over
    # the whole document is what turned every section break into ", -".
    fixes = 0
    lines = []
    for line in text.split("\n"):
        if RULE_LINE.match(line):
            lines.append(line)
            continue
        line, count = DASHES.subn(", ", line)
        fixes += count
        lines.append(line)
    return "\n".join(lines), fixes


def _markers(text: str) -> set:
    return {m.strip() for m in MARKER.findall(text or "")}
