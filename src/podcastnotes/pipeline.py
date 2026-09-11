"""
Sequencing the write-up, including the parts that stop and wait for a person.

The transcription side of this app is a background thread that runs to
completion. The write-up cannot be, because two of its stages are questions:
who is this voice, and what did they mean by that. So this is a state machine
the interface drives rather than a job it starts, and it is Qt-free so the
sequencing can be tested without a screen.

**Corrections are always re-applied from the original transcription, never on
top of the previous pass.** Answering a question adds corrections and the
transcript is then rebuilt, and rebuilding it from the already-corrected text
would substitute into text that has already been substituted: "Ridgeline"
becomes "Ridgeline" again where it can, and any span whose words have moved
quietly stops matching. The original stays untouched precisely so this can be
redone from it, and the confirmed speaker names are re-applied afterwards.

**The state is written to disk after every step.** These runs take minutes and
involve a person answering questions in between, so an app that lost the
context map because somebody closed a window would be worse than one that
never asked. It also means a trip can be reopened and finished later.

No Qt.
"""

import copy
import json
import os
from dataclasses import dataclass, field
from enum import Enum

from src.podcastnotes import attribution, context, correct, output, qa, speaker_library
from src.podcastnotes.context import ContextMap

STATE_FILENAME = "writeup.json"


class Step(Enum):
    """Where the write-up has got to. Ordered, and only ever moves forward."""

    CONTEXT = "context"
    CORRECT = "correct"
    ATTRIBUTE = "attribute"
    QUESTIONS = "questions"
    WRITE = "write"
    DONE = "done"


ORDER = [Step.CONTEXT, Step.CORRECT, Step.ATTRIBUTE, Step.QUESTIONS, Step.WRITE, Step.DONE]


class OutOfOrder(Exception):
    """A step was asked for before the one it depends on had finished.

    Raised rather than tolerated. Every one of these stages consumes the
    previous stage's output, so running one early does not produce a worse
    result, it produces a confident result computed from nothing.
    """


@dataclass
class WriteUp:
    """One trip's write-up, from raw transcript to two documents."""

    work_dir: str
    description: str = ""
    original: dict = field(default_factory=dict)
    boundaries: dict = field(default_factory=dict)

    step: Step = Step.CONTEXT
    context_map: ContextMap = field(default_factory=ContextMap)
    speaker_names: dict = field(default_factory=dict)
    asked: list = field(default_factory=list)
    round_number: int = 1
    transcript: str = ""
    summary: str = ""
    notes: list = field(default_factory=list)

    # --- the sequence ---------------------------------------------------------

    def _require(self, step: Step):
        if ORDER.index(self.step) < ORDER.index(step):
            raise OutOfOrder(
                f"{step.value} needs {self.step.value} to finish first."
            )

    def _advance(self, to: Step):
        """Only ever forwards, so a repeated step cannot rewind the run."""
        if ORDER.index(to) > ORDER.index(self.step):
            self.step = to

    def build_context(self, client=None, on_search=None) -> ContextMap:
        """Stage 2. The long one, and the only one that needs Glean."""
        self.context_map = context.build(
            self.description,
            _as_text(self.original),
            on_search=on_search,
            client=client,
        )
        self._advance(Step.CORRECT)
        self.save()
        return self.context_map

    def working(self) -> dict:
        """The transcript as it currently stands: corrected, and named if known.

        Rebuilt from the original every time rather than stored, because that
        is the only way answering a question can change an earlier decision.
        """
        self._require(Step.CORRECT)
        applied = correct.apply(self.original, self.context_map)
        self.notes = _describe(applied)
        return attribution.apply(applied.payload, self.speaker_names)

    def apply_corrections(self) -> dict:
        """Stage 3. Cheap and deterministic; no Claude and no network."""
        payload = self.working()
        self._advance(Step.ATTRIBUTE)
        self.save()
        return payload

    def voices(self) -> list:
        """The speaker labels as a screen needs them: clip offset, share, lines."""
        self._require(Step.CORRECT)
        return attribution.voices(self.working())

    def speaker_suggestions(self, client=None, library_path: str = "") -> list:
        """Stage 4, the proposing half. Claude first, then the voice library.

        The library's opinion is layered on top rather than replacing Claude's,
        because they know different things: Claude has read what this person
        said today, and the library has heard them before.
        """
        self._require(Step.ATTRIBUTE)
        payload = self.working()
        found = attribution.suggest(
            payload, self.context_map, description=self.description, client=client
        )
        remembered = speaker_library.suggest_names(payload, path=library_path)
        for suggestion in found:
            matches = remembered.get(suggestion.speaker) or []
            if matches and not suggestion.name:
                suggestion.name = matches[0].name if matches[0].confident else ""
                suggestion.evidence = (
                    f"Recognised from a previous trip at {matches[0].score:.2f}."
                )
        return found

    def confirm_speakers(self, names: dict, library_path: str = "") -> dict:
        """Stage 4, the deciding half. Only what a person confirmed."""
        self._require(Step.ATTRIBUTE)
        self.speaker_names = {
            label: name.strip()
            for label, name in (names or {}).items()
            if isinstance(name, str) and name.strip()
        }
        speaker_library.remember_trip(
            self.original, self.speaker_names,
            trip=self.description[:80], path=library_path,
        )
        self._advance(Step.QUESTIONS)
        self.save()
        return self.working()

    def next_questions(self, client=None) -> qa.Round:
        """Stage 6. An empty round means there is nothing left worth asking."""
        self._require(Step.QUESTIONS)
        found = qa.next_round(
            self.working(), self.context_map,
            already_asked=self.asked, number=self.round_number, client=client,
        )
        if not found.questions:
            self._advance(Step.WRITE)
            self.save()
        return found

    def answer(self, answers: dict, client=None):
        """Fold the answers in and move on to the next round, or to writing."""
        self._require(Step.QUESTIONS)
        self.asked.extend(m for m in (answers or {}) if m not in self.asked)
        if answers:
            self.context_map = qa.absorb(self.context_map, answers, client=client)
        self.round_number += 1
        if self.round_number > qa.MAX_ROUNDS:
            self._advance(Step.WRITE)
        self.save()

    def write_documents(self, client=None, style=None) -> output.Documents:
        """Stage 7. Both documents, from a record that is now settled."""
        self._require(Step.WRITE)
        documents = output.write(
            self.working(),
            self.context_map,
            description=self.description,
            boundaries=self.boundaries,
            style=style,
            client=client,
        )
        self.transcript = documents.transcript
        self.summary = documents.summary
        self._advance(Step.DONE)
        self.save()
        return documents

    # --- surviving a closed window --------------------------------------------

    @property
    def path(self) -> str:
        return os.path.join(self.work_dir, STATE_FILENAME)

    def save(self) -> str:
        os.makedirs(self.work_dir, exist_ok=True)
        with open(self.path, "w") as handle:
            json.dump(
                {
                    "version": 1,
                    "description": self.description,
                    "step": self.step.value,
                    "context_map": json.loads(self.context_map.to_json()),
                    "speaker_names": self.speaker_names,
                    "asked": self.asked,
                    "round_number": self.round_number,
                    "transcript": self.transcript,
                    "summary": self.summary,
                },
                handle,
                indent=1,
            )
        return self.path

    @classmethod
    def resume(cls, work_dir: str, original: dict, boundaries: dict = None) -> "WriteUp":
        """Reopen a write-up, or start one if there is nothing saved.

        The transcript is passed back in rather than stored here, because it
        already lives beside the audio as the segment export and keeping a
        second copy invites the two to disagree.
        """
        found = cls(work_dir=work_dir, original=original, boundaries=boundaries or {})
        try:
            with open(found.path) as handle:
                saved = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return found

        found.description = saved.get("description", "")
        found.context_map = ContextMap.from_dict(saved.get("context_map") or {})
        found.speaker_names = dict(saved.get("speaker_names") or {})
        found.asked = list(saved.get("asked") or [])
        found.round_number = int(saved.get("round_number") or 1)
        found.transcript = saved.get("transcript", "")
        found.summary = saved.get("summary", "")
        try:
            found.step = Step(saved.get("step"))
        except ValueError:
            # An unrecognised step is a file from a newer version. Starting
            # over is wrong and guessing is worse, so it resumes at the
            # earliest step, which recomputes rather than inventing.
            found.step = Step.CONTEXT
        return found


def _as_text(payload: dict) -> str:
    """The raw transcript as plain text, for the stages that read prose."""
    lines = []
    speaker = None
    for segment in payload.get("segments") or []:
        who = segment.get("speaker") or ""
        if who != speaker:
            lines.append(f"\n{who}:")
            speaker = who
        lines.append((segment.get("text") or "").strip())
    return "\n".join(lines).strip()


def _describe(applied) -> list:
    """The change log a person reads before publishing."""
    notes = []
    for change in applied.changes:
        notes.append(
            f"{'Flagged' if change.marked else 'Corrected'}: "
            f"{change.heard} -> {change.applied} ({change.occurrences}x)"
        )
    for change in applied.held_back:
        notes.append(f"Held back for review: {change.heard} -> {change.applied}")
    for change in applied.unmatched:
        notes.append(f"Proposed but not found in the transcript: {change.heard}")
    return notes
