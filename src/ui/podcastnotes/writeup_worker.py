"""
Running one write-up step off the GUI thread.

The write-up is a sequence with people in the middle of it, so unlike the
transcription worker this does not run the whole job. It runs exactly one step
and stops, and the window decides what happens next. That keeps the waiting
and the asking in the same place rather than spread between a thread and a
screen.

Steps that need no network are not here at all. Applying corrections is
deterministic string work over a few hundred segments and finishes faster than
starting a thread would, so the window calls it directly.

**Every failure arrives as a message somebody can act on.** A stack trace in a
status pane tells a non-technical colleague nothing, and the one failure most
likely to happen in normal use, an expired sign-in, is routine rather than
alarming: these credentials last a day or two because of the org's session
policy.
"""

from PyQt6.QtCore import QThread, pyqtSignal

# What a step is called on the screen while it runs. Present tense, because it
# is describing what is happening rather than what was asked for.
LABELS = {
    "context": "Reading the background for this trip",
    "speakers": "Working out who is speaking",
    "questions": "Looking for anything still unresolved",
    "answers": "Applying your answers",
    "write": "Writing the transcript and the summary",
}

# One or two words for the heading, so the screen says where in the job it is
# rather than standing on "Writing this up" for a quarter of an hour. "This" is
# nothing anybody can point at, and the same three words through five different
# stages reads as an app that has stopped.
PHASES = {
    "context": "Background",
    "speakers": "Speakers",
    "questions": "Questions",
    "answers": "Questions",
    "write": "Writing",
}


class WriteUpWorker(QThread):
    """One step of a write-up, on a background thread."""

    progress = pyqtSignal(str)      # a line for the status pane
    completed = pyqtSignal(object)  # whatever the step produced
    failed = pyqtSignal(str)        # already phrased for a person

    def __init__(self, writeup, step: str, answers: dict = None, library_path: str = "",
                 parent=None):
        super().__init__(parent)
        # Not `self.run`: QThread's own entry point is a method called run, and
        # holding the write-up under that name shadows it, so the thread starts
        # and immediately does nothing.
        self.writeup = writeup
        self.step = step
        self.answers = answers or {}
        self.library_path = library_path
        self._searches = 0

    @property
    def label(self) -> str:
        return LABELS.get(self.step, "Working")

    @property
    def phase(self) -> str:
        """The heading: where in the job this is, in a word or two."""
        return PHASES.get(self.step, "Working")

    def run(self):
        try:
            self.completed.emit(self._do())
        except Exception as e:
            self.failed.emit(explain(e))

    def _do(self):
        if self.step == "context":
            return self.writeup.build_context(on_search=self._searched)
        if self.step == "speakers":
            return self.writeup.speaker_suggestions(library_path=self.library_path)
        if self.step == "questions":
            return self.writeup.next_questions()
        if self.step == "answers":
            return self.writeup.answer(self.answers)
        if self.step == "write":
            return self.writeup.write_documents()
        raise ValueError(f"There is no write-up step called {self.step!r}.")

    def _searched(self, query: str, found=None):
        """Say what is happening now, not what happened last.

        Reporting only the start of each search leaves the line sitting on a
        finished lookup while Claude reads what came back, which takes up to a
        minute at this effort level and reads as a hang. Watched happening on
        a real run.
        """
        self._searches += 1
        if found is None:
            self.progress.emit(f"Looking up: {query}")
        else:
            self.progress.emit(
                f"Reading {found} result{'s' if found != 1 else ''} for “{query}”, "
                f"{self._searches // 2} searches so far"
            )


def explain(error: Exception) -> str:
    """Turn an exception into something a colleague can act on.

    Deliberately not a stack trace and deliberately not "an error occurred".
    Each of these is a real thing that happens in ordinary use, and each has a
    different next move.
    """
    from src.podcastnotes.context import GleanUnavailable
    from src.podcastnotes.llm import agent, client

    if isinstance(error, GleanUnavailable):
        return (
            "Could not reach your company's knowledge search, so the write-up has "
            "stopped rather than guessing at names.\n\n"
            "Check the connection on the setup screen and start the write-up again. "
            "The transcript is finished and saved either way."
        )
    if isinstance(error, client.NotSignedIn):
        return (
            "Your Google sign-in has expired, which happens every day or two.\n\n"
            "Sign in again on the setup screen and start the write-up again."
        )
    if isinstance(error, client.NoProjectChosen):
        return "No Google Cloud project is chosen yet. Pick one on the setup screen."
    if isinstance(error, client.ClaudeNotEnabled):
        return (
            f"Claude is not enabled in the project {error.project}.\n\n"
            f"Enable it once here: {error.url}"
        )
    if isinstance(error, client.UnsupportedRegion):
        return (
            f"The region {error.region} does not carry the model this app uses. "
            f"Change it to global on the setup screen."
        )
    if isinstance(error, agent.Refused):
        return (
            "Claude declined to work on this recording, so no document was written. "
            "Nothing has been changed or published."
        )
    return f"The write-up could not finish: {error}"
