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
# Roughly how long each step takes, in seconds, measured on the real Ashford
# trip: three recordings, forty-two minutes of audio, a map of eighteen people.
#
# An estimate, and labelled as one on screen. The context stage is an agentic
# loop that genuinely cannot predict its own length, so this is the shape of a
# typical run rather than a prediction about this one. It exists because a bar
# that never moves for six minutes reads as a hang, and because "about four
# minutes left" is a more useful thing to know than nothing.
TYPICAL_SECONDS = {
    "context": 360,
    "speakers": 40,
    "questions": 40,
    "answers": 60,
    "write": 420,
}

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
    found = pyqtSignal(str)         # something learned, for the running list
    fraction = pyqtSignal(float)    # 0..1, an estimate and shown as one
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
        self._started = 0.0
        self._ticker = None

    @property
    def label(self) -> str:
        return LABELS.get(self.step, "Working")

    @property
    def phase(self) -> str:
        """The heading: where in the job this is, in a word or two."""
        return PHASES.get(self.step, "Working")

    @property
    def typical_seconds(self) -> int:
        return TYPICAL_SECONDS.get(self.step, 120)

    def elapsed_fraction(self) -> float:
        """How far along a typical run of this step would be by now.

        Capped just short of full, because arriving at 100% and then continuing
        is worse than never claiming to know: it turns an estimate that was
        merely wrong into one that is visibly lying.
        """
        import time

        if not self._started:
            return 0.0
        return min((time.monotonic() - self._started) / self.typical_seconds, 0.97)

    def run(self):
        import time

        self._started = time.monotonic()
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
            self.found.emit(f"Searched for {query}")


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
