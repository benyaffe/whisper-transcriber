"""
Is the app actually able to do its job right now?

Not a first-run wizard. A standing list, consulted whenever something is not
working. A token expires, a laptop is replaced, an admin revokes access: those
all look identical to the person using the app, and they all get the same list.

**Every check does real work.** A saved credential that has been revoked looks
exactly like a working one until something calls with it, and discovering that
half way through a forty-minute trip is the failure this module exists to
prevent. So the Glean check runs a search, the Claude check asks for a
completion, and the Drive check creates a document and deletes it again.

Checks are ordinary objects with a name, a purpose, a `check()` and a `fix()`,
so adding a sixth is one class rather than an edit in five places.

No Qt. The screen that renders this lives in src/ui/podcastnotes/.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


SETTINGS_LAST_ALL_OK = "setup_last_all_ok"


def was_working_last_time() -> bool:
    """Whether the checks were all green when they last ran.

    Used only to decide which screen to open on. Never used to skip a check:
    the whole premise here is that a credential which worked yesterday tells
    you nothing about today.
    """
    from src.core.config import _settings

    return _settings().value(SETTINGS_LAST_ALL_OK, False, type=bool)


def remember_working(value: bool):
    from src.core.config import _settings

    _settings().setValue(SETTINGS_LAST_ALL_OK, bool(value))


class State(Enum):
    OK = "ok"
    FAILED = "failed"
    # Something upstream is missing, so this one cannot be judged yet. Shown
    # differently from a failure: telling somebody Drive is broken when they
    # have simply not signed in yet sends them to fix the wrong thing.
    BLOCKED = "blocked"
    CHECKING = "checking"
    UNKNOWN = "unknown"


@dataclass
class Result:
    state: State
    # One line, in plain language, for the person looking at the list.
    detail: str = ""
    # What to do about it. Empty when there is nothing to do.
    remedy: str = ""
    # Somewhere to send them, when a page can fix it.
    url: str = ""

    @property
    def ok(self) -> bool:
        return self.state is State.OK


def ok(detail: str = "") -> Result:
    return Result(State.OK, detail=detail)


def failed(detail: str, remedy: str = "", url: str = "") -> Result:
    return Result(State.FAILED, detail=detail, remedy=remedy, url=url)


def blocked(detail: str) -> Result:
    return Result(State.BLOCKED, detail=detail)


@dataclass
class Check:
    """One thing that has to work.

    `key` is stable and used in code; `title` and `purpose` are what a person
    reads. `purpose` says what breaks without it, in their terms, because
    "Vertex AI" means nothing to most people and "writing the summary" does.
    """

    key: str
    title: str
    purpose: str
    run: Callable[[], Result]
    # Keys that must pass first. A check whose requirement failed reports
    # BLOCKED rather than running and producing a second, misleading error.
    requires: list[str] = field(default_factory=list)
    # Whether a person can act on a failure, or it is simply a wait.
    fixable: bool = True

    def __call__(self) -> Result:
        try:
            return self.run()
        except Exception as e:
            # A check that raises is a failed check, never a crashed app. This
            # list is the thing people look at when something is already wrong.
            return failed(
                f"{type(e).__name__}: {e}",
                remedy="This is unexpected. Copy the diagnostics and send them on.",
            )


class Readiness:
    """The whole list, and whether the app can currently work."""

    def __init__(self, checks: list[Check]):
        self.checks = checks
        self._by_key = {c.key: c for c in checks}

    def run(
        self,
        only: Optional[list[str]] = None,
        known: Optional[dict[str, Result]] = None,
        on_start: Optional[Callable[[str], None]] = None,
        on_result: Optional[Callable[[str, Result], None]] = None,
    ) -> dict[str, Result]:
        """Run every check, in order, skipping those whose requirements failed.

        `only` re-checks a subset, which is what the screen does after somebody
        fixes one row. `known` carries the results of the previous run, so a
        subset re-check can still see that a requirement is failing. Without it,
        re-checking Drive on its own while Google is signed out would make a
        doomed call and report a second, more confusing error.

        `on_start` and `on_result` fire per check so a caller can show progress.
        Several of these make network calls, and a list that sits blank for ten
        seconds and then fills in at once looks broken while it is working.

        Returns the merged dict, so the caller can hold one set of results.
        """
        results: dict[str, Result] = dict(known or {})
        for check in self.checks:
            if only is not None and check.key not in only:
                continue
            unmet = [
                self._by_key[r].title
                for r in check.requires
                if r in results and not results[r].ok
            ]
            if unmet:
                results[check.key] = blocked(f"Waiting on {unmet[0]}")
                if on_result:
                    on_result(check.key, results[check.key])
                continue
            if on_start:
                on_start(check.key)
            results[check.key] = check()
            if on_result:
                on_result(check.key, results[check.key])
        return results

    @staticmethod
    def all_ok(results: dict[str, Result]) -> bool:
        return bool(results) and all(r.ok for r in results.values())

    @staticmethod
    def first_problem(results: dict[str, Result]) -> Optional[str]:
        """The key of the thing to fix first: earliest failure, ignoring blocked.

        Blocked checks are downstream of a failure, so sending somebody to one
        of those sends them to a symptom.
        """
        for key, result in results.items():
            if result.state is State.FAILED:
                return key
        return None
