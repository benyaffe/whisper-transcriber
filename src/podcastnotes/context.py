"""
Working out what the trip was actually about, before anything is rewritten.

Claude drives this itself. It gets the trip description, the raw transcript
and a search tool, and it decides what to look up: a name that sounds wrong,
then the project that name belongs to, then the people on that project. Two or
three levels deep, because the useful evidence is rarely in the first result.
A fixed list of searches written in advance cannot do that, since the thing
worth searching for is usually a garbled word nobody knew was in the recording.

The output is a context map the user can read and edit before it is used
anywhere. That preview matters more than it looks: everything downstream
treats this file as fact, so a wrong entry here becomes a wrong name in a
published document, and this is the last point where it is cheap to fix.

**When Glean is unreachable this stops and says so.** It does not proceed on
thin context. A write-up built without the context map is not a worse
write-up, it is a confident document full of names Claude has never seen, and
that is harder to catch than an error. Transcription is unaffected, so an
outage costs the write-up and not the recording.

No Qt, no network at import time.
"""

import json
from dataclasses import asdict, dataclass, field

from src.podcastnotes import glean
from src.podcastnotes.llm import agent

FILENAME = "context_map.json"

# Enough to go three levels deep on a trip with three sites and a dozen names.
# The budget exists to bound a runaway loop, not to save money.
MAX_SEARCHES = 24

SEARCH_TOOL_DESCRIPTION = (
    "Search the company's internal knowledge (documents, chat, tickets, wikis) "
    "for people, projects, prior meetings and terminology. Returns document "
    "bodies, not just titles. Use short, discriminative keyword queries. Do not "
    "write full sentences and do not use boolean operators."
)

SYSTEM = """You are assembling background context so that a garbled machine transcript of a work
trip can be corrected accurately and written up.

You have a search tool over the company's internal knowledge. Use it repeatedly and adaptively.
Start from the trip description, then follow what you find. When the transcript contains a name,
product, study or acronym that looks mis-heard, search for the real spelling. When a search
reveals a new person or project, search again for that. Go several levels deep rather than
issuing one round of searches and stopping.

Speech-to-text mangles proper nouns above all else, so treat every unusual name in the transcript
as a candidate error worth one search.

Finish by returning ONLY a JSON object, with no prose and no code fence around it, with keys:
  "people": [{"name", "role", "why_relevant", "confidence"}]
  "organisations": [{"name", "what_it_is"}]
  "terms": [{"term", "expansion", "note"}]
  "projects": [{"name", "what_it_is", "dates"}]
  "likely_errors": [{"heard", "probably", "evidence", "confidence"}]
  "hard_dates": [{"date", "what"}]
  "open_questions": [string]

"likely_errors" is applied to the transcript by direct substitution, so both sides must be
written to be substituted rather than to describe the problem:
  - "heard" is the exact wording as it appears in the transcript, and no wider than the part
    that is actually wrong. Do not include surrounding words for context.
  - "probably" is exactly what should replace it, so that swapping one for the other leaves a
    correct sentence and loses nothing. Write "100 to 200", not "the figure should be 100 to 200".
  - If one error appears in several wordings, separate them with " / " on both sides, in the
    same order, so that each heard variant lines up with its own replacement.

"confidence" is one of high, medium, low. Never invent a correction you found no evidence for.
If you suspect something and cannot support it, put it in open_questions instead. An honest gap
is useful; a confident wrong name is published and then repeated."""


class GleanUnavailable(Exception):
    """Glean could not be reached, so there is no context to build from.

    Raised rather than degraded deliberately. See the module docstring: the
    quiet version of this failure is a finished-looking document that is
    wrong about every name in it.
    """


@dataclass
class ContextMap:
    """What the trip was about, as far as the company's knowledge can say."""

    people: list = field(default_factory=list)
    organisations: list = field(default_factory=list)
    terms: list = field(default_factory=list)
    projects: list = field(default_factory=list)
    likely_errors: list = field(default_factory=list)
    hard_dates: list = field(default_factory=list)
    open_questions: list = field(default_factory=list)
    searches: list = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Nothing was found. Worth checking before building on it."""
        return not any(
            (self.people, self.organisations, self.terms, self.projects,
             self.likely_errors, self.hard_dates)
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def save(self, path) -> str:
        import pathlib

        target = pathlib.Path(path)
        if target.is_dir():
            target = target / FILENAME
        target.write_text(self.to_json())
        return str(target)

    @classmethod
    def from_dict(cls, payload: dict) -> "ContextMap":
        """Build from whatever came back, tolerating missing and wrong-typed keys.

        The model is asked for seven keys and usually returns seven. "Usually"
        is not a basis for indexing, and a KeyError here would throw away a
        research loop that otherwise worked.
        """
        fields = cls.__dataclass_fields__
        return cls(**{
            name: list(payload.get(name) or [])
            for name in fields
            if isinstance(payload.get(name, []), (list, type(None)))
        })

    @classmethod
    def load(cls, path) -> "ContextMap":
        import pathlib

        target = pathlib.Path(path)
        if target.is_dir():
            target = target / FILENAME
        return cls.from_dict(json.loads(target.read_text()))


def build(
    description: str,
    transcript: str,
    max_searches: int = MAX_SEARCHES,
    on_search=None,
    client=None,
    search=None,
) -> ContextMap:
    """Research the trip and return the map. Raises GleanUnavailable if it cannot.

    `search` is injectable so this can be tested without a Glean credential,
    and so a caller can wrap it, for instance to filter results.
    """
    search = search or glean.research

    # One real search before spending anything on Claude. An outage found here
    # costs a second; found on the tenth tool call it has already produced a
    # half-built map that looks like a result.
    _preflight(search)

    searched = []

    def run_search(payload: dict) -> str:
        query = (payload or {}).get("query", "").strip()
        if not query:
            return "No query given."
        searched.append(query)
        if on_search:
            on_search(query)
        try:
            hits = search(query)
        except glean.NotConfigured as e:
            # The credential died mid-run. Not a bad query, and every
            # remaining search will fail the same way.
            raise GleanUnavailable(str(e)) from e
        if not hits:
            return f"No results for {query!r}. Try different keywords."
        return json.dumps(hits, ensure_ascii=False)

    tool = agent.Tool(
        name="search_company_knowledge",
        description=SEARCH_TOOL_DESCRIPTION,
        schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A short sequence of highly targeted keywords.",
                },
                "why": {
                    "type": "string",
                    "description": "One sentence on what this search is meant to resolve.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        run=run_search,
    )

    answer = agent.ask(
        _prompt(description, transcript),
        system=SYSTEM,
        tools=[tool],
        max_tool_calls=max_searches,
        client=client,
        # Without this the loop turns the outage into an error result and
        # Claude spends the rest of its budget rediscovering that Glean is
        # down, then answers from nothing.
        fatal=(GleanUnavailable,),
    )

    found = ContextMap.from_dict(_parse(answer.text))
    found.searches = searched
    return found


def _preflight(search):
    """Prove Glean answers at all, with a query no organisation will match."""
    try:
        search("podcastnotes context preflight")
    except glean.NotConfigured as e:
        raise GleanUnavailable(str(e)) from e
    except Exception as e:
        raise GleanUnavailable(f"Glean could not be reached: {e}") from e


def _prompt(description: str, transcript: str) -> str:
    return (
        f"Trip description from the user:\n{description}\n\n"
        f"Raw machine transcript. Speaker labels are the diarizer's guesses and may be "
        f"wrong, so do not rely on them:\n\n{transcript}"
    )


def _parse(text: str) -> dict:
    """The JSON out of whatever the model wrapped it in.

    Asking for bare JSON works nearly always. Nearly is the problem: a stray
    code fence or a sentence of preamble would otherwise discard the entire
    research loop, so the object is located between its outermost braces
    rather than assumed to start at the first character. That covers the code
    fence case for free, which is why there is no separate fence handling
    here; a version that stripped fences as well was dead code and a mutation
    run proved no test could tell whether it ran.

    Malformed JSON returns an empty map rather than raising. The realistic
    cause is a response truncated at `max_tokens` mid-object, which leaves an
    opening brace and no closing one.
    """
    body = text.strip()
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        # Always a dict or a raise: the slice starts with "{" and ends with
        # "}" by construction, so no isinstance check is reachable here.
        return json.loads(body[start : end + 1])
    except json.JSONDecodeError:
        return {}
