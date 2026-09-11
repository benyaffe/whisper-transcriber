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

# The budget exists to bound a runaway loop, not to save money, so it is set
# well clear of what the work actually needs.
#
# The first real Ashford run spent 22 of a 24 search budget, which is not a
# loop finishing early; it is a loop being cut off. A trip with three sites
# and a dozen unfamiliar names is not unusual, and every name is worth a
# search of its own before anything downstream repeats it in a document.
MAX_SEARCHES = 60

SEARCH_TOOL_DESCRIPTION = (
    "Search the company's internal knowledge (documents, chat, tickets, wikis) "
    "for people, projects, prior meetings and terminology. Returns document "
    "bodies, not just titles. Use short, discriminative keyword queries. Do not "
    "write full sentences and do not use boolean operators."
)

# Said once, because two prompts depend on it and the correction stage is what
# actually consumes it. A paraphrase in the second prompt is a paraphrase that
# drifts out of step with `correct._proposals` without anything failing.
SUBSTITUTION_RULES = """"likely_errors" is applied to the transcript by direct substitution, one
row at a time, so every row must be atomic and independently substitutable. This matters more
than it sounds:
  - "heard" is the shortest run of words that is actually wrong, which is usually a single term
    or name. Never a whole sentence. Never two different errors combined into one row: if a
    sentence contains two mistakes, that is two rows.
  - "probably" is exactly what replaces it and nothing else. Swapping one for the other must
    leave a correct sentence and lose no words. Write "100 to 200", not "the figure should be
    100 to 200" and not "they see 100 to 200 a day".
  - Substitution happens inside a single transcript segment, and segments are short: the median
    is about nine words. A span longer than a few words will usually straddle a boundary and
    then match nothing at all, so the whole correction is lost. Prefer "Fairmont" to "St. Bede's
    Fairmont at 22101 Fairmont Road".
  - If one error appears in several wordings, separate them with " / " on both sides, in the
    same order, so that each heard variant lines up with its own replacement."""

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

{substitution_rules}

"confidence" is one of high, medium, low. Never invent a correction you found no evidence for.
If you suspect something and cannot support it, put it in open_questions instead. An honest gap
is useful; a confident wrong name is published and then repeated."""

# A plain replace, not .format(): the prompt is full of JSON braces and
# format() reads every one of them as a field.
SYSTEM = SYSTEM.replace("{substitution_rules}", SUBSTITUTION_RULES)


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


def research(
    prompt: str,
    system: str,
    max_searches: int = MAX_SEARCHES,
    on_search=None,  # (query, None) when a search starts, (query, n) when it returns
    client=None,
    search=None,
) -> tuple:
    """Run the agentic Glean loop and return (the answer text, the queries run).

    Extracted so a second pass gets the same three things rather than a copy of
    them that drifts: the outage rule, the preflight before Claude is paid for
    anything, and the distinction between a dead credential, which is fatal,
    and a bad query, which is not.
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
            on_search(query, None)
        try:
            hits = search(query)
            if on_search:
                on_search(query, len(hits))
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
        prompt,
        system=system,
        tools=[tool],
        max_tool_calls=max_searches,
        client=client,
        # Without this the loop turns the outage into an error result and
        # Claude spends the rest of its budget rediscovering that Glean is
        # down, then answers from nothing.
        fatal=(GleanUnavailable,),
    )
    return answer.text, searched


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
    text, searched = research(
        _prompt(description, transcript),
        SYSTEM,
        max_searches=max_searches,
        on_search=on_search,
        client=client,
        search=search,
    )
    found = ContextMap.from_dict(_parse(text))
    found.searches = searched
    return found


def _key(text) -> str:
    """A comparison key for a name or a phrase: case and spacing do not count."""
    return " ".join(str(text or "").split()).casefold()


def merge(base: ContextMap, found: ContextMap, answered=()) -> ContextMap:
    """Fold a second research pass into the first, losing nothing, repeating nothing.

    The two passes know different amounts. The first read the whole transcript
    cold. The second was seeded by a sentence somebody typed, so it knows more
    about one thing and less about the trip. That asymmetry is the merge rule:
    background is filled in by the second pass and never overwritten by it,
    because a thinner `why_relevant` from a narrower pass is less knowledge
    rather than corrected knowledge.

    **`likely_errors` is the exception, and it is why this function exists.**
    `correct._proposals` sorts longest-heard first with a *stable* sort, so a
    new row appended for words an existing row already claims never fires: the
    old row matches first, and the new one is reported as "not found in the
    transcript". The person's own correction, silently ignored, with a change
    log that says so. Matched on `heard` and replaced in place.

    `answered` names the open questions the person has now settled. They have
    to go, not just stop being asked: `output._shared` feeds this list to the
    writer under "do not present these as settled", so an answered question
    left in makes the rewritten summary hedge about the exact thing they
    just cleared up.
    """
    merged = ContextMap(
        people=_fill(base.people, found.people, "name"),
        organisations=_fill(base.organisations, found.organisations, "name"),
        terms=_fill(base.terms, found.terms, "term"),
        projects=_fill(base.projects, found.projects, "name"),
        likely_errors=_replace(base.likely_errors, found.likely_errors, "heard"),
        hard_dates=_dedupe(base.hard_dates, found.hard_dates, ("date", "what")),
        open_questions=_questions(base.open_questions, found.open_questions, answered),
        searches=list(base.searches) + list(found.searches),
    )
    return merged


def _replace(base: list, found: list, key: str) -> list:
    """Later wins, in place, so the ordering the correction stage relies on holds."""
    out = [dict(row) for row in base]
    index = {_key(row.get(key)): i for i, row in enumerate(out)}
    for row in found:
        at = index.get(_key(row.get(key)))
        if at is None:
            index[_key(row.get(key))] = len(out)
            out.append(dict(row))
        else:
            out[at] = dict(row)
    return out


def _fill(base: list, found: list, key: str) -> list:
    """Union, filling only the gaps. A narrower pass never overwrites a wider one."""
    out = [dict(row) for row in base]
    index = {_key(row.get(key)): i for i, row in enumerate(out)}
    for row in found:
        at = index.get(_key(row.get(key)))
        if at is None:
            index[_key(row.get(key))] = len(out)
            out.append(dict(row))
            continue
        for field, value in row.items():
            if value and not out[at].get(field):
                out[at][field] = value
    return out


def _dedupe(base: list, found: list, keys: tuple) -> list:
    out = [dict(row) for row in base]
    seen = {tuple(_key(row.get(k)) for k in keys) for row in out}
    for row in found:
        signature = tuple(_key(row.get(k)) for k in keys)
        if signature not in seen:
            seen.add(signature)
            out.append(dict(row))
    return out


def _questions(base: list, found: list, answered) -> list:
    settled = {_key(a) for a in (answered or [])}
    out = [q for q in base if _key(q) not in settled]
    known = {_key(q) for q in out}
    for question in found:
        if _key(question) not in known and _key(question) not in settled:
            known.add(_key(question))
            out.append(question)
    return out


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
