"""
Letting Claude use tools, and keeping count of what it spent doing so.

Every AI stage that has to look something up needs the same loop: ask, run
whatever tools came back, hand the results over, ask again. Written once here
rather than once per stage, because the parts that are easy to get wrong are
the same every time and none of them announce themselves.

Three of those are worth naming, since a version without them looks like it
works:

Tool results all go back in a **single** user message. Splitting them across
several is accepted by the API and quietly teaches Claude to stop asking for
more than one tool at a time, so the loop gets slower for no visible reason.

A tool that raises comes back as a result marked `is_error`, not as an
exception. A Glean search can fail for a bad query while the other three in
the same turn succeed, and dropping the whole turn over one of them loses the
other three and the reasoning that asked for them.

Spend is accumulated across every turn. Reading it off the final response
counts one turn out of however many actually happened, which understates a
long research loop by roughly the amount that makes it worth measuring.

No Qt, no network at import time.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from src.podcastnotes.llm.client import MODEL_BEST, build_client

# Generous, because a truncated document is worse than a slow one and the
# streaming path means large values do not risk an HTTP timeout. A trip twice
# the length of Ashford still has room here, and hitting this ceiling is a
# silent quality failure: the document simply stops, and reads as finished.
MAX_TOKENS = 64000

# How hard Claude works before answering. Not the default, which is "high".
#
# The guidance for long-horizon agentic work is high or xhigh rather than max:
# max earns its keep on a single hard question, while a research loop spends
# it re-thinking on every turn for very little. Stages that ask one difficult
# question and want the best possible answer pass "max" explicitly.
DEFAULT_EFFORT = "xhigh"

# A loop that cannot end is worse than one that ends badly. This is not the
# tool budget: it is the backstop for a model that keeps taking turns without
# ever calling a tool or finishing.
MAX_TURNS = 40


class Refused(Exception):
    """Claude declined the request outright.

    Distinct from an error, and from an empty answer. A stage that treats a
    refusal as "no results" writes a confident document with a hole in it,
    which is the failure this whole pipeline is trying to avoid.
    """

    def __init__(self, category: str = "", explanation: str = ""):
        self.category = category
        self.explanation = explanation
        super().__init__(explanation or f"Claude declined the request ({category or 'no reason given'}).")


@dataclass
class Tool:
    """One thing Claude can call.

    `run` takes the parsed input dict and returns a string. Returning a string
    rather than a structure is deliberate: everything the model reads is text
    in the end, and letting each tool choose its own serialisation keeps the
    loop from having opinions about any particular tool's output.
    """

    name: str
    description: str
    schema: dict
    run: Callable[[dict], str]

    def definition(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.schema,
        }


@dataclass
class Spend:
    """What the run cost, in tokens. Dollars are somebody else's problem."""

    input: int = 0
    output: int = 0
    cache_write: int = 0
    cache_read: int = 0
    turns: int = 0

    def add(self, usage):
        self.input += getattr(usage, "input_tokens", 0) or 0
        self.output += getattr(usage, "output_tokens", 0) or 0
        self.cache_write += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.cache_read += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.turns += 1


@dataclass
class Answer:
    """What came back, plus how it got there."""

    text: str
    spend: Spend
    tool_calls: list = field(default_factory=list)
    stopped_early: bool = False


def ask(
    prompt,
    system: str = "",
    tools: Optional[list] = None,
    model: str = MODEL_BEST,
    max_tool_calls: int = 20,
    client=None,
    on_tool: Optional[Callable[[str, dict], None]] = None,
    fatal: tuple = (),
    effort: str = DEFAULT_EFFORT,
) -> Answer:
    """Run a conversation to completion and return the final text.

    `prompt` is either a string or a list of content blocks, so a caller that
    wants a cache breakpoint or several documents can pass blocks without this
    function needing to know why.

    `max_tool_calls` is a budget rather than a hard stop. When it runs out the
    model is told so and asked to answer with what it has, which produces a
    partial context map instead of an exception. A stage that wanted the
    exception can check `stopped_early`.

    `fatal` names the exceptions a tool can raise that must end the run rather
    than become an error result. Turning every failure into a result is right
    for a bad query and wrong for a revoked credential: the first is one
    search out of twenty, the second means every remaining search fails too
    and the model spends its whole budget rediscovering that. Without this the
    caller's own outage rule is silently swallowed by the retry-friendly
    behaviour above it.
    """
    tools = tools or []
    by_name = {t.name: t for t in tools}
    client = client or build_client()

    messages = [{"role": "user", "content": prompt}]
    spend = Spend()
    called = []
    used = 0
    stopped_early = False

    for _ in range(MAX_TURNS):
        request = {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "messages": messages,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
        }
        if system:
            request["system"] = system
        if tools:
            request["tools"] = [t.definition() for t in tools]

        with client.messages.stream(**request) as stream:
            message = stream.get_final_message()
        spend.add(message.usage)

        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            raise Refused(
                getattr(details, "category", "") or "",
                getattr(details, "explanation", "") or "",
            )

        if message.stop_reason != "tool_use":
            return Answer(_text(message), spend, called, stopped_early)

        messages.append({"role": "assistant", "content": message.content})

        # One user message carrying every result, however many there were.
        results = []
        for block in message.content:
            if getattr(block, "type", "") != "tool_use":
                continue

            if used >= max_tool_calls:
                stopped_early = True
                results.append(_result(block.id, _EXHAUSTED))
                continue

            used += 1
            called.append({"name": block.name, "input": block.input})
            if on_tool:
                on_tool(block.name, block.input)

            tool = by_name.get(block.name)
            if tool is None:
                # Claude invented a tool. Saying so is more useful than
                # failing, since it can recover on the next turn.
                results.append(_result(block.id, f"No such tool: {block.name}", error=True))
                continue

            try:
                results.append(_result(block.id, tool.run(block.input)))
            except fatal:
                raise
            except Exception as e:
                results.append(_result(block.id, f"{type(e).__name__}: {e}", error=True))

        messages.append({"role": "user", "content": results})

    # Fell out of the turn cap without a final answer.
    return Answer("", spend, called, stopped_early=True)


_EXHAUSTED = (
    "The tool budget for this task is used up. Do not call any more tools. "
    "Answer now with what you already have."
)


def _result(tool_use_id: str, content: str, error: bool = False) -> dict:
    block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if error:
        block["is_error"] = True
    return block


def _text(message) -> str:
    return "".join(
        b.text for b in message.content if getattr(b, "type", "") == "text"
    ).strip()
