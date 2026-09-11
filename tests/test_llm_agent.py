"""
Tests for the tool-use loop.

The happy path here is the least interesting thing. What these cover is the
set of mistakes that a working-looking loop makes silently: splitting tool
results across messages, losing a whole turn because one tool raised, counting
only the last turn's tokens, and treating a refusal as an empty answer.

Everything is driven by a scripted fake client, so no test needs a credential
and none of them reach the network.

Run with: python -m pytest tests/test_llm_agent.py -v
"""

import pytest

from src.podcastnotes.llm import agent


# --- the fake model -----------------------------------------------------------


class _Text:
    type = "text"

    def __init__(self, text):
        self.text = text


class _ToolUse:
    type = "tool_use"

    def __init__(self, name, tool_input, id="tu_1"):
        self.name = name
        self.input = tool_input
        self.id = id


class _Usage:
    def __init__(self, i=0, o=0, cw=0, cr=0):
        self.input_tokens = i
        self.output_tokens = o
        self.cache_creation_input_tokens = cw
        self.cache_read_input_tokens = cr


class _Details:
    def __init__(self, category, explanation):
        self.category = category
        self.explanation = explanation


class _Message:
    def __init__(self, content, stop_reason="end_turn", usage=None, details=None):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage or _Usage()
        self.stop_details = details


class _FakeClient:
    """Replays a scripted list of responses and records what it was sent."""

    def __init__(self, script):
        self.script = list(script)
        self.sent = []

        outer = self

        class _Stream:
            def __init__(self, kwargs):
                outer.sent.append(kwargs)

            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            def get_final_message(self_inner):
                if not outer.script:
                    raise AssertionError("the loop asked for more turns than were scripted")
                return outer.script.pop(0)

        class _Messages:
            def stream(self, **kwargs):
                return _Stream(kwargs)

        self.messages = _Messages()


def _tool(name="search", run=None, calls=None):
    def default(payload):
        if calls is not None:
            calls.append(payload)
        return "a result"

    return agent.Tool(
        name=name,
        description="d",
        schema={"type": "object", "properties": {}},
        run=run or default,
    )


# --- the loop itself ----------------------------------------------------------


def test_a_plain_answer_comes_straight_back():
    client = _FakeClient([_Message([_Text("  hello  ")])])

    assert agent.ask("hi", client=client).text == "hello"


def test_a_tool_is_run_and_its_answer_fed_back():
    calls = []
    client = _FakeClient([
        _Message([_ToolUse("search", {"query": "nadeau"})], stop_reason="tool_use"),
        _Message([_Text("Dr. Miles Nadeau")]),
    ])

    answer = agent.ask("who", tools=[_tool(calls=calls)], client=client)

    assert calls == [{"query": "nadeau"}]
    assert answer.text == "Dr. Miles Nadeau"
    assert answer.tool_calls == [{"name": "search", "input": {"query": "nadeau"}}]


def test_every_result_from_one_turn_goes_back_in_a_single_message():
    """Splitting them is accepted and quietly trains Claude out of asking for
    several tools at once, which slows every later turn for no visible reason."""
    client = _FakeClient([
        _Message(
            [_ToolUse("search", {"q": "a"}, id="t1"), _ToolUse("search", {"q": "b"}, id="t2")],
            stop_reason="tool_use",
        ),
        _Message([_Text("done")]),
    ])

    agent.ask("go", tools=[_tool()], client=client)

    final = client.sent[-1]["messages"]
    user_turns = [m for m in final if m["role"] == "user"]
    results = [m for m in user_turns if isinstance(m["content"], list)
               and any(b.get("type") == "tool_result" for b in m["content"])]

    assert len(results) == 1, "results were split across messages"
    assert len(results[0]["content"]) == 2


def test_one_failing_tool_does_not_lose_the_others():
    def explode(payload):
        if payload["q"] == "bad":
            raise RuntimeError("query rejected")
        return "fine"

    client = _FakeClient([
        _Message(
            [_ToolUse("search", {"q": "bad"}, id="t1"), _ToolUse("search", {"q": "good"}, id="t2")],
            stop_reason="tool_use",
        ),
        _Message([_Text("done")]),
    ])

    agent.ask("go", tools=[_tool(run=explode)], client=client)

    blocks = client.sent[-1]["messages"][-1]["content"]
    assert len(blocks) == 2
    failed = [b for b in blocks if b.get("is_error")]
    assert len(failed) == 1
    assert "query rejected" in failed[0]["content"]
    assert any(b["content"] == "fine" for b in blocks)


def test_a_fatal_tool_failure_ends_the_run():
    """Right for a bad query, wrong for a revoked credential. Without this the
    caller's own outage rule is swallowed by the retry-friendly behaviour."""
    class Revoked(Exception):
        pass

    client = _FakeClient([
        _Message([_ToolUse("search", {})], stop_reason="tool_use"),
        _Message([_Text("should never be reached")]),
    ])

    def revoked(payload):
        raise Revoked("token gone")

    with pytest.raises(Revoked):
        agent.ask("go", tools=[_tool(run=revoked)], client=client, fatal=(Revoked,))


def test_an_unlisted_failure_is_still_only_an_error_result():
    class Revoked(Exception):
        pass

    client = _FakeClient([
        _Message([_ToolUse("search", {})], stop_reason="tool_use"),
        _Message([_Text("carried on")]),
    ])

    def boom(payload):
        raise ValueError("just a bad query")

    answer = agent.ask("go", tools=[_tool(run=boom)], client=client, fatal=(Revoked,))

    assert answer.text == "carried on"


def test_an_invented_tool_is_reported_rather_than_crashing():
    client = _FakeClient([
        _Message([_ToolUse("no_such_tool", {})], stop_reason="tool_use"),
        _Message([_Text("sorry")]),
    ])

    agent.ask("go", tools=[_tool()], client=client)

    block = client.sent[-1]["messages"][-1]["content"][0]
    assert block["is_error"] is True
    assert "no_such_tool" in block["content"]


# --- budget -------------------------------------------------------------------


def test_the_tool_budget_is_enforced():
    calls = []
    client = _FakeClient([
        _Message([_ToolUse("search", {"n": 1}, id="t1")], stop_reason="tool_use"),
        _Message([_ToolUse("search", {"n": 2}, id="t2")], stop_reason="tool_use"),
        _Message([_Text("stopped")]),
    ])

    answer = agent.ask("go", tools=[_tool(calls=calls)], client=client, max_tool_calls=1)

    assert calls == [{"n": 1}], "the loop ran a tool it had no budget for"
    assert answer.stopped_early is True


def test_running_out_of_budget_asks_for_an_answer_rather_than_raising():
    """A partial context map beats an exception, because the stage after this
    can still do something with the names that were found."""
    client = _FakeClient([
        _Message([_ToolUse("search", {}, id="t1")], stop_reason="tool_use"),
        _Message([_Text("partial answer")]),
    ])

    answer = agent.ask("go", tools=[_tool()], client=client, max_tool_calls=0)

    assert answer.text == "partial answer"
    blocks = client.sent[-1]["messages"][-1]["content"]
    assert "Do not call any more tools" in blocks[0]["content"]


def test_a_model_that_never_finishes_is_cut_off():
    forever = [_Message([_ToolUse("search", {}, id="t")], stop_reason="tool_use")] * 200
    client = _FakeClient(forever)

    answer = agent.ask("go", tools=[_tool()], client=client, max_tool_calls=10_000)

    assert answer.stopped_early is True
    assert answer.spend.turns == agent.MAX_TURNS


# --- spend --------------------------------------------------------------------


def test_spend_is_summed_over_every_turn_not_just_the_last():
    """Reading usage off the final response undercounts a research loop by
    most of what it actually cost."""
    client = _FakeClient([
        _Message([_ToolUse("search", {}, id="t1")], stop_reason="tool_use", usage=_Usage(100, 10)),
        _Message([_ToolUse("search", {}, id="t2")], stop_reason="tool_use", usage=_Usage(200, 20)),
        _Message([_Text("done")], usage=_Usage(300, 30, cw=5, cr=7)),
    ])

    spend = agent.ask("go", tools=[_tool()], client=client).spend

    assert (spend.input, spend.output) == (600, 60)
    assert (spend.cache_write, spend.cache_read) == (5, 7)
    assert spend.turns == 3


# --- refusal ------------------------------------------------------------------


def test_a_refusal_is_raised_not_returned_as_an_empty_answer():
    """A stage that reads a refusal as "nothing found" publishes a confident
    document with a hole in it."""
    client = _FakeClient([
        _Message([], stop_reason="refusal", details=_Details("cyber", "declined"))
    ])

    with pytest.raises(agent.Refused) as caught:
        agent.ask("go", client=client)

    assert caught.value.category == "cyber"


def test_a_refusal_with_no_details_still_raises():
    client = _FakeClient([_Message([], stop_reason="refusal")])

    with pytest.raises(agent.Refused):
        agent.ask("go", client=client)


# --- what gets sent -----------------------------------------------------------


def test_effort_is_set_rather_than_left_at_the_default():
    """The API default is "high". Leaving it there is a decision not to make
    one, and this pipeline writes documents somebody publishes under their own
    name, so it wants the deliberate setting."""
    client = _FakeClient([_Message([_Text("hi")])])

    agent.ask("go", client=client)

    assert client.sent[0]["output_config"]["effort"] == agent.DEFAULT_EFFORT


def test_a_stage_can_ask_for_more_effort_than_the_default():
    client = _FakeClient([_Message([_Text("hi")])])

    agent.ask("go", client=client, effort="max")

    assert client.sent[0]["output_config"]["effort"] == "max"


def test_thinking_is_on():
    """Opus 5 runs adaptive thinking by default, but the request says so
    explicitly, so that a later refactor cannot quietly disable it."""
    client = _FakeClient([_Message([_Text("hi")])])

    agent.ask("go", client=client)

    assert client.sent[0]["thinking"] == {"type": "adaptive"}


def test_the_output_ceiling_leaves_room_for_a_long_document():
    """Hitting max_tokens is a silent quality failure: the document stops and
    reads as though it finished."""
    assert agent.MAX_TOKENS >= 64000


def test_tools_are_only_declared_when_there_are_some():
    """An empty tools list is a 400, so the key has to be absent rather than
    present and empty."""
    client = _FakeClient([_Message([_Text("hi")])])

    agent.ask("go", client=client)

    assert "tools" not in client.sent[0]


def test_content_blocks_are_passed_through_untouched():
    """A caller placing a cache breakpoint needs its blocks to arrive as
    written, not flattened into a string."""
    blocks = [{"type": "text", "text": "big", "cache_control": {"type": "ephemeral"}}]
    client = _FakeClient([_Message([_Text("ok")])])

    agent.ask(blocks, client=client)

    assert client.sent[0]["messages"][0]["content"] == blocks


def test_the_progress_callback_sees_each_call_as_it_happens():
    seen = []
    client = _FakeClient([
        _Message([_ToolUse("search", {"q": "x"})], stop_reason="tool_use"),
        _Message([_Text("done")]),
    ])

    agent.ask("go", tools=[_tool()], client=client, on_tool=lambda n, i: seen.append((n, i)))

    assert seen == [("search", {"q": "x"})]
