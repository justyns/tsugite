"""Named execution groups: `with tsu_group("..."):` inside agent Python."""

import contextlib

import pytest

from tsugite.core.executor import LocalExecutor
from tsugite.core.subprocess_executor import SubprocessExecutor
from tsugite.core.tools import Tool
from tsugite.events import EventBus, ExecutionGroupEndEvent, ExecutionGroupStartEvent, ToolCallEvent


def _bus_with_capture():
    events = []
    bus = EventBus()
    bus.subscribe(lambda e: events.append(e))
    return bus, events


def _cleanup(executor):
    """Only the subprocess executor has a child process to reap."""
    if hasattr(executor, "cleanup"):
        executor.cleanup()


def _groups(events):
    return [e for e in events if isinstance(e, (ExecutionGroupStartEvent, ExecutionGroupEndEvent))]


@pytest.mark.parametrize("executor_cls", [SubprocessExecutor, LocalExecutor])
@pytest.mark.asyncio
async def test_group_brackets_the_code_it_wraps(executor_cls):
    bus, events = _bus_with_capture()
    executor = executor_cls(event_bus=bus)
    try:
        result = await executor.execute('with tsu_group("fetch issues"):\n    print("working")\n')
        assert result.error is None
        assert "working" in result.output
    finally:
        _cleanup(executor)

    start, end = _groups(events)
    assert (start.title, start.parent_group_id) == ("fetch issues", None)
    assert (end.group_id, end.success, end.error) == (start.group_id, True, None)


@pytest.mark.parametrize("executor_cls", [SubprocessExecutor, LocalExecutor])
@pytest.mark.asyncio
async def test_nested_groups_carry_their_parent(executor_cls):
    bus, events = _bus_with_capture()
    executor = executor_cls(event_bus=bus)
    try:
        result = await executor.execute('with tsu_group("outer"):\n    with tsu_group("inner"):\n        pass\n')
        assert result.error is None
    finally:
        _cleanup(executor)

    outer, inner, inner_end, outer_end = _groups(events)
    assert [outer.title, inner.title] == ["outer", "inner"]
    assert (outer.parent_group_id, inner.parent_group_id) == (None, outer.group_id)
    assert [inner_end.group_id, outer_end.group_id] == [inner.group_id, outer.group_id]


@pytest.mark.asyncio
async def test_sequential_groups_close_back_to_the_root():
    bus, events = _bus_with_capture()
    executor = SubprocessExecutor(event_bus=bus)
    try:
        result = await executor.execute('for _ in range(2):\n    with tsu_group("same"):\n        pass\n')
        assert result.error is None
    finally:
        _cleanup(executor)

    first, first_end, second, second_end = _groups(events)
    assert first.group_id != second.group_id
    assert (first_end.group_id, second_end.group_id) == (first.group_id, second.group_id)
    assert (first.parent_group_id, second.parent_group_id) == (None, None)


@pytest.mark.asyncio
async def test_an_exception_fails_the_group_and_still_propagates():
    bus, events = _bus_with_capture()
    executor = SubprocessExecutor(event_bus=bus)
    try:
        result = await executor.execute('with tsu_group("risky"):\n    raise ValueError("boom")\n')
        assert "boom" in result.error
    finally:
        _cleanup(executor)

    _start, end = _groups(events)
    assert end.success is False
    assert "boom" in end.error


@pytest.mark.parametrize("executor_cls", [SubprocessExecutor, LocalExecutor])
@pytest.mark.asyncio
async def test_tool_calls_inside_a_group_carry_its_id(executor_cls):
    bus, events = _bus_with_capture()
    executor = executor_cls(event_bus=bus)
    echo = Tool(
        name="echo",
        description="Echo a value",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": []},
        function=lambda value="": value,
    )
    executor.set_tools([echo], bus)
    try:
        result = await executor.execute(
            'echo(value="outside")\nwith tsu_group("labelled"):\n    echo(value="inside")\n'
        )
        assert result.error is None
    finally:
        _cleanup(executor)

    start = _groups(events)[0]
    calls = [e for e in events if isinstance(e, ToolCallEvent)]
    assert [c.group_id for c in calls] == [None, start.group_id]
    # Replay reads membership off the stored tool-call records, not the live events.
    assert [c.get("group_id") for c in result.tool_calls] == [None, start.group_id]


def test_group_events_reach_the_jsonl_protocol(capsys):
    """The daemon's session handler subclasses JSONLUIHandler, so these frames are what
    it writes to a session's event log and pushes over SSE to the web UI."""
    import json

    from tsugite.ui.jsonl import JSONLUIHandler

    handler = JSONLUIHandler()
    handler.handle_event(ExecutionGroupStartEvent(group_id="g1", title="fetch issues"))
    handler.handle_event(ExecutionGroupEndEvent(group_id="g1", success=False, duration_ms=12, error="boom"))

    frames = [json.loads(line) for line in capsys.readouterr().out.strip().split("\n") if line]
    assert [f["type"] for f in frames] == ["group_start", "group_end"]
    start, end = frames
    assert (start["title"], start["group_id"]) == ("fetch issues", "g1")
    assert (end["success"], end["error"]) == (False, "boom")


@pytest.mark.asyncio
async def test_groups_ride_the_execution_result_for_replay():
    """The group events never reach the agent's history; what survives a reload is the
    group records on the execution result, which the agent loop stores with
    `code_execution` the same way it stores tool calls."""
    executor = SubprocessExecutor()
    try:
        result = await executor.execute(
            'with tsu_group("outer"):\n'
            '    with tsu_group("inner"):\n'
            "        pass\n"
            'try:\n    with tsu_group("failed"):\n        raise ValueError("boom")\nexcept ValueError:\n    pass\n'
        )
        assert result.error is None
    finally:
        _cleanup(executor)

    outer, inner, failed = result.groups
    assert [g["title"] for g in result.groups] == ["outer", "inner", "failed"]
    assert (outer["parent_group_id"], inner["parent_group_id"]) == (None, outer["group_id"])
    assert (outer["success"], failed["success"]) == (True, False)
    assert "boom" in failed["error"]


@pytest.mark.parametrize("executor_cls", [SubprocessExecutor, LocalExecutor])
@pytest.mark.asyncio
async def test_an_interrupted_group_is_not_recorded_as_successful(executor_cls):
    """The two executors carry separate copies of tsu_group, so anything they could
    disagree on needs pinning in both: `except Exception` closes a cancelled group as a
    success, which is why both copies catch `BaseException`."""
    bus, events = _bus_with_capture()
    executor = executor_cls(event_bus=bus)
    try:
        # The subprocess executor returns the failure on the result; the local one
        # lets a BaseException past. Either way the group must close as failed.
        with contextlib.suppress(KeyboardInterrupt):
            await executor.execute('with tsu_group("interrupted"):\n    raise KeyboardInterrupt("ctrl-c")\n')
    finally:
        _cleanup(executor)

    end = _groups(events)[-1]
    assert end.success is False
    assert "KeyboardInterrupt" in end.error


def test_group_title_and_error_are_masked():
    """A group title is model-authored and its error is exception text from arbitrary
    code in the block, so both can carry a secret the registry handed out."""
    from tsugite.secrets.registry import get_registry

    registry = get_registry()
    registry.register("probe-token", "sk-supersecret")
    try:
        start = ExecutionGroupStartEvent(group_id="g1", title="fetch with sk-supersecret")
        end = ExecutionGroupEndEvent(group_id="g1", success=False, error="HTTPError: 401 for token sk-supersecret")

        assert "sk-supersecret" not in start.title
        assert "sk-supersecret" not in end.error
    finally:
        registry.clear()


def test_the_system_prompt_shows_the_model_how_to_group():
    """Nothing but the prompt teaches the model `tsu_group`, and the worked example is
    the part that got uptake in testing, so a reword needs re-measuring rather than a
    green test."""
    from tsugite.core.agent import build_system_prompt

    prompt = build_system_prompt([])

    assert "with tsu_group(" in prompt
