"""Tests for subprocess-based code executor."""

import asyncio
import json
import shutil

import pytest

from tsugite.core.subprocess_executor import SubprocessExecutor
from tsugite.events import EventBus, InfoEvent, ToolCallEvent, ToolResultEvent
from tsugite.core.tools import Tool


def _make_tool(name, func, parent_only=False):
    """Helper to create a Tool for testing."""
    tool = Tool(
        name=name,
        description=f"Test tool: {name}",
        parameters={"type": "object", "properties": {}, "required": []},
        function=func,
    )
    tool._parent_only = parent_only
    return tool


@pytest.mark.asyncio
async def test_simple_code_execution():
    executor = SubprocessExecutor()
    try:
        result = await executor.execute("x = 1 + 1\nprint(x)")
        assert result.error is None
        assert "2" in result.output
    finally:
        executor.cleanup()


@pytest.mark.asyncio
async def test_state_persists_between_turns():
    executor = SubprocessExecutor()
    try:
        result1 = await executor.execute("x = 42")
        assert result1.error is None

        result2 = await executor.execute("print(x)")
        assert result2.error is None
        assert "42" in result2.output
    finally:
        executor.cleanup()


@pytest.mark.asyncio
async def test_tool_call_ipc():
    """Parent-only tool is called via IPC from child."""
    call_log = []

    async def mock_tool(question: str = "") -> str:
        call_log.append(question)
        return "yes"

    tool = _make_tool("ask_user", mock_tool, parent_only=True)
    event_bus = EventBus()

    executor = SubprocessExecutor(event_bus=event_bus)
    executor.set_tools([tool], event_bus)
    try:
        result = await executor.execute("answer = ask_user(question='continue?')\nprint(answer)")
        assert result.error is None
        assert "yes" in result.output
        assert call_log == ["continue?"]
    finally:
        executor.cleanup()


@pytest.mark.asyncio
async def test_final_answer_ipc():
    executor = SubprocessExecutor()
    try:
        result = await executor.execute("final_answer('done')")
        assert result.error is None
        assert result.final_answer == "done"
    finally:
        executor.cleanup()


@pytest.mark.asyncio
async def test_send_message_ipc():
    events = []
    event_bus = EventBus()
    event_bus.subscribe(lambda e: events.append(e))

    executor = SubprocessExecutor(event_bus=event_bus)
    try:
        result = await executor.execute("send_message('hello from child')")
        assert result.error is None
        info_events = [e for e in events if isinstance(e, InfoEvent)]
        assert any("hello from child" in e.message for e in info_events)
    finally:
        executor.cleanup()


@pytest.mark.asyncio
async def test_last_expr_eval():
    executor = SubprocessExecutor()
    try:
        result = await executor.execute("x = 5\nx + 3")
        assert result.error is None
        assert "8" in result.output
    finally:
        executor.cleanup()


@pytest.mark.asyncio
async def test_non_serializable_variable():
    """Functions can't be JSON-serialized — should be dropped with warning, not crash."""
    executor = SubprocessExecutor()
    try:
        result = await executor.execute("def my_func(): pass\nx = 42")
        assert result.error is None
        # x should persist, my_func should be dropped
        result2 = await executor.execute("print(x)")
        assert result2.error is None
        assert "42" in result2.output
    finally:
        executor.cleanup()


@pytest.mark.asyncio
async def test_child_crash():
    executor = SubprocessExecutor()
    try:
        result = await executor.execute("import sys; sys.exit(1)")
        assert result.error is not None
    finally:
        executor.cleanup()


@pytest.mark.asyncio
async def test_code_safety_check():
    executor = SubprocessExecutor()
    try:
        result = await executor.execute("f = open('test.txt')")
        assert result.error is not None
        assert "open()" in result.error
    finally:
        executor.cleanup()


@pytest.mark.asyncio
async def test_variables_set_tracking():
    executor = SubprocessExecutor()
    try:
        result = await executor.execute("x = 42\ny = 'hello'\nz = [1, 2, 3]")
        assert result.error is None
        assert "x" in result.variables_set
        assert "y" in result.variables_set
        assert "z" in result.variables_set
    finally:
        executor.cleanup()


@pytest.mark.asyncio
async def test_audit_events_emitted():
    """Non-parent-only tools emit audit events via IPC."""
    events = []
    event_bus = EventBus()
    event_bus.subscribe(lambda e: events.append(e))

    async def mock_read(path: str = "") -> str:
        return "file content"

    tool = _make_tool("read_file", mock_read, parent_only=False)
    executor = SubprocessExecutor(event_bus=event_bus)
    executor.set_tools([tool], event_bus)
    try:
        result = await executor.execute("content = read_file(path='test.txt')")
        assert result.error is None
        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert any(e.tool_name == "read_file" for e in tool_calls)
        assert any(e.tool_name == "read_file" for e in tool_results)
    finally:
        executor.cleanup()


@pytest.mark.asyncio
async def test_send_variables():
    executor = SubprocessExecutor()
    try:
        await executor.send_variables({"injected": 99})
        result = await executor.execute("print(injected)")
        assert result.error is None
        assert "99" in result.output
    finally:
        executor.cleanup()


@pytest.mark.asyncio
async def test_multiple_tool_calls():
    """Child can call parent-only tools multiple times in one turn."""
    async def greet(name: str = "") -> str:
        return f"Hello, {name}!"

    tool = _make_tool("greet", greet, parent_only=True)
    event_bus = EventBus()
    executor = SubprocessExecutor(event_bus=event_bus)
    executor.set_tools([tool], event_bus)
    try:
        result = await executor.execute(
            "a = greet(name='Alice')\nb = greet(name='Bob')\nprint(a, b)"
        )
        assert result.error is None
        assert "Hello, Alice!" in result.output
        assert "Hello, Bob!" in result.output
    finally:
        executor.cleanup()


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bwrap not installed")
@pytest.mark.asyncio
async def test_sandboxed_execution():
    """Integration: SubprocessExecutor actually invokes bwrap when sandbox_config is set."""
    from tsugite.core.sandbox import SandboxConfig

    config = SandboxConfig(no_network=True)
    executor = SubprocessExecutor(sandbox_config=config)
    try:
        result = await executor.execute("import os\nprint('sandboxed:', os.getpid())")
        assert result.error is None
        assert "sandboxed:" in result.output
    finally:
        executor.cleanup()


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bwrap not installed")
@pytest.mark.asyncio
async def test_sandbox_blocks_network():
    """Integration: --no-network actually prevents network access."""
    from tsugite.core.sandbox import SandboxConfig

    config = SandboxConfig(no_network=True)
    executor = SubprocessExecutor(sandbox_config=config)
    try:
        result = await executor.execute(
            "import urllib.request\n"
            "try:\n"
            "    urllib.request.urlopen('http://1.1.1.1', timeout=2)\n"
            "    print('NETWORK_OK')\n"
            "except Exception as e:\n"
            "    print(f'BLOCKED: {type(e).__name__}')\n"
        )
        assert result.error is None
        assert "BLOCKED:" in result.output
        assert "NETWORK_OK" not in result.output
    finally:
        executor.cleanup()
