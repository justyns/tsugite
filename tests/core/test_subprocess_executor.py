"""Tests for subprocess-based code executor."""

import asyncio
import os
import shutil
import tempfile

import pytest

from tsugite.core.subprocess_executor import SubprocessExecutor
from tsugite.core.tools import Tool
from tsugite.events import EventBus, InfoEvent, ToolCallEvent, ToolResultEvent


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
async def test_exec_wall_clock_timeout_recovers_from_a_hung_tool():
    """A parent-only tool that never returns must not wedge the whole turn - the
    exec wall-clock timeout kills the child and returns an actionable error."""

    async def hang(**kwargs):
        await asyncio.sleep(60)
        return "never"

    tool = _make_tool("hang", hang, parent_only=True)
    executor = SubprocessExecutor(exec_timeout=2)
    executor.set_tools([tool], EventBus())
    # Outer guard so a regression fails fast instead of hanging the suite.
    result = await asyncio.wait_for(executor.execute("hang()"), timeout=20)
    assert result.error is not None
    assert "wall-clock timeout" in result.error


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
async def test_subprocess_inherits_workspace_cwd(tmp_path):
    """Subprocess runs in the configured workspace so relative paths in
    LLM-generated code resolve against it rather than the daemon's cwd."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    executor = SubprocessExecutor(workspace_dir=workspace)
    try:
        result = await executor.execute("import os; print(os.getcwd())")
        assert result.error is None, result.error
        assert str(workspace) in result.output, f"Subprocess cwd is {result.output!r}, expected {workspace!s}"
    finally:
        executor.cleanup()


@pytest.mark.asyncio
async def test_state_persists_between_turns():
    """`state` carries values across turns; plain locals do not."""
    executor = SubprocessExecutor()
    try:
        result1 = await executor.execute("state['x'] = 42")
        assert result1.error is None

        result2 = await executor.execute("print(state['x'])")
        assert result2.error is None
        assert "42" in result2.output
    finally:
        executor.cleanup()


@pytest.mark.asyncio
async def test_turn_namespace_is_fresh():
    """Local assignments in one turn must not leak into the next."""
    executor = SubprocessExecutor()
    try:
        result1 = await executor.execute("x = 42")
        assert result1.error is None

        result2 = await executor.execute("print(x)")
        assert result2.error is not None
        assert "NameError" in result2.error
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
async def test_poll_loop_with_sleeps_does_not_lose_tool_calls():
    """A block that pauses between tool calls (the poll-with-sleeps shape a
    session supervising a job uses) must get every tool result. The old IPC read
    loop abandoned a blocked readline on every 1s idle tick (wait_for cancelled
    the future but the pool thread stayed in readline); an abandoned reader then
    consumed a later request line and dropped it (its future was already
    cancelled), wedging the child on resp.fifo until the exec wall-clock cap."""
    call_log = []

    async def mock_tool(question: str = "") -> str:
        call_log.append(question)
        return f"answer-{len(call_log)}"

    tool = _make_tool("poll_job", mock_tool, parent_only=True)
    event_bus = EventBus()
    executor = SubprocessExecutor(event_bus=event_bus, exec_timeout=20)
    executor.set_tools([tool], event_bus)
    try:
        result = await executor.execute(
            "import time\nfor i in range(3):\n    time.sleep(2.5)\n    print(poll_job(question=f'poll-{i}'))\n"
        )
        assert result.error is None, f"polling with sleeps must not wedge the turn: {result.error}"
        assert call_log == ["poll-0", "poll-1", "poll-2"], f"every poll must reach the parent, got {call_log}"
    finally:
        executor.cleanup()


@pytest.mark.asyncio
async def test_idle_wait_does_not_leak_reader_threads():
    """While the child computes locally, the parent's IPC wait must hold ONE
    blocked reader, not stack a new pool thread every second."""
    import threading

    executor = SubprocessExecutor(exec_timeout=20)
    executor.set_tools([], EventBus())
    try:
        baseline = threading.active_count()
        task = asyncio.get_running_loop().create_task(executor.execute("import time\ntime.sleep(6)\nprint('ok')"))
        await asyncio.sleep(5)
        during = threading.active_count()
        result = await task
        assert result.error is None
        assert during - baseline <= 3, (
            f"IPC idle ticks must not stack reader threads: {during - baseline} new threads during a 6s sleep"
        )
    finally:
        executor.cleanup()


@pytest.mark.asyncio
async def test_final_answer_ipc():
    executor = SubprocessExecutor()
    try:
        result = await executor.execute("final_answer('done')")
        assert result.error is None
        assert result.return_value == "done"
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
async def test_non_serializable_state_value_surfaces_error():
    """Assigning a non-JSON value to state must surface a clear error at turn end."""
    executor = SubprocessExecutor()
    try:
        result = await executor.execute("state['s'] = {1, 2, 3}")
        assert result.error is not None
        assert "s" in result.error
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
        result = await executor.execute("a = greet(name='Alice')\nb = greet(name='Bob')\nprint(a, b)")
        assert result.error is None
        assert "Hello, Alice!" in result.output
        assert "Hello, Bob!" in result.output
    finally:
        executor.cleanup()


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bwrap not installed")
@pytest.mark.asyncio
async def test_sandbox_blocks_filesystem_read():
    """Sandbox can't read files outside workspace (e.g., /etc/passwd)."""
    from pathlib import Path

    from tsugite.core.sandbox import SandboxConfig

    with tempfile.TemporaryDirectory() as workspace:
        config = SandboxConfig(no_network=True)
        executor = SubprocessExecutor(workspace_dir=Path(workspace), sandbox_config=config)
        try:
            result = await executor.execute(
                "import os\n"
                "try:\n"
                "    with __builtins__['open']('/etc/passwd', 'r') as f:\n"
                "        print('READ:', f.read()[:20])\n"
                "except Exception as e:\n"
                "    print(f'BLOCKED: {type(e).__name__}')\n"
            )
            assert result.error is None
            assert "BLOCKED:" in result.output
            assert "READ:" not in result.output
        finally:
            executor.cleanup()


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bwrap not installed")
@pytest.mark.asyncio
async def test_sandbox_blocks_sensitive_dotfiles():
    """Sandbox can't read ~/.ssh, ~/.gnupg, ~/.bashrc even though home is partially visible."""

    from tsugite.core.sandbox import SandboxConfig

    config = SandboxConfig(no_network=True)
    executor = SubprocessExecutor(sandbox_config=config)
    try:
        result = await executor.execute(
            "import os\n"
            "home = os.path.expanduser('~')\n"
            "blocked = 0\n"
            "for name in ['.ssh', '.gnupg', '.bashrc', '.config']:\n"
            "    path = os.path.join(home, name)\n"
            "    if not os.path.exists(path):\n"
            "        blocked += 1\n"
            "    else:\n"
            "        print(f'LEAKED: {name}')\n"
            "print(f'BLOCKED: {blocked}/4')\n"
        )
        assert result.error is None
        assert "LEAKED:" not in result.output
        assert "BLOCKED: 4/4" in result.output
    finally:
        executor.cleanup()


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bwrap not installed")
@pytest.mark.asyncio
async def test_sandbox_write_isolation():
    """Writes to /tmp inside sandbox don't escape to host /tmp."""

    from tsugite.core.sandbox import SandboxConfig

    marker = f"sandbox_escape_test_{os.getpid()}.txt"
    host_path = os.path.join("/tmp", marker)

    config = SandboxConfig(no_network=True)
    executor = SubprocessExecutor(sandbox_config=config)
    try:
        result = await executor.execute(
            f"with __builtins__['open']('/tmp/{marker}', 'w') as f:\n    f.write('escaped')\nprint('WROTE')\n"
        )
        assert result.error is None
        assert "WROTE" in result.output
        assert not os.path.exists(host_path), "File escaped sandbox to host /tmp"
    finally:
        executor.cleanup()
        # Clean up just in case
        if os.path.exists(host_path):
            os.unlink(host_path)


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bwrap not installed")
@pytest.mark.asyncio
async def test_sandbox_pid_namespace():
    """Sandbox process runs in an isolated PID namespace."""
    from tsugite.core.sandbox import SandboxConfig

    config = SandboxConfig(no_network=True)
    executor = SubprocessExecutor(sandbox_config=config)
    try:
        result = await executor.execute("import os\nprint(os.getpid())")
        assert result.error is None
        pid = int(result.output.strip())
        assert pid < 100, f"PID {pid} suggests PID namespace is not isolated"
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


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bwrap not installed")
@pytest.mark.asyncio
async def test_sandboxed_executor_binds_workspace_and_isolates_fs(tmp_path, shell_tools):
    """End-to-end: a sandboxed SubprocessExecutor (resolving the bwrap backend via the
    sandbox seam) runs the shell `run()` tool inside the workspace and cannot read
    outside it."""
    from pathlib import Path

    from tsugite.core.sandbox import SandboxConfig
    from tsugite.core.tools import create_tool_from_tsugite

    (tmp_path / "inside.txt").write_text("hi")
    executor = SubprocessExecutor(workspace_dir=Path(tmp_path), sandbox_config=SandboxConfig(no_network=True))
    executor.set_tools([create_tool_from_tsugite("run")])
    try:
        pwd = await executor.execute("print(run(command='pwd'))")
        assert pwd.error is None, pwd.error
        assert str(tmp_path) in pwd.output

        listing = await executor.execute("print(run(command='ls'))")
        assert "inside.txt" in listing.output

        # /etc/shadow is not bound into the sandbox, so it is unreadable.
        shadow = await executor.execute("print(run(command='cat /etc/shadow 2>&1 || true'))")
        assert "inside.txt" not in shadow.output
        assert any(s in shadow.output for s in ("No such file", "Permission denied", "can't open"))
    finally:
        executor.cleanup()
