"""`spawn_agent`'s timeout bounds the whole call, not just the child's exit."""

import os
import pathlib
import signal
import subprocess
import sys
import threading
import time

import pytest

from tsugite.tools import agents as agents_tool


class _HangingStdout:
    """A child that produces nothing and never closes its stdout."""

    def __init__(self):
        self.released = threading.Event()

    def __iter__(self):
        return self

    def __next__(self):
        self.released.wait()
        raise StopIteration


class _HangingProc:
    def __init__(self, *_args, **_kwargs):
        self.stdout = _HangingStdout()
        self.stdin = _FakeStdin()
        self.stderr = _FakeStderr()
        self.pid = 424242

    def wait(self, timeout=None):
        raise AssertionError("wait() is reached only after stdout EOF, which never comes here")


class _FakeStdin:
    def write(self, _data):
        pass

    def close(self):
        pass


class _FakeStderr:
    def read(self):
        return ""


@pytest.fixture
def hanging_agent(tmp_path, monkeypatch):
    agent = tmp_path / "stuck.md"
    agent.write_text("---\nname: stuck\nmodel: openai:gpt-4o-mini\nmax_turns: 1\n---\ndo it\n")
    spawned = {"killed": []}

    def fake_popen(*args, **kwargs):
        proc = _HangingProc()
        spawned["proc"] = proc
        return proc

    def fake_kill_tree(proc):
        spawned["killed"].append(proc.pid)
        spawned["proc"].stdout.released.set()

    # spawn_agent imports subprocess inside the function, so patch the module itself.
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(agents_tool, "_kill_process_tree", fake_kill_tree)
    monkeypatch.chdir(tmp_path)
    return agent, spawned


def test_timeout_bounds_a_child_that_never_closes_stdout(hanging_agent):
    agent, spawned = hanging_agent
    result = {}

    def call():
        try:
            agents_tool.spawn_agent(agent_path=str(agent), prompt="go", timeout=1)
        except BaseException as exc:  # noqa: BLE001 - the assertion is on what came out
            result["error"] = exc

    worker = threading.Thread(target=call, daemon=True)
    worker.start()
    # Outer guard so a regression fails the test instead of wedging the suite.
    worker.join(timeout=20)

    assert not worker.is_alive(), "spawn_agent never returned; the timeout did not bound the read loop"
    assert isinstance(result.get("error"), RuntimeError)
    assert "timed out" in str(result["error"])
    assert spawned["killed"] == [spawned["proc"].pid], "the timed-out child's process group was left running"


def _process_alive(pid: int) -> bool:
    # A killed orphan stays a zombie where pid 1 does not reap (a CI container).
    try:
        state = pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state != "Z"


@pytest.fixture
def real_subagent(tmp_path, monkeypatch):
    """Run spawn_agent against a real subprocess whose command the test supplies."""
    agent = tmp_path / "child.md"
    agent.write_text("---\nname: child\nmodel: openai:gpt-4o-mini\nmax_turns: 1\n---\ndo it\n")
    monkeypatch.chdir(tmp_path)

    def use(script_body: str, *args: str):
        script = tmp_path / "child.py"
        script.write_text(script_body)
        cmd = [sys.executable, str(script), *args]
        monkeypatch.setattr(agents_tool, "_build_subagent_cmd", lambda *a, **k: cmd)
        return str(agent)

    return use


def test_timeout_kills_the_subagents_own_children(tmp_path, real_subagent):
    pid_file = tmp_path / "grandchild.pid"
    agent_path = real_subagent(
        "import subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "open(sys.argv[1], 'w').write(str(child.pid))\n"
        "child.wait()\n",
        str(pid_file),
    )

    with pytest.raises(RuntimeError, match="timed out"):
        agents_tool.spawn_agent(agent_path=agent_path, prompt="go", timeout=2)

    grandchild = int(pid_file.read_text())
    try:
        deadline = time.monotonic() + 5
        while _process_alive(grandchild) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _process_alive(grandchild), "a process from the subagent's tree survived the timeout"
    finally:
        if _process_alive(grandchild):
            os.kill(grandchild, signal.SIGKILL)


def test_large_stderr_shows_in_the_error_without_stalling_the_call(real_subagent):
    # The 64 KB pipe buffer must overflow for an undrained stderr to block the child.
    agent_path = real_subagent(
        "import sys\n"
        "sys.stderr.write('MARKER-from-stderr\\n')\n"
        "sys.stderr.write('x' * 200_000)\n"
        "sys.stderr.flush()\n"
        "sys.exit(3)\n"
    )

    with pytest.raises(RuntimeError) as excinfo:
        agents_tool.spawn_agent(agent_path=agent_path, prompt="go", timeout=10)

    message = str(excinfo.value)
    assert "timed out" not in message
    assert "exit code 3" in message
    assert "MARKER-from-stderr" in message
