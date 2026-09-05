"""`spawn_agent`'s timeout bounds the whole call, not just the child's exit."""

import subprocess
import threading

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
        self.killed = False

    def wait(self, timeout=None):
        raise AssertionError("wait() is reached only after stdout EOF, which never comes here")

    def kill(self):
        self.killed = True
        self.stdout.released.set()


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
    spawned = {}

    def fake_popen(*args, **kwargs):
        proc = _HangingProc()
        spawned["proc"] = proc
        return proc

    # spawn_agent imports subprocess inside the function, so patch the module itself.
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
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
    assert spawned["proc"].killed, "the timed-out child was left running"
