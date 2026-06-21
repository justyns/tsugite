"""Spawned daemon sessions / jobs inherit the sandbox policy.

A sandboxed agent stamps a sandbox_override onto the records it spawns; that
override rides session metadata into the adapter chokepoint (resolved by
resolve_sandbox_exec_options, tested in test_sandbox_wiring), keeping the child
sandboxed regardless of the target agent's own config.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from tsugite_daemon.session_store import Session, SessionSource, SessionStore


@pytest.fixture
def mock_adapter():
    adapter = MagicMock()
    adapter.handle_message = AsyncMock(return_value="done")
    adapter.agent_config = MagicMock()
    adapter.agent_config.workspace_dir = Path("/tmp/test")
    adapter._resolve_agent_path = MagicMock(return_value=Path("/tmp/test/agent.md"))
    adapter.session_store = MagicMock()
    return adapter


@pytest.mark.asyncio
async def test_session_run_propagates_sandbox_override(tmp_path, mock_adapter):
    from tsugite_daemon.session_runner import SessionRunner

    store = SessionStore(tmp_path / "session_store.json")
    runner = SessionRunner(store, {"default": mock_adapter})
    override = {"enabled": True, "allow_domains": ["github.com"], "no_network": False}
    session = Session(
        id="s1",
        agent="default",
        source=SessionSource.BACKGROUND.value,
        prompt="task",
        metadata={"sandbox_override": override},
    )
    runner.start_session(session)
    await asyncio.sleep(0.3)

    ctx = mock_adapter.handle_message.call_args[1]["channel_context"]
    assert ctx.metadata["sandbox_override"] == override


@pytest.mark.asyncio
async def test_session_run_no_override_when_unset(tmp_path, mock_adapter):
    from tsugite_daemon.session_runner import SessionRunner

    store = SessionStore(tmp_path / "session_store.json")
    runner = SessionRunner(store, {"default": mock_adapter})
    session = Session(id="s2", agent="default", source=SessionSource.BACKGROUND.value, prompt="task")
    runner.start_session(session)
    await asyncio.sleep(0.3)

    ctx = mock_adapter.handle_message.call_args[1]["channel_context"]
    assert "sandbox_override" not in ctx.metadata


def test_spawn_session_stamps_sandbox_override(monkeypatch):
    """spawn_session (the preferred daemon spawn tool) must stamp the inherited
    policy onto the new session, like start_session does."""
    from types import SimpleNamespace

    from tsugite.agent_runner.helpers import SandboxContext, clear_sandbox_context, set_sandbox_context
    from tsugite.tools import sessions as sessions_mod

    captured = {}

    def fake_call(fn, session, *a, **k):
        captured["session"] = session
        return session  # a Session dataclass; spawn_session does asdict(result)

    monkeypatch.setattr(sessions_mod, "_session_runner", SimpleNamespace(start_session=None))
    monkeypatch.setattr(sessions_mod, "_call", fake_call)
    set_sandbox_context(SandboxContext(allow_domains=["github.com"]))
    try:
        sessions_mod.spawn_session(prompt="do work", agent="researcher")
    finally:
        clear_sandbox_context()

    assert captured["session"].metadata.get("sandbox_override", {}).get("enabled") is True
    assert captured["session"].metadata["sandbox_override"]["allow_domains"] == ["github.com"]


def test_spawn_session_no_override_when_not_sandboxed(monkeypatch):
    from types import SimpleNamespace

    from tsugite.agent_runner.helpers import clear_sandbox_context
    from tsugite.tools import sessions as sessions_mod

    captured = {}

    def fake_call(fn, session, *a, **k):
        captured["session"] = session
        return session

    monkeypatch.setattr(sessions_mod, "_session_runner", SimpleNamespace(start_session=None))
    monkeypatch.setattr(sessions_mod, "_call", fake_call)
    clear_sandbox_context()
    sessions_mod.spawn_session(prompt="do work", agent="researcher")
    assert "sandbox_override" not in captured["session"].metadata


def test_with_sandbox_helper_threads_job_override():
    from tsugite_daemon.job_store import Job
    from tsugite_daemon.jobs_orchestrator import _with_sandbox

    job = Job(id="j1", parent_session_id="p", prompt="x", sandbox_override={"enabled": True})
    assert _with_sandbox(job, {"job_id": "j1"})["sandbox_override"] == {"enabled": True}

    plain = Job(id="j2", parent_session_id="p", prompt="x")
    assert "sandbox_override" not in _with_sandbox(plain, {"job_id": "j2"})


def test_sandbox_context_to_override_roundtrip():
    from tsugite.agent_runner.helpers import (
        SandboxContext,
        clear_sandbox_context,
        sandbox_context_to_override,
        set_sandbox_context,
    )

    clear_sandbox_context()
    assert sandbox_context_to_override() is None
    set_sandbox_context(
        SandboxContext(allow_domains=["x.com"], no_network=True, extra_ro_binds=[Path("/a")], pass_env=["MY_VAR"])
    )
    try:
        ov = sandbox_context_to_override()
    finally:
        clear_sandbox_context()
    assert ov["enabled"] is True
    assert ov["allow_domains"] == ["x.com"]
    assert ov["no_network"] is True
    assert ov["extra_ro_binds"] == ["/a"]  # stringified for JSON metadata
    assert ov["pass_env"] == ["MY_VAR"]


def test_build_sandbox_policy_threads_pass_env():
    """pass_env flows ExecutionOptions -> SandboxContext + SandboxConfig."""
    from tsugite.agent_runner.helpers import build_sandbox_policy
    from tsugite.core.sandbox import sandbox_available
    from tsugite.options import ExecutionOptions

    if not sandbox_available():
        return
    opts = ExecutionOptions(sandbox=True, no_network=True, pass_env=["FOO", "BAR"])
    config, ctx = build_sandbox_policy(opts, workspace_dir=Path("/tmp"))
    assert ctx.pass_env == ["FOO", "BAR"]
    assert config.pass_env == ["FOO", "BAR"]
