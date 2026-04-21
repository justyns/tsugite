"""H4: concurrent handle_message calls must not race on process-wide os.chdir.

`tsugite/daemon/adapters/base.py` wraps agent execution in a `run_in_workspace`
closure that does `os.chdir(workspace)` then restores the original cwd.
`os.chdir` is process-wide on Linux, not thread-local — so two concurrent
agent runs with different workspaces will clobber each other's cwd, and
subsequent tool calls (bash_run, git, etc.) see the wrong directory.

This test monkeypatches `run_agent` with a probe that sleeps briefly while cwd
is set, then records what cwd and marker files it observed. Under contention
the recorded values cross between adapters.
"""

import asyncio
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest  # noqa: F401

from tsugite.daemon.adapters.http import HTTPAgentAdapter
from tsugite.daemon.adapters.base import ChannelContext
from tsugite.daemon.config import AgentConfig
from tsugite.daemon.session_store import SessionStore


@pytest.fixture
def two_workspaces(tmp_path):
    ws_a = tmp_path / "ws_a"
    ws_b = tmp_path / "ws_b"
    ws_a.mkdir()
    ws_b.mkdir()
    (ws_a / "MARKER_A").write_text("a")
    (ws_b / "MARKER_B").write_text("b")
    return ws_a, ws_b


@pytest.fixture
def session_store(tmp_path):
    return SessionStore(tmp_path / "session_store.json")


def _make_adapter(name: str, workspace: Path, store: SessionStore) -> HTTPAgentAdapter:
    cfg = AgentConfig(workspace_dir=workspace, agent_file="default")
    return HTTPAgentAdapter(name, cfg, store)


@pytest.mark.asyncio
async def test_concurrent_handle_message_does_not_race_on_cwd(
    monkeypatch, two_workspaces, session_store
):
    ws_a, ws_b = two_workspaces

    adapter_a = _make_adapter("agent-a", ws_a, session_store)
    adapter_b = _make_adapter("agent-b", ws_b, session_store)

    observations: list[dict] = []
    obs_lock = threading.Lock()

    def probe_run_agent(**kwargs):
        """Stand-in for run_agent: sleep briefly while cwd is set, then record
        what cwd + files the probe observed. If another concurrent run clobbers
        os.chdir during the sleep, this one sees the wrong workspace."""
        time.sleep(0.1)
        cwd_after_sleep = os.getcwd()
        entries_after_sleep = sorted(p.name for p in Path(".").iterdir())
        with obs_lock:
            observations.append(
                {
                    "expected_workspace": kwargs["path_context"].workspace_dir.name,
                    "cwd_after_sleep": cwd_after_sleep,
                    "entries_after_sleep": entries_after_sleep,
                }
            )
        return SimpleNamespace(
            output="ok",
            token_count=0,
            cost=0,
            execution_steps=[],
            system_message=None,
            attachments=None,
            provider_state={},
            last_input_tokens=0,
        )

    monkeypatch.setattr("tsugite.daemon.adapters.base.run_agent", probe_run_agent)

    ctx_a = ChannelContext(source="http", channel_id=None, user_id="alice", reply_to="http:alice")
    ctx_b = ChannelContext(source="http", channel_id=None, user_id="bob", reply_to="http:bob")

    await asyncio.gather(
        adapter_a.handle_message("alice", "hi", ctx_a),
        adapter_b.handle_message("bob", "hi", ctx_b),
    )

    assert len(observations) == 2, observations

    for obs in observations:
        marker = "MARKER_A" if obs["expected_workspace"] == "ws_a" else "MARKER_B"
        assert marker in obs["entries_after_sleep"], (
            f"Workspace {obs['expected_workspace']} saw {obs['entries_after_sleep']} "
            f"after the concurrent chdir race — expected {marker}. "
            f"Cwd observed: {obs['cwd_after_sleep']}"
        )
