"""Concurrent handle_message calls must each see their own workspace via
the task-local workspace ContextVar, not clobber a shared process state."""

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tsugite.cli.helpers import get_workspace_dir
from tsugite.daemon.adapters.base import ChannelContext
from tsugite.daemon.adapters.http import HTTPAgentAdapter
from tsugite.daemon.config import AgentConfig
from tsugite.daemon.session_store import SessionStore


@pytest.fixture
def two_workspaces(tmp_path):
    ws_a = tmp_path / "ws_a"
    ws_b = tmp_path / "ws_b"
    ws_a.mkdir()
    ws_b.mkdir()
    return ws_a, ws_b


@pytest.fixture
def session_store(tmp_path):
    return SessionStore(tmp_path / "session_store.json")


def _make_adapter(name: str, workspace: Path, store: SessionStore) -> HTTPAgentAdapter:
    cfg = AgentConfig(workspace_dir=workspace, agent_file="default")
    return HTTPAgentAdapter(name, cfg, store)


@pytest.mark.asyncio
async def test_concurrent_handle_message_sees_own_workspace_via_contextvar(
    monkeypatch, two_workspaces, session_store
):
    ws_a, ws_b = two_workspaces

    adapter_a = _make_adapter("agent-a", ws_a, session_store)
    adapter_b = _make_adapter("agent-b", ws_b, session_store)

    observations: list[dict] = []
    obs_lock = threading.Lock()

    def probe_run_agent(**kwargs):
        """Sleep briefly to interleave with the other run, then record the
        workspace the ContextVar reports for this task."""
        time.sleep(0.1)
        observed_ws = get_workspace_dir()
        with obs_lock:
            observations.append(
                {
                    "expected_workspace": kwargs["path_context"].workspace_dir,
                    "observed_workspace": observed_ws,
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
        assert obs["observed_workspace"] == obs["expected_workspace"], (
            f"Concurrent run for {obs['expected_workspace']} saw "
            f"workspace {obs['observed_workspace']!r} via the CV — expected "
            f"{obs['expected_workspace']!r}. ContextVar isolation regressed."
        )
