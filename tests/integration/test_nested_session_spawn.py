"""A child session spawned from inside a parent handle_message must make
progress — the parent must not serialize against the child."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from tsugite.daemon.adapters.base import ChannelContext
from tsugite.daemon.adapters.http import HTTPAgentAdapter
from tsugite.daemon.config import AgentConfig
from tsugite.daemon.session_store import SessionStore


@pytest.fixture
def workspaces(tmp_path):
    ws_parent = tmp_path / "parent_ws"
    ws_child = tmp_path / "child_ws"
    ws_parent.mkdir()
    ws_child.mkdir()
    return ws_parent, ws_child


@pytest.fixture
def session_store(tmp_path):
    return SessionStore(tmp_path / "session_store.json")


def _stub_run_agent_result() -> SimpleNamespace:
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


@pytest.mark.asyncio
async def test_child_handle_message_not_starved_by_parent(monkeypatch, workspaces, session_store):
    ws_parent, ws_child = workspaces
    parent_adapter = HTTPAgentAdapter(
        "parent-agent",
        AgentConfig(workspace_dir=ws_parent, agent_file="default"),
        session_store,
    )
    child_adapter = HTTPAgentAdapter(
        "child-agent",
        AgentConfig(workspace_dir=ws_child, agent_file="default"),
        session_store,
    )

    loop = asyncio.get_running_loop()
    child_started = threading.Event()
    child_future_holder: dict = {}

    def shared_probe(**kwargs) -> SimpleNamespace:
        ws = kwargs["path_context"].workspace_dir
        if ws == ws_parent:
            child_ctx = ChannelContext(
                source="spawn",
                channel_id=None,
                user_id="child-user",
                reply_to="spawn:child-user",
            )
            child_future_holder["future"] = asyncio.run_coroutine_threadsafe(
                child_adapter.handle_message("child-user", "hi from child", child_ctx),
                loop,
            )
            if not child_started.wait(timeout=1.5):
                raise AssertionError(
                    "Child handle_message never reached its probe within 1.5s — "
                    "starved while parent handle_message was running."
                )
            return _stub_run_agent_result()

        if ws == ws_child:
            child_started.set()
            return _stub_run_agent_result()

        raise AssertionError(f"Unexpected workspace in probe: {ws}")

    monkeypatch.setattr("tsugite.daemon.adapters.base.run_agent", shared_probe)

    parent_ctx = ChannelContext(
        source="http",
        channel_id=None,
        user_id="parent-user",
        reply_to="http:parent-user",
    )

    await asyncio.wait_for(
        parent_adapter.handle_message("parent-user", "hi from parent", parent_ctx),
        timeout=5.0,
    )

    child_future = child_future_holder.get("future")
    assert child_future is not None, "Parent probe did not schedule a child handle_message"
    await asyncio.wait_for(asyncio.wrap_future(child_future), timeout=2.0)
