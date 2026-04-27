"""Interaction backend must be scoped per session task, so that concurrent
sessions (a parent interactive chat and a nested spawned background session,
or two independent incoming requests) each see their own backend."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tsugite.daemon.adapters.base import ChannelContext
from tsugite.daemon.adapters.http import HTTPAgentAdapter
from tsugite.daemon.config import AgentConfig
from tsugite.daemon.session_runner import SessionRunner
from tsugite.daemon.session_store import Session, SessionStore
from tsugite.interaction import (
    NonInteractiveBackend,
    get_interaction_backend,
    set_interaction_backend,
)


@pytest.fixture
def session_store(tmp_path):
    return SessionStore(tmp_path / "session_store.json")


@pytest.fixture
def workspaces(tmp_path):
    ws_a = tmp_path / "ws_a"
    ws_b = tmp_path / "ws_b"
    ws_a.mkdir()
    ws_b.mkdir()
    return ws_a, ws_b


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
async def test_parent_interactive_backend_survives_nested_session_spawn(monkeypatch, workspaces, session_store):
    """A parent handle_message with an interactive backend must still see that
    backend after a child session runs concurrently and installs its own
    NonInteractiveBackend."""
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

    runner = SessionRunner(
        store=session_store,
        adapters={"child-agent": child_adapter},
    )

    parent_backend = MagicMock(spec=NonInteractiveBackend)
    parent_backend.ask_user = MagicMock(return_value="parent-said-yes")
    parent_backend.name = "parent-backend"

    loop = asyncio.get_running_loop()
    observations: dict = {}

    def shared_probe(**kwargs):
        ws = kwargs["path_context"].workspace_dir
        if ws == ws_parent:
            # Kick off a child session whose _run_session will install its own
            # NonInteractiveBackend. Wait for it to complete before observing
            # the backend that the parent still sees.
            child_session = Session(
                id="",
                agent="child-agent",
                prompt="child hi",
                model=None,
                agent_file=None,
            )
            future = asyncio.run_coroutine_threadsafe(
                _run_child(runner, child_session),
                loop,
            )
            future.result(timeout=5.0)
            observations["parent_backend_after"] = get_interaction_backend()
            return _stub_run_agent_result()
        if ws == ws_child:
            observations["child_backend"] = get_interaction_backend()
            return _stub_run_agent_result()
        raise AssertionError(f"Unexpected workspace: {ws}")

    monkeypatch.setattr("tsugite.daemon.adapters.base.run_agent", shared_probe)

    set_interaction_backend(parent_backend)
    parent_ctx = ChannelContext(
        source="http",
        channel_id=None,
        user_id="parent-user",
        reply_to="http:parent-user",
    )
    await asyncio.wait_for(
        parent_adapter.handle_message("parent-user", "hi", parent_ctx),
        timeout=10.0,
    )

    assert isinstance(observations["child_backend"], NonInteractiveBackend), (
        f"Child session should see its own NonInteractiveBackend, got {observations['child_backend']!r}"
    )
    assert observations["parent_backend_after"] is parent_backend, (
        "Parent's interactive backend was clobbered by the child session's "
        f"set_interaction_backend. Parent now sees: "
        f"{observations['parent_backend_after']!r}"
    )


async def _run_child(runner: SessionRunner, session: Session) -> None:
    result_session = runner.start_session(session)
    task = runner._active_tasks.get(result_session.id)
    if task:
        await task


@pytest.mark.asyncio
async def test_concurrent_sessions_each_see_their_own_backend(monkeypatch, workspaces, session_store):
    """Two sessions running concurrently must each see their own interaction
    backend inside handle_message — one session's set_interaction_backend must
    not leak into the other's context."""
    ws_a, ws_b = workspaces
    adapter_a = HTTPAgentAdapter(
        "agent-a",
        AgentConfig(workspace_dir=ws_a, agent_file="default"),
        session_store,
    )
    adapter_b = HTTPAgentAdapter(
        "agent-b",
        AgentConfig(workspace_dir=ws_b, agent_file="default"),
        session_store,
    )

    runner = SessionRunner(store=session_store, adapters={"agent-a": adapter_a, "agent-b": adapter_b})

    backend_a = MagicMock(spec=NonInteractiveBackend)
    backend_a.tag = "backend-a"
    backend_b = MagicMock(spec=NonInteractiveBackend)
    backend_b.tag = "backend-b"

    sessions_observed: dict[str, object] = {}
    both_entered = asyncio.Event()
    entered_count = 0
    counter_lock = asyncio.Lock()

    def shared_probe(**kwargs):
        ws = kwargs["path_context"].workspace_dir
        which = "a" if ws == ws_a else "b"
        # Overwrite with a session-specific backend, then hand control back to
        # the event loop so the two sessions can interleave.
        set_interaction_backend(backend_a if which == "a" else backend_b)

        import time

        time.sleep(0.05)
        sessions_observed[which] = get_interaction_backend()
        return _stub_run_agent_result()

    monkeypatch.setattr("tsugite.daemon.adapters.base.run_agent", shared_probe)

    ctx_a = ChannelContext(source="http", channel_id=None, user_id="u-a", reply_to="http:u-a")
    ctx_b = ChannelContext(source="http", channel_id=None, user_id="u-b", reply_to="http:u-b")

    await asyncio.gather(
        adapter_a.handle_message("u-a", "hi", ctx_a),
        adapter_b.handle_message("u-b", "hi", ctx_b),
    )

    assert sessions_observed["a"] is backend_a, (
        f"Session A saw {sessions_observed['a']!r}, expected backend_a. "
        "Concurrent session B's set_interaction_backend leaked into A's context."
    )
    assert sessions_observed["b"] is backend_b, f"Session B saw {sessions_observed['b']!r}, expected backend_b."
