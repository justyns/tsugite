"""Every turn in a session runs that session's own agent file, whatever path started it."""

from unittest.mock import MagicMock

import pytest
from tsugite_daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite_daemon.config import RuntimeDefaults
from tsugite_daemon.session_runner import SessionRunner
from tsugite_daemon.session_store import Session, SessionSource, SessionStore


class _StubAdapter(BaseAdapter):
    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "default.md").write_text("---\nname: default-agent\n---\n\nHi.\n")
    (ws / "pinned.md").write_text("---\nname: pinned-agent\n---\n\nHi.\n")
    return ws


@pytest.fixture
def adapter(tmp_path, workspace, monkeypatch):
    runtime = RuntimeDefaults(workspace_dir=workspace, agent_file="default", model="openai:gpt-4o-mini")
    adapter = _StubAdapter(runtime, SessionStore(tmp_path / "store.json"))
    monkeypatch.setattr(adapter, "_build_message_context", lambda message, *a, **kw: message)
    monkeypatch.setattr(adapter, "_build_agent_context", lambda *a, **kw: {})
    monkeypatch.setattr(adapter, "_update_skill_ttl", lambda *a, **kw: None)
    return adapter


@pytest.fixture
def run_calls(monkeypatch):
    calls = []

    def fake_run_agent(**kwargs):
        calls.append(kwargs)
        result = MagicMock()
        result.__str__ = lambda self: "ok"
        result.token_count = None
        return result

    monkeypatch.setattr("tsugite_daemon.adapters.base.run_agent", fake_run_agent)
    return calls


def _context(session_id, **metadata):
    return ChannelContext(
        source="http",
        channel_id=None,
        user_id="alice",
        reply_to="http:alice",
        metadata={"conv_id_override": session_id, **metadata},
    )


def _pinned_session(adapter, **fields):
    return adapter.session_store.create_session(
        Session(id="", source=SessionSource.INTERACTIVE.value, user_id="alice", agent_file="pinned", **fields)
    )


@pytest.mark.asyncio
async def test_chat_turn_runs_the_session_agent_file(adapter, run_calls, workspace):
    session = _pinned_session(adapter)

    await adapter.handle_message("alice", "hi", _context(session.id))

    assert run_calls[-1]["agent_path"] == workspace / "pinned.md"


@pytest.mark.asyncio
async def test_session_without_agent_file_runs_the_runtime_agent(adapter, run_calls, workspace):
    session = adapter.session_store.create_session(Session(id="", user_id="alice"))

    await adapter.handle_message("alice", "hi", _context(session.id))

    assert run_calls[-1]["agent_path"] == workspace / "default.md"


@pytest.mark.asyncio
async def test_per_turn_override_beats_the_session_agent_file(adapter, run_calls, workspace):
    session = _pinned_session(adapter)

    await adapter.handle_message("alice", "hi", _context(session.id, agent_file_override=str(workspace / "default.md")))

    assert run_calls[-1]["agent_path"] == workspace / "default.md"


@pytest.mark.asyncio
async def test_reply_to_session_runs_the_session_agent_file(adapter, run_calls, workspace):
    session = _pinned_session(adapter)
    runner = SessionRunner(adapter.session_store, adapter)

    await runner.reply_to_session(session.id, "follow up")

    assert run_calls[-1]["agent_path"] == workspace / "pinned.md"


@pytest.mark.parametrize("rotate", ["compact", "branch"])
def test_compaction_and_branch_keep_the_agent_file(adapter, rotate, monkeypatch):
    session = _pinned_session(adapter)
    store = adapter.session_store
    if rotate == "compact":
        successor = store.compact_session(session.id)
    else:
        monkeypatch.setattr(
            "tsugite_daemon.session_store.get_history_backend",
            lambda: MagicMock(create_branch=lambda *a, **kw: "branch-1"),
        )
        successor = store.branch_session(session.id, at_event_id=1)

    assert successor.agent_file == "pinned"
