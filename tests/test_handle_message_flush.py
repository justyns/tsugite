"""Regression tests for #265 — adapter session-store mutations must persist to
disk and SSE listeners must be notified after each turn."""

from pathlib import Path

import pytest

from tsugite.daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite.daemon.config import AgentConfig
from tsugite.daemon.session_store import Session, SessionSource, SessionStore


class _StubAdapter(BaseAdapter):
    async def start(self):
        pass

    async def stop(self):
        pass


class _RecordingEventBus:
    def __init__(self):
        self.emitted: list[tuple[str, dict]] = []

    def emit(self, event_type, payload=None):
        self.emitted.append((event_type, dict(payload or {})))


@pytest.fixture
def adapter(tmp_path):
    workspace = tmp_path / "workspace"
    store = SessionStore(tmp_path / "session_store.json")
    config = AgentConfig(workspace_dir=workspace, agent_file="default")
    a = _StubAdapter("test-agent", config, store)
    a.event_bus = _RecordingEventBus()
    return a, store


@pytest.mark.asyncio
async def test_handle_message_persists_dirty_state(adapter, monkeypatch):
    """handle_message must flush() so session metadata mutations land on disk."""
    a, store = adapter
    session = Session(
        id="s1", agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id="u1"
    )
    store.create_session(session)

    async def fake_inner(*_args, **kwargs):
        store.update_token_count("s1", tokens_used=100)
        if (st := kwargs.get("_broadcast_state")) is not None:
            st["conv_id"] = "s1"
        return "ok"

    monkeypatch.setattr(a, "_handle_message_inner", fake_inner)

    ctx = ChannelContext(source="test", channel_id="c1", user_id="u1", reply_to="test")
    await a.handle_message("u1", "hi", ctx)

    assert not store._dirty

    reloaded = SessionStore(Path(store._path))
    s = reloaded.get_session("s1")
    assert s.message_count == 1


@pytest.mark.asyncio
async def test_handle_message_flushes_even_when_inner_raises(adapter, monkeypatch):
    """Flush must run on the error path too, otherwise failed-turn state stays stale."""
    a, store = adapter
    session = Session(
        id="s1", agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id="u1"
    )
    store.create_session(session)

    async def fake_inner(*_args, **kwargs):
        store.update_session("s1", error="boom")
        if (st := kwargs.get("_broadcast_state")) is not None:
            st["conv_id"] = "s1"
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(a, "_handle_message_inner", fake_inner)

    ctx = ChannelContext(source="test", channel_id="c1", user_id="u1", reply_to="test")
    with pytest.raises(RuntimeError):
        await a.handle_message("u1", "hi", ctx)

    assert not store._dirty

    reloaded = SessionStore(Path(store._path))
    assert reloaded.get_session("s1").error == "boom"


@pytest.mark.asyncio
async def test_handle_message_broadcasts_history_and_session_update(adapter, monkeypatch):
    """SSE listeners must see history_update + session_update so the web UI refreshes."""
    a, store = adapter
    session = Session(
        id="s1", agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id="u1"
    )
    store.create_session(session)

    async def fake_inner(*_args, **kwargs):
        if (st := kwargs.get("_broadcast_state")) is not None:
            st["conv_id"] = "s1"
        return "ok"

    monkeypatch.setattr(a, "_handle_message_inner", fake_inner)

    ctx = ChannelContext(source="test", channel_id="c1", user_id="u1", reply_to="test")
    await a.handle_message("u1", "hi", ctx)

    types = [t for t, _ in a.event_bus.emitted]
    assert "history_update" in types
    assert "session_update" in types
    session_update = next(p for t, p in a.event_bus.emitted if t == "session_update")
    assert session_update == {"action": "updated", "id": "s1"}


@pytest.mark.asyncio
async def test_handle_message_broadcasts_on_error_path(adapter, monkeypatch):
    """Broadcast must fire on the error path too — UI should see the failed-turn state."""
    a, store = adapter
    session = Session(
        id="s1", agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id="u1"
    )
    store.create_session(session)

    async def fake_inner(*_args, **kwargs):
        if (st := kwargs.get("_broadcast_state")) is not None:
            st["conv_id"] = "s1"
        raise RuntimeError("boom")

    monkeypatch.setattr(a, "_handle_message_inner", fake_inner)

    ctx = ChannelContext(source="test", channel_id="c1", user_id="u1", reply_to="test")
    with pytest.raises(RuntimeError):
        await a.handle_message("u1", "hi", ctx)

    types = [t for t, _ in a.event_bus.emitted]
    assert "history_update" in types
    assert "session_update" in types
