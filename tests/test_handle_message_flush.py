"""Regression test for #265 — adapter session-store mutations must persist to disk."""

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


@pytest.fixture
def adapter(tmp_path):
    workspace = tmp_path / "workspace"
    store = SessionStore(tmp_path / "session_store.json")
    config = AgentConfig(workspace_dir=workspace, agent_file="default")
    return _StubAdapter("test-agent", config, store), store


@pytest.mark.asyncio
async def test_handle_message_persists_dirty_state(adapter, monkeypatch):
    """handle_message must flush() so session metadata mutations land on disk."""
    a, store = adapter
    session = Session(
        id="s1", agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id="u1"
    )
    store.create_session(session)

    async def fake_inner(*_args, **_kwargs):
        store.update_token_count("s1", tokens_used=100)
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

    async def fake_inner(*_args, **_kwargs):
        store.update_session("s1", error="boom")
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(a, "_handle_message_inner", fake_inner)

    ctx = ChannelContext(source="test", channel_id="c1", user_id="u1", reply_to="test")
    with pytest.raises(RuntimeError):
        await a.handle_message("u1", "hi", ctx)

    assert not store._dirty

    reloaded = SessionStore(Path(store._path))
    assert reloaded.get_session("s1").error == "boom"
