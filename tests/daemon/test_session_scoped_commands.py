"""Session-scoped slash commands (/status, /context, /compact) must act on the
chat they were typed into (the web UI auto-injects session_id), not on the
user's primary/default session."""

from unittest.mock import MagicMock

import pytest

from tsugite.daemon.commands import cmd_context, cmd_status
from tsugite.daemon.session_store import Session, SessionSource, SessionStore


def _adapter(tmp_path):
    store = SessionStore(tmp_path / "session_store.json")
    adapter = MagicMock()
    adapter.session_store = store
    adapter.agent_name = "default"
    adapter.resolve_model.return_value = "claude_code:haiku"
    return adapter, store


def _session(store, sid="sess-open-chat", tokens=1234):
    s = Session(
        id=sid,
        agent="default",
        source=SessionSource.INTERACTIVE.value,
        user_id="user-1",
        cumulative_tokens=tokens,
        message_count=4,
    )
    store.create_session(s)
    return s


@pytest.mark.asyncio
async def test_cmd_status_targets_injected_session(tmp_path):
    """A non-primary chat (the normal multi-session web case) must report ITS
    stats, not 'No active session'."""
    adapter, store = _adapter(tmp_path)
    _session(store)
    result = await cmd_status(adapter=adapter, user_id="user-1", session_id="sess-open-chat")
    assert "No active session" not in result
    assert "Messages: 4" in result


@pytest.mark.asyncio
async def test_cmd_status_falls_back_to_default_session_without_id(tmp_path):
    adapter, store = _adapter(tmp_path)
    result = await cmd_status(adapter=adapter, user_id="user-1")
    assert "No active session" in result


@pytest.mark.asyncio
async def test_cmd_status_stale_session_id_falls_back_gracefully(tmp_path):
    adapter, store = _adapter(tmp_path)
    result = await cmd_status(adapter=adapter, user_id="user-1", session_id="sess-gone")
    assert "No active session" in result


@pytest.mark.asyncio
async def test_cmd_context_targets_injected_session(tmp_path):
    adapter, store = _adapter(tmp_path)
    _session(store)
    result = await cmd_context(adapter=adapter, user_id="user-1", session_id="sess-open-chat")
    # The session exists but has no prompt snapshots yet - the reply must be the
    # 'no data yet' message, NOT 'No active session'.
    assert "No active session" not in result
