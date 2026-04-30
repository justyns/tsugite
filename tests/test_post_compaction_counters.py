"""Regression tests for the post-compaction session counter bug.

Symptom: after compaction, the new session's `cumulative_tokens`,
`message_count`, and `last_active` stayed at 0 / creation_time despite real
activity. Root cause: `_compact_session` returned `None`, so callers
(`_run_compaction`, HTTP `/compact`, `cmd_compact`) rediscovered the new
session via `get_or_create_interactive(user_id, agent)`. That helper returns
the *default* interactive session for the (user, agent) pair, which is the
wrong session whenever the user compacts a non-default (named) session or a
non-INTERACTIVE session. Counter updates then flowed to the wrong id and
silently no-op'd against the unknown session.

These tests pin the contract that `_compact_session` returns the actual new
`Session`, and that downstream counter updates land on it.
"""

from contextlib import ExitStack
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from tsugite.daemon.adapters.base import BaseAdapter
from tsugite.daemon.config import AgentConfig
from tsugite.daemon.session_store import Session, SessionSource, SessionStore
from tsugite.history.models import Event


class _StubAdapter(BaseAdapter):
    async def start(self):
        pass

    async def stop(self):
        pass


@pytest.fixture
def workspace_dir(tmp_path):
    return tmp_path / "workspace"


@pytest.fixture
def history_dir(tmp_path):
    d = tmp_path / "history"
    d.mkdir()
    return d


def _seed_session_events(session_path, count=6):
    from tsugite.history import SessionStorage

    storage = SessionStorage.create(
        agent_name="test-agent",
        model="anthropic:claude-sonnet-4-5",
        session_path=session_path,
    )
    for i in range(count):
        storage.record("user_input", text=f"message {i}")
        storage.record("model_response", raw_content=f"reply {i}")


def _patches(history_dir):
    """The shared patch list for driving `_compact_session` in isolation."""
    old_events = [
        Event(type="user_input", ts=datetime.now(timezone.utc), data={"text": f"old {i}"}) for i in range(4)
    ]
    recent_events = [
        Event(type="user_input", ts=datetime.now(timezone.utc), data={"text": f"recent {i}"}) for i in range(2)
    ]

    async def fake_summarize(messages, model=None, max_context_tokens=None, progress_callback=None):
        return "Summary"

    return [
        patch("tsugite.daemon.memory.get_context_limit", return_value=200_000),
        patch("tsugite.daemon.memory.infer_compaction_model", return_value="anthropic:claude-3-haiku-20240307"),
        patch(
            "tsugite.daemon.memory.split_events_for_compaction",
            return_value=(old_events, recent_events),
        ),
        patch("tsugite.daemon.memory.summarize_session", new=fake_summarize),
        patch("tsugite.history.get_history_dir", return_value=history_dir),
        patch("tsugite.history.storage.get_history_dir", return_value=history_dir),
        patch("tsugite.history.storage.get_machine_name", return_value="test"),
        patch("tsugite.hooks.fire_compact_hooks", new_callable=AsyncMock, return_value=[]),
    ]


@pytest.mark.asyncio
async def test_compact_session_returns_new_session(workspace_dir, history_dir, tmp_path):
    """`_compact_session` must return the `Session` it created so callers
    don't have to rediscover it via the fragile `_interactive_index`.
    """
    store = SessionStore(tmp_path / "session_store.json", context_limits={"test-agent": 1_000_000})
    session = store.get_or_create_interactive("test-user", "test-agent")
    conv_id = session.id

    _seed_session_events(history_dir / f"{conv_id}.jsonl")

    agent_config = AgentConfig(workspace_dir=workspace_dir, agent_file="default")
    agent_config.context_limit = 1_000_000
    adapter = _StubAdapter("test-agent", agent_config, store)

    with ExitStack() as stack:
        for p in _patches(history_dir):
            stack.enter_context(p)
        new_session = await adapter._compact_session(conv_id)

    assert new_session is not None, "_compact_session should return the new Session it created"
    assert isinstance(new_session, Session)
    assert new_session.id != conv_id, "new session must have a different id from the old one"
    assert store.get_session(new_session.id).id == new_session.id
    assert store.get_session(conv_id).superseded_by == new_session.id


@pytest.mark.asyncio
async def test_post_compaction_counters_update(workspace_dir, history_dir, tmp_path):
    """After `_compact_session`, calling `update_token_count` on the returned
    session id must actually move `cumulative_tokens`, `message_count`, and
    `last_active`. This is the load-bearing assertion: if the returned id is
    wrong (the old bug), the update silently no-ops.
    """
    store = SessionStore(tmp_path / "session_store.json", context_limits={"test-agent": 1_000_000})
    session = store.get_or_create_interactive("test-user", "test-agent")
    conv_id = session.id

    _seed_session_events(history_dir / f"{conv_id}.jsonl")

    agent_config = AgentConfig(workspace_dir=workspace_dir, agent_file="default")
    agent_config.context_limit = 1_000_000
    adapter = _StubAdapter("test-agent", agent_config, store)

    with ExitStack() as stack:
        for p in _patches(history_dir):
            stack.enter_context(p)
        new_session = await adapter._compact_session(conv_id)

    creation_ts = new_session.last_active
    assert new_session.cumulative_tokens == 0
    assert new_session.message_count == 0

    store.update_token_count(new_session.id, 1234)

    refreshed = store.get_session(new_session.id)
    assert refreshed.cumulative_tokens == 1234
    assert refreshed.message_count == 1
    assert refreshed.last_active != creation_ts


@pytest.mark.asyncio
async def test_compact_session_for_named_session_returns_correct_id(workspace_dir, history_dir, tmp_path):
    """The reproducer: the user has a default interactive session AND a named
    interactive session. Compacting the named one must return the named
    successor - not the default-interactive session that
    `get_or_create_interactive` would surface.
    """
    store = SessionStore(tmp_path / "session_store.json", context_limits={"test-agent": 1_000_000})

    # Default interactive session - would be returned by get_or_create_interactive.
    default = store.get_or_create_interactive("alice", "test-agent")

    # Named session the user is actually viewing/compacting.
    named_id = "named-session-A"
    named = Session(
        id=named_id,
        agent="test-agent",
        source=SessionSource.INTERACTIVE.value,
        user_id="alice",
        title="Reading",
    )
    store.create_session(named)
    store.update_token_count(named.id, 100)

    _seed_session_events(history_dir / f"{named_id}.jsonl")

    agent_config = AgentConfig(workspace_dir=workspace_dir, agent_file="default")
    agent_config.context_limit = 1_000_000
    adapter = _StubAdapter("test-agent", agent_config, store)

    with ExitStack() as stack:
        for p in _patches(history_dir):
            stack.enter_context(p)
        new_session = await adapter._compact_session(named_id)

    assert new_session is not None
    assert new_session.id != default.id, (
        "compaction returned the default-interactive session id - the exact regression we are guarding against"
    )
    assert store.get_session(named_id).superseded_by == new_session.id


@pytest.mark.asyncio
async def test_compact_session_returns_none_when_nothing_to_compact(workspace_dir, history_dir, tmp_path):
    """Early-exit branch: when all events fit in the retention budget, no
    rotation happens and `_compact_session` should return `None` so callers
    can keep using the original session id.
    """
    store = SessionStore(tmp_path / "session_store.json", context_limits={"test-agent": 1_000_000})
    session = store.get_or_create_interactive("test-user", "test-agent")
    conv_id = session.id

    _seed_session_events(history_dir / f"{conv_id}.jsonl", count=2)

    agent_config = AgentConfig(workspace_dir=workspace_dir, agent_file="default")
    agent_config.context_limit = 1_000_000
    adapter = _StubAdapter("test-agent", agent_config, store)

    async def fake_summarize(messages, model=None, max_context_tokens=None, progress_callback=None):
        return "Summary"

    with (
        patch("tsugite.daemon.memory.get_context_limit", return_value=200_000),
        patch("tsugite.daemon.memory.infer_compaction_model", return_value="anthropic:claude-3-haiku-20240307"),
        # Empty old_events triggers the early-exit at base.py:753-755.
        patch("tsugite.daemon.memory.split_events_for_compaction", return_value=([], [])),
        patch("tsugite.daemon.memory.summarize_session", new=fake_summarize),
        patch("tsugite.history.get_history_dir", return_value=history_dir),
        patch("tsugite.history.storage.get_history_dir", return_value=history_dir),
        patch("tsugite.history.storage.get_machine_name", return_value="test"),
        patch("tsugite.hooks.fire_compact_hooks", new_callable=AsyncMock, return_value=[]),
    ):
        result = await adapter._compact_session(conv_id)

    assert result is None, "early-exit (nothing to compact) should return None"
    assert store.get_session(conv_id).superseded_by is None
