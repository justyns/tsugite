"""Compaction writes bidirectional chain pointers across the two JSONL files.

After a session is compacted:
- The NEW file's leading `compaction` event carries `source_session_id`
  pointing back to the predecessor.
- The OLD file gets a final `compacted_into` event pointing forward to the
  successor.

Together these make each JSONL self-describing for chain traversal without
needing `session_store.json`. Web UI consumes both directions.
"""

from contextlib import ExitStack

import pytest

from tests.test_post_compaction_counters import _patches, _seed_session_events, _StubAdapter
from tsugite.daemon.config import AgentConfig
from tsugite.daemon.session_store import SessionStore
from tsugite.history import SessionStorage


@pytest.fixture
def workspace_dir(tmp_path):
    return tmp_path / "workspace"


@pytest.fixture
def history_dir(tmp_path):
    d = tmp_path / "history"
    d.mkdir()
    return d


@pytest.mark.asyncio
async def test_new_session_compaction_event_carries_source_session_id(workspace_dir, history_dir, tmp_path):
    """The leading `compaction` event on the new JSONL must include
    `source_session_id` so the UI back-link bubble can navigate to the
    predecessor without needing `session_start.parent_session`.
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
        new_session = await adapter._compact_session(conv_id, reason="manual")

    assert new_session is not None
    new_events = SessionStorage.load(history_dir / f"{new_session.id}.jsonl").load_events()
    leading_compaction = next(e for e in new_events if e.type == "compaction")
    assert leading_compaction.data.get("source_session_id") == conv_id


@pytest.mark.asyncio
async def test_old_session_gets_compacted_into_terminal_event(workspace_dir, history_dir, tmp_path):
    """The OLD JSONL must end with a `compacted_into` event so the file is
    self-describing: anyone tailing it learns where the conversation continued
    without consulting session_store.json.
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
        new_session = await adapter._compact_session(conv_id, reason="manual")

    assert new_session is not None
    old_events = SessionStorage.load(history_dir / f"{conv_id}.jsonl").load_events()
    assert (
        old_events[-1].type == "compacted_into"
    ), f"expected last event to be 'compacted_into', got {old_events[-1].type}"
    payload = old_events[-1].data
    assert payload["new_session_id"] == new_session.id
    assert payload["reason"] == "manual"
    # Stub split returned 4 old user_inputs, 2 recent user_inputs.
    assert payload["replaced_count"] == 4
    assert payload["retained_count"] == 2


@pytest.mark.asyncio
async def test_compacted_into_not_written_when_nothing_to_compact(workspace_dir, history_dir, tmp_path):
    """Early-exit branch (everything fits in retention budget) returns None
    without rotating. The OLD file must NOT get a `compacted_into` event in
    that case — there is no successor to point at.
    """
    from unittest.mock import AsyncMock, patch

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
        patch("tsugite.daemon.memory.split_events_for_compaction", return_value=([], [])),
        patch("tsugite.daemon.memory.summarize_session", new=fake_summarize),
        patch("tsugite.history.get_history_dir", return_value=history_dir),
        patch("tsugite.history.storage.get_history_dir", return_value=history_dir),
        patch("tsugite.history.storage.get_machine_name", return_value="test"),
        patch("tsugite.hooks.fire_compact_hooks", new_callable=AsyncMock, return_value=[]),
    ):
        result = await adapter._compact_session(conv_id)

    assert result is None
    old_events = SessionStorage.load(history_dir / f"{conv_id}.jsonl").load_events()
    assert all(
        e.type != "compacted_into" for e in old_events
    ), "early-exit must not write a forward pointer when no rotation happened"
