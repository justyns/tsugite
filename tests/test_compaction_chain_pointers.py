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
from tsugite_daemon.config import AgentConfig
from tsugite_daemon.session_store import SessionStore

from tests.test_post_compaction_counters import _patches, _seed_session_events, _StubAdapter
from tsugite.history import SessionStorage


@pytest.fixture
def workspace_dir(tmp_path):
    return tmp_path / "workspace"


@pytest.fixture
def history_dir(tmp_path):
    from tsugite.history import JsonlHistoryBackend, set_history_backend

    d = tmp_path / "history"
    d.mkdir()
    # These tests assert on JSONL file structure; drive the gateway with jsonl.
    set_history_backend(JsonlHistoryBackend())
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
    assert old_events[-1].type == "compacted_into", (
        f"expected last event to be 'compacted_into', got {old_events[-1].type}"
    )
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
        patch("tsugite_daemon.memory.get_context_limit", return_value=200_000),
        patch("tsugite_daemon.memory.infer_compaction_model", return_value="anthropic:claude-3-haiku-20240307"),
        patch("tsugite_daemon.memory.split_events_for_compaction", return_value=([], [])),
        patch("tsugite_daemon.memory.summarize_session", new=fake_summarize),
        patch("tsugite.history.get_history_dir", return_value=history_dir),
        patch("tsugite.history.storage.get_history_dir", return_value=history_dir),
        patch("tsugite.hooks.fire_compact_hooks", new_callable=AsyncMock, return_value=[]),
    ):
        result = await adapter._compact_session(conv_id)

    assert result is None
    old_events = SessionStorage.load(history_dir / f"{conv_id}.jsonl").load_events()
    assert all(e.type != "compacted_into" for e in old_events), (
        "early-exit must not write a forward pointer when no rotation happened"
    )


@pytest.mark.asyncio
async def test_retained_model_response_loses_provider_session_id(workspace_dir, history_dir, tmp_path):
    """Retained `model_response` events must not carry their `state_delta.session_id`
    forward into the new post-compaction JSONL.

    Otherwise `get_resumable_session_state` (which scans backward for the last
    model_response with a session_id, skipping only events before the compaction
    marker by file index) will find the stale id and the next turn resumes the
    pre-compaction Claude Code session — defeating the point of compaction since
    Claude Code keeps the full server-side conversation history on resume.
    """
    from datetime import datetime, timezone
    from unittest.mock import AsyncMock, patch

    from tsugite.agent_runner.history_integration import get_resumable_session_state
    from tsugite.history.models import Event

    store = SessionStore(tmp_path / "session_store.json", context_limits={"test-agent": 1_000_000})
    session = store.get_or_create_interactive("test-user", "test-agent")
    conv_id = session.id

    _seed_session_events(history_dir / f"{conv_id}.jsonl")

    old_events = [Event(type="user_input", ts=datetime.now(timezone.utc), data={"text": "old"})]
    recent_events = [
        Event(type="user_input", ts=datetime.now(timezone.utc), data={"text": "recent"}),
        Event(
            type="model_response",
            ts=datetime.now(timezone.utc),
            data={
                "raw_content": "reply",
                "state_delta": {
                    "session_id": "pre-compaction-claude-code-id",
                    "compacted": False,
                    "context_window": 1_000_000,
                },
            },
        ),
    ]

    async def fake_summarize(messages, model=None, max_context_tokens=None, progress_callback=None):
        return "Summary"

    agent_config = AgentConfig(workspace_dir=workspace_dir, agent_file="default")
    agent_config.context_limit = 1_000_000
    adapter = _StubAdapter("test-agent", agent_config, store)

    with (
        patch("tsugite_daemon.memory.get_context_limit", return_value=1_000_000),
        patch("tsugite_daemon.memory.infer_compaction_model", return_value="anthropic:claude-3-haiku-20240307"),
        patch("tsugite_daemon.memory.split_events_for_compaction", return_value=(old_events, recent_events)),
        patch("tsugite_daemon.memory.summarize_session", new=fake_summarize),
        patch("tsugite.history.get_history_dir", return_value=history_dir),
        patch("tsugite.history.storage.get_history_dir", return_value=history_dir),
        patch("tsugite.hooks.fire_compact_hooks", new_callable=AsyncMock, return_value=[]),
    ):
        new_session = await adapter._compact_session(conv_id, reason="manual")

    assert new_session is not None
    new_events = SessionStorage.load(history_dir / f"{new_session.id}.jsonl").load_events()
    model_responses = [e for e in new_events if e.type == "model_response"]
    assert len(model_responses) == 1
    state_delta = model_responses[0].data.get("state_delta") or {}
    assert "session_id" not in state_delta, f"Retained model_response leaked pre-compaction session_id: {state_delta}"

    with patch("tsugite.history.storage.get_history_dir", return_value=history_dir):
        info = get_resumable_session_state(new_session.id)
    assert info is None, f"Expected no resume target post-compaction, got {info}"


@pytest.mark.asyncio
async def test_retained_events_preserve_timestamps_and_drop_session_end(workspace_dir, history_dir, tmp_path):
    """Carried-forward retained events must keep their ORIGINAL `ts` (not the
    compaction-spawn time), and per-turn `session_end` markers must not be
    copied mid-file. Otherwise a retained turn's whole timeline collapses into
    the spawn instant and the new session structurally "ends" before it ends.
    """
    from datetime import datetime, timezone
    from unittest.mock import AsyncMock, patch

    from tsugite.history.models import Event

    store = SessionStore(tmp_path / "session_store.json", context_limits={"test-agent": 1_000_000})
    session = store.get_or_create_interactive("test-user", "test-agent")
    conv_id = session.id
    _seed_session_events(history_dir / f"{conv_id}.jsonl")

    old_ts = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    old_events = [Event(type="user_input", ts=datetime.now(timezone.utc), data={"text": "old"})]
    recent_events = [
        Event(type="user_input", ts=old_ts, data={"text": "recent"}),
        Event(type="model_response", ts=old_ts, data={"raw_content": "reply"}),
        Event(type="session_end", ts=old_ts, data={"status": "success"}),
    ]

    async def fake_summarize(messages, model=None, max_context_tokens=None, progress_callback=None):
        return "Summary"

    agent_config = AgentConfig(workspace_dir=workspace_dir, agent_file="default")
    agent_config.context_limit = 1_000_000
    adapter = _StubAdapter("test-agent", agent_config, store)

    with (
        patch("tsugite_daemon.memory.get_context_limit", return_value=1_000_000),
        patch("tsugite_daemon.memory.infer_compaction_model", return_value="anthropic:claude-3-haiku-20240307"),
        patch("tsugite_daemon.memory.split_events_for_compaction", return_value=(old_events, recent_events)),
        patch("tsugite_daemon.memory.summarize_session", new=fake_summarize),
        patch("tsugite.history.get_history_dir", return_value=history_dir),
        patch("tsugite.history.storage.get_history_dir", return_value=history_dir),
        patch("tsugite.hooks.fire_compact_hooks", new_callable=AsyncMock, return_value=[]),
    ):
        new_session = await adapter._compact_session(conv_id, reason="manual")

    assert new_session is not None
    new_events = SessionStorage.load(history_dir / f"{new_session.id}.jsonl").load_events()
    assert all(e.type != "session_end" for e in new_events), "retained session_end must not be copied mid-file"
    carried = [e for e in new_events if e.type in ("user_input", "model_response")]
    assert carried, "expected carried turns in the new session"
    assert all(e.ts == old_ts for e in carried), f"carried events lost their original ts: {[e.ts for e in carried]}"


@pytest.mark.asyncio
async def test_compaction_session_start_records_effective_model_override(workspace_dir, history_dir, tmp_path):
    """When a mid-session model override is active, the new session's
    `session_start.model` must reflect the model that will actually run (the
    override), not the agent's config default — otherwise the post-compaction
    session is born mislabeled.
    """
    store = SessionStore(tmp_path / "session_store.json", context_limits={"test-agent": 1_000_000})
    session = store.get_or_create_interactive("test-user", "test-agent")
    conv_id = session.id
    store.set_model_override(conv_id, "codex_cli:gpt-5.5")
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
    start = next(e for e in new_events if e.type == "session_start")
    assert start.data["model"] == "codex_cli:gpt-5.5"


def test_compact_session_preserves_topic_and_type_metadata(tmp_path):
    """Compaction must carry `topic` (and `type`) metadata to the successor so
    the conversation keeps its subject across the rotation, consistent with how
    `title` is already preserved.
    """
    store = SessionStore(tmp_path / "session_store.json", context_limits={"test-agent": 1_000_000})
    session = store.get_or_create_interactive("test-user", "test-agent")
    store.set_metadata(session.id, "topic", "researching widgets")
    store.set_metadata(session.id, "type", "research")

    new_session = store.compact_session(session.id)

    assert new_session.metadata.get("topic") == "researching widgets"
    assert new_session.metadata.get("type") == "research"


@pytest.mark.asyncio
async def test_compaction_records_summary_token_usage_to_usage_store(workspace_dir, history_dir, tmp_path):
    """Compaction summarization spend must be recorded in the UsageStore (what
    `tsugite usage` reads) under source="compaction" - the same sink normal
    turns use - so compaction cost is visible instead of untracked.
    """
    from datetime import datetime, timezone
    from unittest.mock import AsyncMock, MagicMock, patch

    from tsugite.history.models import Event
    from tsugite.providers.base import CompletionResponse, Usage

    store = SessionStore(tmp_path / "session_store.json", context_limits={"test-agent": 1_000_000})
    session = store.get_or_create_interactive("test-user", "test-agent")
    conv_id = session.id
    _seed_session_events(history_dir / f"{conv_id}.jsonl")

    old_events = [Event(type="user_input", ts=datetime.now(timezone.utc), data={"text": "old"})]
    recent_events = [Event(type="user_input", ts=datetime.now(timezone.utc), data={"text": "recent"})]

    provider = MagicMock()
    provider.acompletion = AsyncMock(
        return_value=CompletionResponse(content="Summary", usage=Usage(prompt_tokens=500, completion_tokens=40))
    )
    provider.count_tokens = MagicMock(return_value=10)
    usage_store = MagicMock()

    agent_config = AgentConfig(workspace_dir=workspace_dir, agent_file="default")
    agent_config.context_limit = 1_000_000
    adapter = _StubAdapter("test-agent", agent_config, store)

    with (
        patch("tsugite_daemon.memory.get_context_limit", return_value=1_000_000),
        patch("tsugite_daemon.memory.infer_compaction_model", return_value="openai:gpt-4o-mini"),
        patch("tsugite_daemon.memory.split_events_for_compaction", return_value=(old_events, recent_events)),
        patch("tsugite.models.get_provider_and_model", return_value=("openai:gpt-4o-mini", provider, "gpt-4o-mini")),
        patch("tsugite.usage.get_usage_store", return_value=usage_store),
        patch("tsugite.history.get_history_dir", return_value=history_dir),
        patch("tsugite.history.storage.get_history_dir", return_value=history_dir),
        patch("tsugite.hooks.fire_compact_hooks", new_callable=AsyncMock, return_value=[]),
    ):
        new_session = await adapter._compact_session(conv_id, reason="manual")

    assert new_session is not None
    usage_store.record.assert_called_once()
    kwargs = usage_store.record.call_args.kwargs
    assert kwargs["source"] == "compaction"
    assert kwargs["session_id"] == conv_id
    assert kwargs["input_tokens"] == 500
    assert kwargs["output_tokens"] == 40
