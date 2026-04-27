"""Tests for compaction progress broadcasting (turn counts + phase events).

Verifies that when compaction runs, the SSE event stream carries enough
information for the UI to show "summarizing N turns…" instead of a bare
spinner.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from tsugite.daemon.adapters.base import BaseAdapter
from tsugite.daemon.config import AgentConfig
from tsugite.daemon.session_store import SessionStore
from tsugite.history.models import Event


class _StubAdapter(BaseAdapter):
    async def start(self):
        pass

    async def stop(self):
        pass


class _RecordingBus:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def emit(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, dict(payload)))


def _make_adapter(workspace_dir, session_store):
    agent_config = AgentConfig(workspace_dir=workspace_dir, agent_file="default")
    return _StubAdapter("test-agent", agent_config, session_store)


@pytest.fixture
def workspace_dir(tmp_path):
    return tmp_path / "workspace"


@pytest.mark.asyncio
async def test_run_compaction_broadcasts_counts_and_phase(workspace_dir, tmp_path):
    from tsugite.history import SessionStorage

    history_dir = tmp_path / "history"
    history_dir.mkdir()

    store = SessionStore(tmp_path / "session_store.json", context_limits={"test-agent": 128_000})
    session = store.get_or_create_interactive("test-user", "test-agent")
    conv_id = session.id

    session_path = history_dir / f"{conv_id}.jsonl"
    storage = SessionStorage.create(
        agent_name="test-agent",
        model="openai:gpt-4o-mini",
        session_path=session_path,
    )
    for i in range(6):
        storage.record("user_input", text=f"message {i}")
        storage.record("model_response", raw_content=f"reply {i}")

    adapter = _make_adapter(workspace_dir, store)
    bus = _RecordingBus()
    adapter.event_bus = bus

    old_events = [Event(type="user_input", ts=datetime.now(timezone.utc), data={"text": f"old {i}"}) for i in range(4)]
    recent_events = [
        Event(type="user_input", ts=datetime.now(timezone.utc), data={"text": f"recent {i}"}) for i in range(2)
    ]

    async def fake_summarize(messages, model=None, max_context_tokens=None, progress_callback=None):
        if progress_callback:
            progress_callback({"phase": "chunking"})
            progress_callback({"phase": "summarizing", "chunk_index": 1, "chunk_total": 1})
        return "Summary"

    with (
        patch("tsugite.daemon.memory.get_context_limit", return_value=128_000),
        patch("tsugite.daemon.memory.infer_compaction_model", return_value="openai:gpt-4o-mini"),
        patch(
            "tsugite.daemon.memory.split_events_for_compaction",
            return_value=(old_events, recent_events),
        ),
        patch("tsugite.daemon.memory.summarize_session", new=fake_summarize),
        patch("tsugite.history.get_history_dir", return_value=history_dir),
        patch("tsugite.history.storage.get_history_dir", return_value=history_dir),
        patch("tsugite.history.storage.get_machine_name", return_value="test"),
        patch("tsugite.hooks.fire_compact_hooks", new_callable=AsyncMock, return_value=[]),
    ):
        await adapter._run_compaction("test-user", conv_id, reason="manual")

    types = [t for t, _ in bus.events]
    assert "compaction_started" in types
    assert "compaction_progress" in types
    assert "compaction_finished" in types

    started_idx = types.index("compaction_started")
    finished_idx = types.index("compaction_finished")
    assert started_idx < finished_idx

    progress_payloads = [p for t, p in bus.events if t == "compaction_progress"]
    starting = next(p for p in progress_payloads if p["phase"] == "starting")
    assert starting["replaced_count"] == 4
    assert starting["retained_count"] == 2

    phases = [p["phase"] for p in progress_payloads]
    assert "chunking" in phases
    assert "summarizing" in phases
    summarizing = next(p for p in progress_payloads if p["phase"] == "summarizing")
    assert summarizing["chunk_index"] == 1
    assert summarizing["chunk_total"] == 1
    for p in progress_payloads:
        assert p["agent"] == "test-agent"


@pytest.mark.asyncio
async def test_run_compaction_finishes_even_with_no_event_bus(workspace_dir, tmp_path):
    """Adapters without an attached event_bus (e.g. CLI) still complete cleanly."""
    from tsugite.history import SessionStorage

    history_dir = tmp_path / "history"
    history_dir.mkdir()

    store = SessionStore(tmp_path / "session_store.json", context_limits={"test-agent": 128_000})
    session = store.get_or_create_interactive("test-user", "test-agent")
    conv_id = session.id

    session_path = history_dir / f"{conv_id}.jsonl"
    storage = SessionStorage.create(
        agent_name="test-agent",
        model="openai:gpt-4o-mini",
        session_path=session_path,
    )
    storage.record("user_input", text="hi")
    storage.record("model_response", raw_content="hello")

    adapter = _make_adapter(workspace_dir, store)
    assert adapter.event_bus is None

    old_events = [Event(type="user_input", ts=datetime.now(timezone.utc), data={"text": "old"})]
    recent_events = [Event(type="user_input", ts=datetime.now(timezone.utc), data={"text": "recent"})]

    with (
        patch("tsugite.daemon.memory.get_context_limit", return_value=128_000),
        patch("tsugite.daemon.memory.infer_compaction_model", return_value="openai:gpt-4o-mini"),
        patch(
            "tsugite.daemon.memory.split_events_for_compaction",
            return_value=(old_events, recent_events),
        ),
        patch("tsugite.daemon.memory.summarize_session", new_callable=AsyncMock, return_value="Summary"),
        patch("tsugite.history.get_history_dir", return_value=history_dir),
        patch("tsugite.history.storage.get_history_dir", return_value=history_dir),
        patch("tsugite.history.storage.get_machine_name", return_value="test"),
        patch("tsugite.hooks.fire_compact_hooks", new_callable=AsyncMock, return_value=[]),
    ):
        result = await adapter._run_compaction("test-user", conv_id, reason="manual")

    assert isinstance(result, str) and result
