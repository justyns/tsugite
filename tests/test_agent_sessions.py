"""Tests for async agent sessions."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tsugite.daemon.session_store import (
    Session,
    SessionSource,
    SessionStatus,
    SessionStore,
)

# ── Store: Session CRUD ──


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "session_store.json")


@pytest.fixture
def sample_session():
    return Session(id="s1", agent="default", source=SessionSource.BACKGROUND.value, prompt="do stuff")


def test_create_and_get_session(store, sample_session):
    store.create_session(sample_session)
    got = store.get_session("s1")
    assert got.agent == "default"
    assert got.status == SessionStatus.RUNNING.value or got.status == sample_session.status


def test_create_duplicate_raises(store, sample_session):
    store.create_session(sample_session)
    with pytest.raises(ValueError, match="already exists"):
        store.create_session(sample_session)


def test_get_missing_raises(store):
    with pytest.raises(ValueError, match="not found"):
        store.get_session("nope")


def test_update_session(store, sample_session):
    store.create_session(sample_session)
    updated = store.update_session("s1", status=SessionStatus.RUNNING.value)
    assert updated.status == SessionStatus.RUNNING.value


def test_list_sessions_filter(store):
    store.create_session(
        Session(
            id="a", agent="x", source=SessionSource.BACKGROUND.value, status=SessionStatus.RUNNING.value, prompt="p"
        )
    )
    store.create_session(
        Session(
            id="b", agent="x", source=SessionSource.BACKGROUND.value, status=SessionStatus.COMPLETED.value, prompt="p"
        )
    )
    assert len(store.list_sessions()) == 2
    assert len(store.list_sessions(status=SessionStatus.RUNNING.value)) == 1


# ── Store: Persistence ──


def test_persistence_roundtrip(tmp_path):
    path = tmp_path / "session_store.json"
    store1 = SessionStore(path)
    store1.create_session(Session(id="s1", agent="default", source=SessionSource.BACKGROUND.value, prompt="hi"))

    store2 = SessionStore(path)
    assert store2.get_session("s1").agent == "default"


def test_stale_session_recovery(tmp_path):
    path = tmp_path / "session_store.json"
    store1 = SessionStore(path)
    store1.create_session(
        Session(
            id="s1", agent="x", source=SessionSource.BACKGROUND.value, status=SessionStatus.RUNNING.value, prompt="p"
        )
    )
    store1.create_session(
        Session(
            id="s2", agent="x", source=SessionSource.BACKGROUND.value, status=SessionStatus.COMPLETED.value, prompt="p"
        )
    )

    # Simulate daemon restart
    store2 = SessionStore(path)
    assert store2.get_session("s1").status == SessionStatus.FAILED.value
    assert store2.get_session("s2").status == SessionStatus.COMPLETED.value


# ── Store: Event Log ──


def test_event_log(store, sample_session):
    store.create_session(sample_session)
    store.append_event("s1", {"type": "step_start", "name": "research"})
    store.append_event("s1", {"type": "code_execution", "code": "ls"})
    events = store.read_events("s1")
    assert len(events) == 2
    assert events[0]["type"] == "step_start"
    assert store.event_count("s1") == 2


def test_read_events_missing_session(store):
    assert store.read_events("nope") == []
    assert store.event_count("nope") == 0


# ── Store: Progress Summary ──


def test_session_progress_summary_no_events(store, sample_session):
    store.create_session(sample_session)
    summary = store.session_progress_summary("s1")
    assert summary == {
        "turn_count": 0,
        "tool_count": 0,
        "status_text": "Starting...",
        "last_event_time": None,
    }


def test_session_progress_summary_after_session_start(store, sample_session):
    store.create_session(sample_session)
    store.append_event("s1", {"type": "session_start", "timestamp": "2026-04-23T10:00:00+00:00", "agent": "default"})
    summary = store.session_progress_summary("s1")
    assert summary["turn_count"] == 0
    assert summary["tool_count"] == 0
    assert summary["status_text"] == "Starting..."
    assert summary["last_event_time"] == "2026-04-23T10:00:00+00:00"


def test_session_progress_summary_multi_turn_with_tools(store, sample_session):
    store.create_session(sample_session)
    store.append_event("s1", {"type": "session_start", "timestamp": "2026-04-23T10:00:00+00:00"})
    store.append_event("s1", {"type": "turn_start", "turn": 1, "timestamp": "2026-04-23T10:00:01+00:00"})
    store.append_event(
        "s1", {"type": "tool_result", "tool": "bash", "success": True, "timestamp": "2026-04-23T10:00:02+00:00"}
    )
    store.append_event("s1", {"type": "turn_start", "turn": 2, "timestamp": "2026-04-23T10:00:03+00:00"})
    store.append_event(
        "s1", {"type": "tool_result", "tool": "read_file", "success": True, "timestamp": "2026-04-23T10:00:04+00:00"}
    )
    store.append_event(
        "s1", {"type": "tool_result", "tool": "unknown", "success": True, "timestamp": "2026-04-23T10:00:05+00:00"}
    )
    store.append_event("s1", {"type": "thought", "content": "thinking", "timestamp": "2026-04-23T10:00:06+00:00"})
    summary = store.session_progress_summary("s1")
    assert summary["turn_count"] == 2
    assert summary["tool_count"] == 2  # 'unknown' excluded (it's raw code output)
    assert summary["status_text"] == "Thinking..."
    assert summary["last_event_time"] == "2026-04-23T10:00:06+00:00"


def test_session_progress_summary_tool_status(store, sample_session):
    store.create_session(sample_session)
    store.append_event("s1", {"type": "turn_start", "turn": 1, "timestamp": "2026-04-23T10:00:00+00:00"})
    store.append_event(
        "s1", {"type": "tool_result", "tool": "bash", "success": True, "timestamp": "2026-04-23T10:00:01+00:00"}
    )
    summary = store.session_progress_summary("s1")
    assert summary["status_text"] == "Tool: bash"


def test_session_progress_summary_turn_start_status(store, sample_session):
    store.create_session(sample_session)
    store.append_event("s1", {"type": "turn_start", "turn": 3, "timestamp": "2026-04-23T10:00:00+00:00"})
    summary = store.session_progress_summary("s1")
    assert summary["turn_count"] == 3
    assert summary["status_text"] == "Turn 3..."


def test_session_progress_summary_resets_on_session_complete(store, sample_session):
    store.create_session(sample_session)
    store.append_event("s1", {"type": "turn_start", "turn": 1, "timestamp": "2026-04-23T10:00:00+00:00"})
    store.append_event(
        "s1", {"type": "tool_result", "tool": "bash", "success": True, "timestamp": "2026-04-23T10:00:01+00:00"}
    )
    store.append_event(
        "s1", {"type": "session_complete", "result_preview": "done", "timestamp": "2026-04-23T10:00:02+00:00"}
    )
    summary = store.session_progress_summary("s1")
    # Session ended — live progress fields should be cleared so the sidebar shows no stale label.
    assert summary["turn_count"] == 0
    assert summary["tool_count"] == 0
    assert summary["status_text"] == ""
    assert summary["last_event_time"] == "2026-04-23T10:00:02+00:00"


def test_session_progress_summary_resets_on_final_result(store, sample_session):
    """After an interactive turn ends with final_result, the live progress fields clear."""
    store.create_session(sample_session)
    store.append_event("s1", {"type": "turn_start", "turn": 2, "timestamp": "2026-04-23T10:00:00+00:00"})
    store.append_event(
        "s1", {"type": "reasoning_content", "content": "thinking...", "timestamp": "2026-04-23T10:00:01+00:00"}
    )
    store.append_event("s1", {"type": "final_result", "result": "ok", "timestamp": "2026-04-23T10:00:02+00:00"})
    summary = store.session_progress_summary("s1")
    assert summary["turn_count"] == 0
    assert summary["tool_count"] == 0
    assert summary["status_text"] == ""


def test_session_progress_summary_persisted_model_request(store, sample_session):
    """Persisted (non-broadcast) events drive the post-refresh sidebar status.
    `model_request` means the LLM call is in flight; show 'Waiting on LLM...'.
    Without this, after page refresh the status is stuck at 'Starting...'.
    """
    store.create_session(sample_session)
    store.append_event("s1", {"type": "session_start", "timestamp": "2026-04-23T10:00:00+00:00"})
    store.append_event("s1", {"type": "user_input", "text": "hi", "timestamp": "2026-04-23T10:00:01+00:00"})
    store.append_event("s1", {"type": "model_request", "timestamp": "2026-04-23T10:00:02+00:00"})
    summary = store.session_progress_summary("s1")
    assert summary["status_text"] == "Waiting on LLM..."


def test_session_progress_summary_persisted_code_execution(store, sample_session):
    """`code_execution` is persisted when the agent runs raw code (e.g. final_answer).
    Status should reflect that, not stay at 'Starting...'.
    """
    store.create_session(sample_session)
    store.append_event("s1", {"type": "session_start", "timestamp": "2026-04-23T10:00:00+00:00"})
    store.append_event("s1", {"type": "model_response", "timestamp": "2026-04-23T10:00:01+00:00"})
    store.append_event(
        "s1", {"type": "code_execution", "code": "ls", "timestamp": "2026-04-23T10:00:02+00:00"}
    )
    summary = store.session_progress_summary("s1")
    assert summary["status_text"] == "Running code..."


def test_session_progress_summary_persisted_tool_invocation(store, sample_session):
    """`tool_invocation` is the persisted form of a named tool call. Mirror the
    'Tool: <name>' wording the broadcast `tool_result` already uses.
    """
    store.create_session(sample_session)
    store.append_event("s1", {"type": "turn_start", "turn": 1, "timestamp": "2026-04-23T10:00:00+00:00"})
    store.append_event(
        "s1",
        {"type": "tool_invocation", "name": "read_file", "timestamp": "2026-04-23T10:00:01+00:00"},
    )
    summary = store.session_progress_summary("s1")
    assert summary["tool_count"] == 1
    assert summary["status_text"] == "Tool: read_file"


def test_session_progress_summary_resets_on_unprefixed_error(store, sample_session):
    """The HTTP adapter persists `error`/`cancelled` (no `session_` prefix); they must reset progress too."""
    store.create_session(sample_session)
    store.append_event("s1", {"type": "turn_start", "turn": 1, "timestamp": "2026-04-23T10:00:00+00:00"})
    store.append_event("s1", {"type": "reasoning_content", "content": "x", "timestamp": "2026-04-23T10:00:01+00:00"})
    store.append_event("s1", {"type": "error", "error": "boom", "timestamp": "2026-04-23T10:00:02+00:00"})
    summary = store.session_progress_summary("s1")
    assert summary["turn_count"] == 0
    assert summary["status_text"] == ""


def test_session_progress_summary_resets_on_unprefixed_cancelled(store, sample_session):
    store.create_session(sample_session)
    store.append_event("s1", {"type": "turn_start", "turn": 1, "timestamp": "2026-04-23T10:00:00+00:00"})
    store.append_event("s1", {"type": "reasoning_content", "content": "x", "timestamp": "2026-04-23T10:00:01+00:00"})
    store.append_event("s1", {"type": "cancelled", "reason": "user", "timestamp": "2026-04-23T10:00:02+00:00"})
    summary = store.session_progress_summary("s1")
    assert summary["turn_count"] == 0
    assert summary["status_text"] == ""


def test_session_progress_summary_llm_wait_progress(store, sample_session):
    """Heartbeat events surface as 'Waiting on LLM (Ns)' in the sidebar."""
    store.create_session(sample_session)
    store.append_event("s1", {"type": "turn_start", "turn": 1, "timestamp": "2026-04-23T10:00:00+00:00"})
    store.append_event(
        "s1", {"type": "llm_wait_progress", "elapsed_seconds": 15, "timestamp": "2026-04-23T10:00:15+00:00"}
    )
    summary = store.session_progress_summary("s1")
    assert summary["status_text"] == "Waiting on LLM (15s)"


def test_session_progress_summary_llm_wait_progress_zero(store, sample_session):
    """Zero-elapsed heartbeat falls back to a no-elapsed label."""
    store.create_session(sample_session)
    store.append_event(
        "s1", {"type": "llm_wait_progress", "elapsed_seconds": 0, "timestamp": "2026-04-23T10:00:00+00:00"}
    )
    summary = store.session_progress_summary("s1")
    assert summary["status_text"] == "Waiting on LLM..."


def test_session_progress_summary_repopulates_on_next_turn(store, sample_session):
    """After a turn ends, a subsequent turn_start should populate fields again."""
    store.create_session(sample_session)
    store.append_event("s1", {"type": "turn_start", "turn": 1, "timestamp": "2026-04-23T10:00:00+00:00"})
    store.append_event("s1", {"type": "final_result", "result": "ok", "timestamp": "2026-04-23T10:00:01+00:00"})
    store.append_event("s1", {"type": "turn_start", "turn": 2, "timestamp": "2026-04-23T10:01:00+00:00"})
    store.append_event("s1", {"type": "reasoning_content", "content": "y", "timestamp": "2026-04-23T10:01:01+00:00"})
    summary = store.session_progress_summary("s1")
    assert summary["turn_count"] == 2
    assert summary["status_text"] == "Reasoning..."


def test_session_progress_summary_missing_session(store):
    assert store.session_progress_summary("nope") == {
        "turn_count": 0,
        "tool_count": 0,
        "status_text": "Starting...",
        "last_event_time": None,
    }


# ── SessionRunner Tests ──


@pytest.fixture
def mock_adapter():
    adapter = MagicMock()
    adapter.handle_message = AsyncMock(return_value="done")
    adapter.agent_config = MagicMock()
    adapter.agent_config.workspace_dir = Path("/tmp/test")
    adapter._resolve_agent_path = MagicMock(return_value=Path("/tmp/test/agent.md"))
    adapter.workspace_attachments = []
    adapter.session_store = MagicMock()
    adapter.resolve_model = MagicMock(return_value="test-model")
    return adapter


@pytest.mark.asyncio
async def test_session_runner_start_and_complete(tmp_path, mock_adapter):
    from tsugite.daemon.session_runner import SessionRunner

    store = SessionStore(tmp_path / "session_store.json")
    adapters = {"default": mock_adapter}
    runner = SessionRunner(store, adapters)

    session = Session(id="s1", agent="default", source=SessionSource.BACKGROUND.value, prompt="test task")
    result = runner.start_session(session)
    assert result.status == SessionStatus.RUNNING.value
    assert store.get_session("s1").status == SessionStatus.RUNNING.value

    await asyncio.sleep(0.5)

    updated = store.get_session("s1")
    assert updated.status == SessionStatus.COMPLETED.value
    assert updated.result == "done"


@pytest.mark.asyncio
async def test_session_runner_cancel(tmp_path, mock_adapter):
    from tsugite.daemon.session_runner import SessionRunner

    hang_event = asyncio.Event()

    async def slow_handler(*args, **kwargs):
        await hang_event.wait()
        return "done"

    mock_adapter.handle_message = slow_handler

    store = SessionStore(tmp_path / "session_store.json")
    runner = SessionRunner(store, {"default": mock_adapter})

    session = Session(id="s1", agent="default", source=SessionSource.BACKGROUND.value, prompt="slow task")
    runner.start_session(session)
    await asyncio.sleep(0.1)

    runner.cancel_session("s1")
    await asyncio.sleep(0.3)

    updated = store.get_session("s1")
    assert updated.status == SessionStatus.CANCELLED.value


@pytest.mark.asyncio
async def test_session_runner_failure(tmp_path):
    from tsugite.daemon.session_runner import SessionRunner

    adapter = MagicMock()
    adapter.handle_message = AsyncMock(side_effect=RuntimeError("boom"))
    adapter.agent_config = MagicMock()
    adapter.agent_config.workspace_dir = Path("/tmp/test")
    adapter._resolve_agent_path = MagicMock(return_value=Path("/tmp/test/agent.md"))
    adapter.workspace_attachments = []
    adapter.session_store = MagicMock()

    store = SessionStore(tmp_path / "session_store.json")
    runner = SessionRunner(store, {"default": adapter})

    session = Session(id="s1", agent="default", source=SessionSource.BACKGROUND.value, prompt="fail task")
    runner.start_session(session)
    await asyncio.sleep(0.5)

    updated = store.get_session("s1")
    assert updated.status == SessionStatus.FAILED.value
    assert "boom" in updated.error


@pytest.mark.asyncio
async def test_session_runner_notify_callback(tmp_path, mock_adapter):
    from tsugite.daemon.session_runner import SessionRunner

    notify_called = asyncio.Event()
    notify_args = {}

    async def on_complete(session, result):
        notify_args["session_id"] = session.id
        notify_args["result"] = result
        notify_called.set()

    store = SessionStore(tmp_path / "session_store.json")
    runner = SessionRunner(store, {"default": mock_adapter}, notify_callback=on_complete)

    session = Session(id="s1", agent="default", source=SessionSource.BACKGROUND.value, prompt="notify test")
    runner.start_session(session)
    await asyncio.wait_for(notify_called.wait(), timeout=2.0)

    assert notify_args["session_id"] == "s1"
    assert notify_args["result"] == "done"


@pytest.mark.asyncio
async def test_session_runner_event_logging(tmp_path, mock_adapter):
    from tsugite.daemon.session_runner import SessionRunner

    store = SessionStore(tmp_path / "session_store.json")
    runner = SessionRunner(store, {"default": mock_adapter})

    session = Session(id="s1", agent="default", source=SessionSource.BACKGROUND.value, prompt="log test")
    runner.start_session(session)
    await asyncio.sleep(0.5)

    events = store.read_events("s1")
    types = [e["type"] for e in events]
    assert "session_start" in types
    assert "session_complete" in types


# ── Session Tools Tests ──


@pytest.mark.asyncio
async def test_session_tools_start(tmp_path, mock_adapter):
    from tsugite.daemon.session_runner import SessionRunner

    store = SessionStore(tmp_path / "session_store.json")
    runner = SessionRunner(store, {"default": mock_adapter})

    session = Session(id="tool-s1", agent="default", source=SessionSource.BACKGROUND.value, prompt="hello")
    result = runner.start_session(session)
    assert result.status == SessionStatus.RUNNING.value
    assert result.id == "tool-s1"
    await asyncio.sleep(0.5)

    completed = store.get_session("tool-s1")
    assert completed.status == SessionStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_session_tools_list(tmp_path, mock_adapter):
    from tsugite.daemon.session_runner import SessionRunner

    store = SessionStore(tmp_path / "session_store.json")
    runner = SessionRunner(store, {"default": mock_adapter})

    runner.start_session(Session(id="tool-s1", agent="default", source=SessionSource.BACKGROUND.value, prompt="hello"))
    sessions = store.list_sessions()
    assert len(sessions) >= 1
    await asyncio.sleep(0.5)


@pytest.mark.asyncio
async def test_session_tools_status(tmp_path, mock_adapter):
    from tsugite.daemon.session_runner import SessionRunner

    store = SessionStore(tmp_path / "session_store.json")
    runner = SessionRunner(store, {"default": mock_adapter})

    runner.start_session(Session(id="tool-s1", agent="default", source=SessionSource.BACKGROUND.value, prompt="hello"))
    session = store.get_session("tool-s1")
    assert session.agent == "default"
    await asyncio.sleep(0.5)


@pytest.mark.asyncio
async def test_session_tools_cancel(tmp_path, mock_adapter):
    from tsugite.daemon.session_runner import SessionRunner

    hang = asyncio.Event()

    async def slow_handler(*a, **kw):
        await hang.wait()
        return "done"

    mock_adapter.handle_message = slow_handler

    store = SessionStore(tmp_path / "session_store.json")
    runner = SessionRunner(store, {"default": mock_adapter})

    runner.start_session(Session(id="tool-s1", agent="default", source=SessionSource.BACKGROUND.value, prompt="hello"))
    await asyncio.sleep(0.1)

    runner.cancel_session("tool-s1")
    await asyncio.sleep(0.3)

    session = store.get_session("tool-s1")
    assert session.status == SessionStatus.CANCELLED.value
