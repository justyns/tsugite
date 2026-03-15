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
