"""Tests for async agent sessions with review gates."""

import asyncio
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tsugite.daemon.agent_session import (
    AgentSession,
    AgentSessionStore,
    ReviewDecision,
    ReviewGate,
    SessionState,
)

# ── Store: Session CRUD ──


@pytest.fixture
def store(tmp_path):
    return AgentSessionStore(tmp_path / "sessions.json")


@pytest.fixture
def sample_session():
    return AgentSession(id="s1", agent="default", prompt="do stuff")


def test_create_and_get_session(store, sample_session):
    store.create_session(sample_session)
    got = store.get_session("s1")
    assert got.agent == "default"
    assert got.state == SessionState.PENDING.value


def test_create_duplicate_raises(store, sample_session):
    store.create_session(sample_session)
    with pytest.raises(ValueError, match="already exists"):
        store.create_session(sample_session)


def test_get_missing_raises(store):
    with pytest.raises(ValueError, match="not found"):
        store.get_session("nope")


def test_update_session(store, sample_session):
    store.create_session(sample_session)
    updated = store.update_session("s1", state=SessionState.RUNNING.value)
    assert updated.state == SessionState.RUNNING.value
    assert updated.updated_at != sample_session.created_at


def test_list_sessions_filter(store):
    store.create_session(AgentSession(id="a", agent="x", prompt="p", state=SessionState.RUNNING.value))
    store.create_session(AgentSession(id="b", agent="x", prompt="p", state=SessionState.COMPLETED.value))
    assert len(store.list_sessions()) == 2
    assert len(store.list_sessions(state=SessionState.RUNNING.value)) == 1


# ── Store: Review CRUD ──


def test_create_and_get_review(store, sample_session):
    store.create_session(sample_session)
    review = ReviewGate(id="r1", session_id="s1", title="Approve deploy?")
    store.create_review(review)
    got = store.get_review("r1")
    assert got.title == "Approve deploy?"
    assert got.decision == ReviewDecision.PENDING.value
    # Parent session should be in waiting state
    assert store.get_session("s1").state == SessionState.WAITING_FOR_REVIEW.value
    assert store.get_session("s1").current_review_id == "r1"


def test_resolve_review(store, sample_session):
    store.create_session(sample_session)
    store.create_review(ReviewGate(id="r1", session_id="s1", title="OK?"))
    resolved = store.resolve_review("r1", ReviewDecision.APPROVED.value, "lgtm")
    assert resolved.decision == ReviewDecision.APPROVED.value
    assert resolved.reviewer_comment == "lgtm"
    assert resolved.resolved_at is not None


def test_resolve_already_resolved_raises(store, sample_session):
    store.create_session(sample_session)
    store.create_review(ReviewGate(id="r1", session_id="s1", title="OK?"))
    store.resolve_review("r1", ReviewDecision.APPROVED.value)
    with pytest.raises(ValueError, match="already resolved"):
        store.resolve_review("r1", ReviewDecision.DECLINED.value)


def test_list_reviews_filter(store, sample_session):
    store.create_session(sample_session)
    store.create_review(ReviewGate(id="r1", session_id="s1", title="A"))
    store.create_review(ReviewGate(id="r2", session_id="s1", title="B"))
    store.resolve_review("r1", ReviewDecision.APPROVED.value)
    assert len(store.list_reviews()) == 2
    assert len(store.list_reviews(status=ReviewDecision.PENDING.value)) == 1
    assert len(store.list_reviews(session_id="s1")) == 2


# ── Store: Persistence ──


def test_persistence_roundtrip(tmp_path):
    path = tmp_path / "sessions.json"
    store1 = AgentSessionStore(path)
    store1.create_session(AgentSession(id="s1", agent="default", prompt="hi"))
    store1.create_review(ReviewGate(id="r1", session_id="s1", title="Check"))

    # Reload from disk
    store2 = AgentSessionStore(path)
    assert store2.get_session("s1").agent == "default"
    assert store2.get_review("r1").title == "Check"


def test_stale_session_recovery(tmp_path):
    path = tmp_path / "sessions.json"
    store1 = AgentSessionStore(path)
    store1.create_session(AgentSession(id="s1", agent="x", prompt="p", state=SessionState.RUNNING.value))
    store1.create_session(AgentSession(id="s2", agent="x", prompt="p", state=SessionState.WAITING_FOR_REVIEW.value))
    store1.create_session(AgentSession(id="s3", agent="x", prompt="p", state=SessionState.COMPLETED.value))
    store1.create_review(ReviewGate(id="r1", session_id="s2", title="Pending review"))

    # Simulate daemon restart
    store2 = AgentSessionStore(path)
    assert store2.get_session("s1").state == SessionState.INTERRUPTED.value
    assert store2.get_session("s2").state == SessionState.INTERRUPTED.value
    assert store2.get_session("s3").state == SessionState.COMPLETED.value
    # Pending review on interrupted session should be auto-declined
    assert store2.get_review("r1").decision == ReviewDecision.DECLINED.value


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
    adapter.session_manager = MagicMock()
    adapter.resolve_model = MagicMock(return_value="test-model")
    return adapter


@pytest.mark.asyncio
async def test_session_runner_start_and_complete(tmp_path, mock_adapter):
    from tsugite.daemon.session_runner import SessionRunner

    store = AgentSessionStore(tmp_path / "sessions.json")
    adapters = {"default": mock_adapter}
    runner = SessionRunner(store, adapters)

    session = AgentSession(id="s1", agent="default", prompt="test task")
    result = runner.start_session(session)
    assert result.state == SessionState.RUNNING.value
    assert store.get_session("s1").state == SessionState.RUNNING.value

    # Wait for background execution
    await asyncio.sleep(0.5)

    updated = store.get_session("s1")
    assert updated.state == SessionState.COMPLETED.value
    assert updated.result == "done"


@pytest.mark.asyncio
async def test_session_runner_cancel(tmp_path, mock_adapter):
    from tsugite.daemon.session_runner import SessionRunner

    # Make adapter hang so we can cancel it
    hang_event = asyncio.Event()

    async def slow_handler(*args, **kwargs):
        await hang_event.wait()
        return "done"

    mock_adapter.handle_message = slow_handler

    store = AgentSessionStore(tmp_path / "sessions.json")
    runner = SessionRunner(store, {"default": mock_adapter})

    session = AgentSession(id="s1", agent="default", prompt="slow task")
    runner.start_session(session)
    await asyncio.sleep(0.1)

    runner.cancel_session("s1")
    await asyncio.sleep(0.3)

    updated = store.get_session("s1")
    assert updated.state == SessionState.CANCELLED.value


@pytest.mark.asyncio
async def test_session_runner_failure(tmp_path):
    from tsugite.daemon.session_runner import SessionRunner

    adapter = MagicMock()
    adapter.handle_message = AsyncMock(side_effect=RuntimeError("boom"))
    adapter.agent_config = MagicMock()
    adapter.agent_config.workspace_dir = Path("/tmp/test")
    adapter._resolve_agent_path = MagicMock(return_value=Path("/tmp/test/agent.md"))
    adapter.workspace_attachments = []
    adapter.session_manager = MagicMock()

    store = AgentSessionStore(tmp_path / "sessions.json")
    runner = SessionRunner(store, {"default": adapter})

    session = AgentSession(id="s1", agent="default", prompt="fail task")
    runner.start_session(session)
    await asyncio.sleep(0.5)

    updated = store.get_session("s1")
    assert updated.state == SessionState.FAILED.value
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

    store = AgentSessionStore(tmp_path / "sessions.json")
    runner = SessionRunner(store, {"default": mock_adapter}, notify_callback=on_complete)

    session = AgentSession(id="s1", agent="default", prompt="notify test")
    runner.start_session(session)
    await asyncio.wait_for(notify_called.wait(), timeout=2.0)

    assert notify_args["session_id"] == "s1"
    assert notify_args["result"] == "done"


@pytest.mark.asyncio
async def test_session_runner_resolve_review(tmp_path, mock_adapter):
    from tsugite.daemon.session_runner import SessionRunner

    store = AgentSessionStore(tmp_path / "sessions.json")
    runner = SessionRunner(store, {"default": mock_adapter})

    # Create a session and manually add a review
    session = AgentSession(id="s1", agent="default", prompt="review test", state=SessionState.RUNNING.value)
    store.create_session(session)

    review = ReviewGate(id="r1", session_id="s1", title="OK?")
    store.create_review(review)

    # Register a review backend so resolve_review can unblock it
    from tsugite.daemon.session_runner import SessionReviewBackend

    backend = SessionReviewBackend(store, "s1")
    backend._review_events["r1"] = threading.Event()
    runner._review_backends["s1"] = backend

    runner.resolve_review("r1", ReviewDecision.APPROVED.value, "yes")

    resolved = store.get_review("r1")
    assert resolved.decision == ReviewDecision.APPROVED.value

    # Session should be back to running
    assert store.get_session("s1").state == SessionState.RUNNING.value


@pytest.mark.asyncio
async def test_session_runner_event_logging(tmp_path, mock_adapter):
    from tsugite.daemon.session_runner import SessionRunner

    store = AgentSessionStore(tmp_path / "sessions.json")
    runner = SessionRunner(store, {"default": mock_adapter})

    session = AgentSession(id="s1", agent="default", prompt="log test")
    runner.start_session(session)
    await asyncio.sleep(0.5)

    # Should have at least start/complete events
    events = store.read_events("s1")
    types = [e["type"] for e in events]
    assert "session_start" in types
    assert "session_complete" in types


# ── ReviewBackend Tests ──


def test_review_backend_creates_and_blocks(tmp_path):
    from tsugite.daemon.session_runner import SessionReviewBackend

    store = AgentSessionStore(tmp_path / "sessions.json")
    store.create_session(AgentSession(id="s1", agent="x", prompt="p", state=SessionState.RUNNING.value))
    backend = SessionReviewBackend(store, "s1")

    result_holder = {}

    def ask_in_thread():
        result_holder["answer"] = backend.ask_user("Approve?", "yes_no")

    t = threading.Thread(target=ask_in_thread)
    t.start()

    # Wait for review to be created
    import time

    for _ in range(50):
        reviews = store.list_reviews(status=ReviewDecision.PENDING.value, session_id="s1")
        if reviews:
            break
        time.sleep(0.05)
    else:
        pytest.fail("Review was not created in time")

    review = reviews[0]
    # Resolve the review
    store.resolve_review(review.id, ReviewDecision.APPROVED.value, "ok")
    backend._review_events[review.id].set()

    t.join(timeout=5)
    assert result_holder.get("answer") == "approved"


# ── Session Tools Tests ──


@pytest.mark.asyncio
async def test_session_tools_start(tmp_path, mock_adapter):
    """Test start_session by calling runner directly (avoids thread-safe wrapper in tests)."""
    from tsugite.daemon.session_runner import SessionRunner

    store = AgentSessionStore(tmp_path / "sessions.json")
    runner = SessionRunner(store, {"default": mock_adapter})

    session = AgentSession(id="tool-s1", agent="default", prompt="hello")
    result = runner.start_session(session)
    assert result.state == SessionState.RUNNING.value
    assert result.id == "tool-s1"
    await asyncio.sleep(0.5)

    completed = store.get_session("tool-s1")
    assert completed.state == SessionState.COMPLETED.value


@pytest.mark.asyncio
async def test_session_tools_list(tmp_path, mock_adapter):
    from tsugite.daemon.session_runner import SessionRunner

    store = AgentSessionStore(tmp_path / "sessions.json")
    runner = SessionRunner(store, {"default": mock_adapter})

    runner.start_session(AgentSession(id="tool-s1", agent="default", prompt="hello"))
    sessions = store.list_sessions()
    assert len(sessions) >= 1
    await asyncio.sleep(0.5)


@pytest.mark.asyncio
async def test_session_tools_status(tmp_path, mock_adapter):
    from tsugite.daemon.session_runner import SessionRunner

    store = AgentSessionStore(tmp_path / "sessions.json")
    runner = SessionRunner(store, {"default": mock_adapter})

    runner.start_session(AgentSession(id="tool-s1", agent="default", prompt="hello"))
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

    store = AgentSessionStore(tmp_path / "sessions.json")
    runner = SessionRunner(store, {"default": mock_adapter})

    runner.start_session(AgentSession(id="tool-s1", agent="default", prompt="hello"))
    await asyncio.sleep(0.1)

    runner.cancel_session("tool-s1")
    await asyncio.sleep(0.3)

    session = store.get_session("tool-s1")
    assert session.state == SessionState.CANCELLED.value
