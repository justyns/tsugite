"""Replying to a session that is already working.

A parent steering a long-running child would otherwise start a second concurrent
turn on a child that is inside one, polling its own job. The message is held as a
queued message and the turn-end drain runs it as the session's next turn. Without a
queue drain (no HTTP server) it is held as a delivery for the next turn instead.
"""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tsugite_daemon.session_runner import SessionRunner
from tsugite_daemon.session_store import Session, SessionSource, SessionStatus, SessionStore

from tsugite.tools import sessions as session_tools

# Importing a fixture name into a test module shadows every parameter of that name
# (F811), and tests/daemon/conftest.py binds `store` and `runner` to the Jobs fakes.


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "store.json")


@pytest.fixture
def adapter():
    a = MagicMock()
    a.agent_name = "bot"
    a.handle_message = AsyncMock(return_value="ack")
    a.session_store = MagicMock()
    a.resolve_model.return_value = "test-model"
    return a


@pytest.fixture
def runner(store, adapter):
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    r = SessionRunner(store, adapter)
    session_tools.set_session_runner(r, loop)
    yield r
    session_tools.set_session_runner(None)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)
    loop.close()


def _child(store: SessionStore, sid: str = "child") -> Session:
    return store.create_session(Session(id=sid, source=SessionSource.BACKGROUND.value))


def _deliveries(store: SessionStore, sid: str) -> list[dict]:
    return [e for e in store.read_events(sid) if e["type"] == "delivery"]


class TestABusySession:
    def test_the_reply_is_queued_rather_than_run_as_a_second_turn(self, store, runner, adapter):
        child = _child(store)
        store.begin_turn(child.id)

        result = session_tools.session_reply(message="stop, the worktree is wrong", session_id=child.id)

        assert adapter.handle_message.await_count == 0
        assert result["status"] == "queued"
        assert result["response"] is None

    def test_without_a_queue_drain_the_message_reaches_the_session_when_the_turn_ends(self, store, runner):
        child = _child(store)
        store.begin_turn(child.id)
        session_tools.session_reply(message="stop, the worktree is wrong", session_id=child.id)

        assert _deliveries(store, child.id) == []

        store.end_turn(child.id)

        assert [d["message"] for d in _deliveries(store, child.id)] == ["stop, the worktree is wrong"]
        assert store.get_session(child.id).queued_messages == []

    def test_without_a_queue_drain_the_message_records_the_session_it_came_from(self, store, runner):
        caller = _child(store, "caller")
        child = _child(store)
        store.begin_turn(child.id)

        with patch.object(session_tools, "get_current_session_id", return_value=caller.id):
            session_tools.session_reply(message="stop", session_id=child.id)
        store.end_turn(child.id)

        assert _deliveries(store, child.id)[0]["from_session"] == caller.id

    def test_with_a_queue_drain_the_message_is_held_in_the_session_queue(self, store, runner):
        runner.set_queued_message_sender(AsyncMock(return_value=True))
        child = _child(store)
        store.begin_turn(child.id)

        session_tools.session_reply(message="stop, the worktree is wrong", session_id=child.id)

        assert [e["text"] for e in store.get_session(child.id).queued_messages] == ["stop, the worktree is wrong"]
        assert _deliveries(store, child.id) == []

    def test_a_full_queue_comes_back_as_a_tool_error(self, store, runner, monkeypatch):
        monkeypatch.setattr(store, "MAX_QUEUED_MESSAGES", 1)
        runner.set_queued_message_sender(AsyncMock(return_value=True))
        child = _child(store)
        store.begin_turn(child.id)
        session_tools.session_reply(message="m0", session_id=child.id)

        result = session_tools.session_reply(message="one too many", session_id=child.id)

        assert result["session_id"] == child.id
        assert "queued messages" in result["error"]
        assert len(store.get_session(child.id).queued_messages) == store.MAX_QUEUED_MESSAGES

    def test_a_background_run_that_never_begins_a_turn_still_queues(self, store, runner, adapter):
        """Scheduled and background runs have status RUNNING and never call
        `begin_turn`, so `turn_in_flight` alone reads them as idle."""
        child = store.create_session(
            Session(id="runner", source=SessionSource.BACKGROUND.value, status=SessionStatus.RUNNING.value)
        )

        result = session_tools.session_reply(message="stop", session_id=child.id)

        assert adapter.handle_message.await_count == 0
        assert result["status"] == "queued"


class TestAnIdleSession:
    def test_the_reply_runs_a_turn_and_reports_it_delivered(self, store, runner, adapter):
        child = _child(store)

        result = session_tools.session_reply(message="carry on", session_id=child.id)

        assert result["status"] == "delivered"
        assert result["response"] == "ack"
        assert adapter.handle_message.await_count == 1


class TestASessionThatIsNotThere:
    def test_an_unknown_id_raises_before_any_turn_runs(self, store, runner, adapter):
        with pytest.raises(ValueError, match="ghost"):
            session_tools.session_reply(message="hello", session_id="ghost")

        assert adapter.handle_message.await_count == 0
