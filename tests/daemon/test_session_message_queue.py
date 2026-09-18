"""Mid-turn message queue: the store and the turn-end drain.

A message sent while a session is busy is held on the session rather than in the
sending client, so every client reads the same queue and a flush still runs after
the sender disconnects.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from tsugite_daemon.session_runner import SessionRunner
from tsugite_daemon.session_store import Session, SessionSource, SessionStore

from .conftest import _wait_until


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "store.json")


def _session(store: SessionStore, sid: str = "s1") -> str:
    store.create_session(Session(id=sid, source=SessionSource.INTERACTIVE.value, user_id="alice"))
    return sid


def _entry(queue_id: str, text: str) -> dict:
    return {
        "id": queue_id,
        "text": text,
        "user_id": "alice",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "reasoning_effort": None,
        "uploaded_files": [],
        "context_metadata": None,
    }


class TestTheQueue:
    def test_a_queued_message_is_held_on_the_session(self, store):
        sid = _session(store)
        entry = _entry("q-1", "hello")

        store.queue_message(sid, entry)

        assert store.get_session(sid).queued_messages == [entry]

    def test_queueing_reports_the_position(self, store):
        sid = _session(store)

        assert store.queue_message(sid, _entry("q-1", "first")) == 1
        assert store.queue_message(sid, _entry("q-2", "second")) == 2

    def test_taking_drains_in_arrival_order(self, store):
        sid = _session(store)
        store.queue_message(sid, _entry("q-1", "first"))
        store.queue_message(sid, _entry("q-2", "second"))

        assert store.take_queued_message(sid)["text"] == "first"
        assert store.take_queued_message(sid)["text"] == "second"
        assert store.take_queued_message(sid) is None

    def test_dropping_by_id_removes_that_entry(self, store):
        sid = _session(store)
        store.queue_message(sid, _entry("q-1", "first"))
        store.queue_message(sid, _entry("q-2", "second"))

        assert store.drop_queued_message(sid, "q-1") is True
        assert [e["id"] for e in store.get_session(sid).queued_messages] == ["q-2"]

    def test_dropping_an_unknown_id_is_a_no_op(self, store):
        sid = _session(store)
        store.queue_message(sid, _entry("q-1", "first"))

        assert store.drop_queued_message(sid, "q-ghost") is False
        assert [e["id"] for e in store.get_session(sid).queued_messages] == ["q-1"]

    def test_queueing_against_an_unknown_session_raises(self, store):
        with pytest.raises(ValueError, match="ghost"):
            store.queue_message("ghost", _entry("q-1", "hello"))

    def test_the_queue_is_persisted(self, store, tmp_path):
        sid = _session(store)
        store.queue_message(sid, _entry("q-1", "survive the restart"))

        reloaded = SessionStore(tmp_path / "store.json")

        assert [e["text"] for e in reloaded.get_session(sid).queued_messages] == ["survive the restart"]

    def test_session_detail_ships_the_wire_rows(self, store):
        """One key and one shape across the endpoints; the raw entry also carries
        the sender, the upload names and the client context."""
        sid = _session(store)
        store.queue_message(sid, _entry("q-1", "hello"))

        detail = store.session_detail(sid)

        assert detail.get("queued") == [{"id": "q-1", "text": "hello"}]
        assert "queued_messages" not in detail

    def test_compaction_moves_the_queue_to_the_successor(self, store):
        sid = _session(store)
        store.queue_message(sid, _entry("q-1", "still pending"))

        successor = store.compact_session(sid)

        assert [e["id"] for e in successor.queued_messages] == ["q-1"]
        assert store.get_session(sid).queued_messages == []


class TestTheTurnEndDrain:
    @pytest.fixture
    def runner(self, store):
        return SessionRunner(store=store, adapter=MagicMock(), event_bus=None)

    @pytest.mark.asyncio
    async def test_the_turn_end_hook_sends_a_queued_message(self, store, runner):
        """No HTTP chat task is involved: a turn begun anywhere flushes at its end."""
        sid = _session(store)
        sender = AsyncMock()
        runner.set_queued_message_sender(sender)
        store.begin_turn(sid)
        store.queue_message(sid, _entry("q-1", "next please"))

        store.end_turn(sid)

        assert await _wait_until(lambda: sender.await_count == 1)
        assert sender.await_args.args[0] == sid
        assert sender.await_args.args[1]["text"] == "next please"
        assert store.get_session(sid).queued_messages == []

    @pytest.mark.asyncio
    async def test_the_drain_keeps_arrival_order(self, store, runner):
        sid = _session(store)
        sender = AsyncMock()
        runner.set_queued_message_sender(sender)
        store.queue_message(sid, _entry("q-1", "first"))
        store.queue_message(sid, _entry("q-2", "second"))

        store.end_turn(sid)

        assert await _wait_until(lambda: sender.await_count == 2)
        assert [c.args[1]["text"] for c in sender.await_args_list] == ["first", "second"]
        assert store.get_session(sid).queued_messages == []

    @pytest.mark.asyncio
    async def test_one_queued_turn_runs_at_a_time(self, store, runner):
        sid = _session(store)
        release = asyncio.Event()
        started = []

        async def sender(session_id, entry):
            started.append(entry["id"])
            await release.wait()

        runner.set_queued_message_sender(sender)
        store.queue_message(sid, _entry("q-1", "first"))
        store.queue_message(sid, _entry("q-2", "second"))

        store.end_turn(sid)
        assert await _wait_until(lambda: started == ["q-1"])
        store.end_turn(sid)
        await asyncio.sleep(0.05)

        assert started == ["q-1"]
        release.set()
        assert await _wait_until(lambda: started == ["q-1", "q-2"])

    @pytest.mark.asyncio
    async def test_a_send_that_raises_does_not_strand_the_rest(self, store, runner):
        sender = AsyncMock(side_effect=[RuntimeError("boom"), None])
        sid = _session(store)
        runner.set_queued_message_sender(sender)
        store.queue_message(sid, _entry("q-1", "first"))
        store.queue_message(sid, _entry("q-2", "second"))

        store.end_turn(sid)

        assert await _wait_until(lambda: sender.await_count == 2)
        assert store.get_session(sid).queued_messages == []

    @pytest.mark.asyncio
    async def test_an_empty_queue_sends_nothing(self, store, runner):
        sid = _session(store)
        sender = AsyncMock()
        runner.set_queued_message_sender(sender)

        store.end_turn(sid)
        await asyncio.sleep(0.05)

        assert sender.await_count == 0
