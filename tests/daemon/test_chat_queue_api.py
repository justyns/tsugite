"""The HTTP half of the mid-turn message queue.

A send to a busy session is accepted (202) and parked on the session. Every client
reads the same queue off the sessions payload and /status, any client can drop an
entry, and the turn-end hook runs the parked message as an ordinary user turn once
the turn it waited on finishes.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio
from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite_daemon.auth import TokenStore
from tsugite_daemon.config import HTTPConfig, RuntimeDefaults
from tsugite_daemon.session_runner import SessionRunner
from tsugite_daemon.session_store import Session, SessionSource, SessionStore

from .conftest import _wait_until


@pytest.fixture
def adapter(tmp_path):
    from tsugite.workspace import WorkspaceNotFoundError

    workspace = tmp_path / "ws"
    workspace.mkdir()
    store = SessionStore(tmp_path / "session_store.json")
    config = RuntimeDefaults(workspace_dir=workspace, agent_file="default")
    with patch("tsugite.workspace.Workspace") as mock_ws:
        mock_ws.load.side_effect = WorkspaceNotFoundError("nope")
        a = HTTPAgentAdapter(runtime=config, session_store=store)
    a.handle_message = AsyncMock(return_value="ok")
    return a


@pytest.fixture
def token_store(tmp_path):
    return TokenStore(tmp_path / "tokens.json")


@pytest.fixture
def token(token_store):
    _t, raw = token_store.create_admin_token(name="t")
    return raw


@pytest.fixture
def server(adapter, token_store):
    server = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8374),
        adapter=adapter,
        webhook_store=None,
        token_store=token_store,
    )
    runner = SessionRunner(store=adapter.session_store, adapter=adapter, event_bus=server.event_bus)
    server.session_runner = runner
    runner.set_queued_message_sender(server.run_queued_message, server.session_busy)
    return server


@pytest_asyncio.fixture
async def client(server, token):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app),
        base_url="http://daemon",
        headers={"Authorization": f"Bearer {token}"},
    ) as c:
        yield c


@pytest.fixture
def broadcasts(server):
    seen: list[tuple[str, dict]] = []
    original = server.event_bus.emit

    def record(name, payload):
        seen.append((name, payload))
        original(name, payload)

    server.event_bus.emit = record
    return seen


def _session(adapter, sid="s-queue"):
    adapter.session_store.create_session(Session(id=sid, source=SessionSource.WEB.value, user_id="u1"))
    return sid


def _busy_session(adapter, sid="s-queue"):
    _session(adapter, sid)
    adapter.session_store.begin_turn(sid)
    return sid


async def _send(client, sid, message):
    return await client.post("/api/chat", json={"message": message, "session_id": sid, "user_id": "u1"})


def _sent_messages(adapter):
    return [c.kwargs["message"] for c in adapter.handle_message.await_args_list]


def _session_events(broadcasts, event_type):
    return [p for n, p in broadcasts if n == "session_event" and p.get("event_type") == event_type]


def _seed_queue(adapter, sid, queue_id, text):
    entry = SessionStore.queue_entry(text, user_id="u1")
    adapter.session_store.queue_message(sid, {**entry, "id": queue_id})


def _hold(adapter, text: str) -> asyncio.Event:
    """Hold handle_message inside the turn for `text`. The returned event fires
    once that turn has entered the handler."""
    running = asyncio.Event()

    async def handle(**kwargs):
        if kwargs["message"] == text:
            running.set()
            await asyncio.Event().wait()
        return "ok"

    adapter.handle_message.side_effect = handle
    return running


@pytest.mark.asyncio
class TestSendingToABusySession:
    async def test_the_send_is_accepted_and_queued(self, adapter, client):
        sid = _busy_session(adapter)

        resp = await _send(client, sid, "and also check the logs")

        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "queued"
        assert body["position"] == 1
        assert body["queue_id"].startswith("q-")

    async def test_another_client_sees_the_queue_on_the_sessions_payload(self, adapter, client):
        sid = _busy_session(adapter)
        await _send(client, sid, "and also check the logs")

        rows = (await client.get("/api/chat/sessions")).json()["sessions"]

        row = next(r for r in rows if r["id"] == sid)
        assert [q["text"] for q in row["queued"]] == ["and also check the logs"]

    async def test_the_queue_change_is_broadcast(self, adapter, client, broadcasts):
        sid = _busy_session(adapter)
        await _send(client, sid, "and also check the logs")

        updates = [p for n, p in broadcasts if n == "session_update" and p.get("action") == "queued"]
        assert updates and updates[-1]["id"] == sid
        assert [q["text"] for q in updates[-1]["queued"]] == ["and also check the logs"]


@pytest.mark.asyncio
class TestTheTurnEnding:
    async def test_the_queued_message_runs_without_the_queuing_client(self, adapter, client):
        """No HTTP chat task exists for a turn begun elsewhere, and the POST that
        queued the message has already returned."""
        sid = _busy_session(adapter)
        await _send(client, sid, "and also check the logs")
        assert adapter.handle_message.await_count == 0

        adapter.session_store.end_turn(sid)

        assert await _wait_until(lambda: adapter.handle_message.await_count == 1)
        assert _sent_messages(adapter) == ["and also check the logs"]
        assert adapter.session_store.get_session(sid).queued_messages == []

    async def test_a_failed_flush_does_not_broadcast_the_entry_as_gone(self, adapter, client, broadcasts):
        """A failing `_build_send_metadata` must not tell every client the message
        left the queue before it actually did."""
        sid = _busy_session(adapter)
        await _send(client, sid, "and also check the logs")

        with patch(
            "tsugite_daemon.adapters.http.agents._build_send_metadata",
            side_effect=RuntimeError("boom"),
        ):
            adapter.session_store.end_turn(sid)
            await asyncio.sleep(0.05)

        assert adapter.handle_message.await_count == 0
        assert [e["text"] for e in adapter.session_store.get_session(sid).queued_messages] == [
            "and also check the logs"
        ]
        updates = [p for n, p in broadcasts if n == "session_update" and p.get("action") == "queued"]
        assert [q["text"] for q in updates[-1]["queued"]] == ["and also check the logs"]

        adapter.session_store.end_turn(sid)

        assert await _wait_until(lambda: adapter.handle_message.await_count == 1)
        assert adapter.session_store.get_session(sid).queued_messages == []

    async def test_two_queued_messages_keep_arrival_order(self, adapter, client):
        sid = _busy_session(adapter)

        first = await _send(client, sid, "first")
        second = await _send(client, sid, "second")

        assert [first.json()["position"], second.json()["position"]] == [1, 2]
        adapter.session_store.end_turn(sid)

        assert await _wait_until(lambda: adapter.handle_message.await_count == 2)
        assert _sent_messages(adapter) == ["first", "second"]

    async def test_the_flush_runs_a_user_turn_rather_than_a_delivery(self, adapter, client):
        sid = _busy_session(adapter)
        await _send(client, sid, "and also check the logs")

        adapter.session_store.end_turn(sid)

        assert await _wait_until(lambda: adapter.handle_message.await_count == 1)
        context = adapter.handle_message.await_args.kwargs["channel_context"]
        assert context.source == "http"
        assert context.metadata["conv_id_override"] == sid

    async def test_the_flush_resolves_the_uploads_and_context_it_parked(self, adapter, client):
        """`_build_send_metadata` runs at flush time, so an Attachment never has
        to survive the queue's JSON round-trip."""
        uploads = adapter.runtime.workspace_dir / "uploads"
        uploads.mkdir()
        (uploads / "notes.txt").write_text("the disk is full")
        sid = _busy_session(adapter)
        # /api/chat validates reasoning_effort against the daemon default model.
        adapter.runtime.model = "openai:o3-mini"

        resp = await client.post(
            "/api/chat",
            json={
                "message": "read this",
                "session_id": sid,
                "user_id": "u1",
                "uploaded_files": [{"name": "notes.txt"}],
                "context_metadata": [{"type": "file", "path": "notes.txt"}],
                "reasoning_effort": "high",
            },
        )
        assert resp.status_code == 202

        adapter.session_store.end_turn(sid)

        assert await _wait_until(lambda: adapter.handle_message.await_count == 1)
        metadata = adapter.handle_message.await_args.kwargs["channel_context"].metadata
        assert [a.name for a in metadata["uploaded_attachments"]] == ["notes.txt"]
        assert all(a.user_upload for a in metadata["uploaded_attachments"])
        assert metadata["context_metadata"] == [{"type": "file", "path": "notes.txt"}]
        assert metadata["reasoning_effort_override"] == "high"


@pytest.mark.asyncio
class TestDequeueing:
    async def test_dropping_an_entry_removes_it_for_every_client(self, adapter, client, broadcasts):
        sid = _busy_session(adapter)
        queue_id = (await _send(client, sid, "drop me")).json()["queue_id"]
        await _send(client, sid, "keep me")

        resp = await client.delete(f"/api/chat/sessions/{sid}/queue/{queue_id}")

        assert resp.status_code == 200
        assert [q["text"] for q in resp.json()["queued"]] == ["keep me"]
        rows = (await client.get("/api/chat/sessions")).json()["sessions"]
        row = next(r for r in rows if r["id"] == sid)
        assert [q["text"] for q in row["queued"]] == ["keep me"]
        updates = [p for n, p in broadcasts if n == "session_update" and p.get("action") == "queued"]
        assert [q["text"] for q in updates[-1]["queued"]] == ["keep me"]

    async def test_dropping_an_unknown_entry_is_a_404(self, adapter, client):
        sid = _busy_session(adapter)
        await _send(client, sid, "keep me")

        resp = await client.delete(f"/api/chat/sessions/{sid}/queue/q-ghost")

        assert resp.status_code == 404
        assert [e["text"] for e in adapter.session_store.get_session(sid).queued_messages] == ["keep me"]

    async def test_a_dropped_entry_never_runs(self, adapter, client):
        sid = _busy_session(adapter)
        queue_id = (await _send(client, sid, "drop me")).json()["queue_id"]
        await client.delete(f"/api/chat/sessions/{sid}/queue/{queue_id}")

        adapter.session_store.end_turn(sid)
        await asyncio.sleep(0.05)

        assert adapter.handle_message.await_count == 0


@pytest.mark.asyncio
class TestCompaction:
    async def test_the_queue_moves_to_the_successor(self, adapter, client):
        sid = _busy_session(adapter)
        await _send(client, sid, "still pending")

        successor = adapter.session_store.compact_session(sid)

        rows = {r["id"]: r for r in (await client.get("/api/chat/sessions?include_superseded=1")).json()["sessions"]}
        assert [q["text"] for q in rows[successor.id]["queued"]] == ["still pending"]
        assert rows[sid]["queued"] == []


@pytest.mark.asyncio
class TestSendingWithEntriesAlreadyParked:
    async def test_an_idle_session_with_parked_entries_queues_then_drains(self, adapter, client):
        """A restart clears turn_in_flight without ending a turn, so nothing is left
        to drain a queue parked across it. Parking while idle drains it instead."""
        sid = _session(adapter, "s-idle-parked")
        _seed_queue(adapter, sid, "q-1", "first")

        resp = await _send(client, sid, "second")

        assert resp.status_code == 202
        assert resp.json()["position"] == 2
        assert await _wait_until(lambda: adapter.handle_message.await_count == 2)
        assert _sent_messages(adapter) == ["first", "second"]
        assert adapter.session_store.get_session(sid).queued_messages == []

    async def test_an_entry_parked_without_source_or_metadata_still_drains(self, adapter, client):
        sid = _session(adapter, "s-old-entry")
        entry = SessionStore.queue_entry("parked earlier", user_id="u1")
        del entry["source"], entry["metadata"]
        adapter.session_store.queue_message(sid, entry)

        resp = await _send(client, sid, "second")

        assert resp.status_code == 202
        assert await _wait_until(lambda: adapter.handle_message.await_count == 2)
        assert _sent_messages(adapter) == ["parked earlier", "second"]
        assert adapter.session_store.get_session(sid).queued_messages == []

    async def test_the_cap_rejects_a_send_with_a_clear_error(self, adapter, client):
        sid = _busy_session(adapter)
        for i in range(adapter.session_store.MAX_QUEUED_MESSAGES):
            assert (await _send(client, sid, f"msg {i}")).status_code == 202

        resp = await _send(client, sid, "one too many")

        assert resp.status_code == 429
        assert resp.json()["code"] == "queue_full"
        assert len(adapter.session_store.get_session(sid).queued_messages) == adapter.session_store.MAX_QUEUED_MESSAGES


@pytest.mark.asyncio
class TestSendingToAnIdleSession:
    async def test_the_send_runs_instead_of_queueing(self, adapter, client):
        sid = _session(adapter, "s-idle")

        resp = await asyncio.wait_for(_send(client, sid, "go"), timeout=5)

        assert resp.status_code == 200
        assert await _wait_until(lambda: adapter.handle_message.await_count == 1)
        assert adapter.session_store.get_session(sid).queued_messages == []

    async def test_a_send_without_a_session_id_targets_the_primary_session(self, adapter, client):
        resp = await asyncio.wait_for(client.post("/api/chat", json={"message": "go", "user_id": "u1"}), timeout=5)

        assert resp.status_code == 200
        assert await _wait_until(lambda: adapter.handle_message.await_count == 1)
        primary = adapter.session_store.find_default_session("u1")
        metadata = adapter.handle_message.await_args.kwargs["channel_context"].metadata
        assert metadata.get("conv_id_override") == primary.id


@pytest.mark.asyncio
class TestAQueuedTurnRunsLikeADirectSend:
    async def test_it_emits_a_final_result_when_the_run_reports_no_final_answer(self, adapter, client, broadcasts):
        sid = _busy_session(adapter)
        await _send(client, sid, "go")

        adapter.session_store.end_turn(sid)

        assert await _wait_until(lambda: _session_events(broadcasts, "final_result"))
        assert _session_events(broadcasts, "final_result")[-1]["result"] == "ok"

    async def test_it_records_the_turn_in_the_session_log(self, adapter, client):
        """The tab that queued the message may be gone by the flush, so a client
        that loads the session afterwards has only the log to read the turn from."""
        sid = _busy_session(adapter)
        await _send(client, sid, "go")

        adapter.session_store.end_turn(sid)

        assert await _wait_until(lambda: adapter.handle_message.await_count == 1)
        assert await _wait_until(
            lambda: [e for e in adapter.session_store.read_events(sid) if e["type"] == "final_result"]
        )
        recorded = [e for e in adapter.session_store.read_events(sid) if e["type"] == "final_result"]
        assert recorded[-1]["result"] == "ok"

    async def test_it_refreshes_the_context_meter(self, adapter, client, broadcasts):
        sid = _busy_session(adapter)
        await _send(client, sid, "go")

        adapter.session_store.end_turn(sid)

        assert await _wait_until(lambda: _session_events(broadcasts, "session_info"))
        assert _session_events(broadcasts, "session_info")[-1]["session_id"] == sid

    async def test_stopping_it_emits_a_cancelled_frame(self, adapter, client, broadcasts):
        sid = _busy_session(adapter)
        running = _hold(adapter, "first")
        await _send(client, sid, "first")
        adapter.session_store.end_turn(sid)
        await asyncio.wait_for(running.wait(), timeout=2)

        assert (await client.post("/api/chat/cancel", json={"user_id": "u1", "session_id": sid})).status_code == 200

        assert await _wait_until(lambda: _session_events(broadcasts, "cancelled"))

    async def test_stopping_one_queued_turn_still_runs_the_rest(self, adapter, client):
        sid = _busy_session(adapter)
        running = _hold(adapter, "first")
        await _send(client, sid, "first")
        await _send(client, sid, "second")
        adapter.session_store.end_turn(sid)
        await asyncio.wait_for(running.wait(), timeout=2)

        assert (await client.post("/api/chat/cancel", json={"user_id": "u1", "session_id": sid})).status_code == 200

        assert await _wait_until(lambda: _sent_messages(adapter) == ["first", "second"])

    async def test_the_prompt_inspector_reads_its_live_prompt(self, adapter, client):
        from tsugite.events import PromptSnapshotEvent

        sid = _busy_session(adapter)
        messages = [{"role": "user", "content": "hi"}]
        running = asyncio.Event()

        async def handle(**kwargs):
            kwargs["custom_logger"].ui_handler.handle_event(PromptSnapshotEvent(messages=messages))
            running.set()
            await asyncio.Event().wait()

        adapter.handle_message.side_effect = handle
        await _send(client, sid, "go")
        adapter.session_store.end_turn(sid)
        await asyncio.wait_for(running.wait(), timeout=2)

        snapshot = (await client.get(f"/api/chat/prompt-snapshot?user_id=u1&session_id={sid}")).json()

        assert snapshot["prompt_snapshot"] == {"messages": messages, "token_breakdown": {}}

    async def test_it_stays_queued_while_the_daemon_is_restarting(self, adapter, server, client):
        sid = _busy_session(adapter)
        await _send(client, sid, "after the restart")
        server.gateway = SimpleNamespace(restart_requested=True, config_path=None)

        adapter.session_store.end_turn(sid)
        await asyncio.sleep(0.05)

        assert adapter.handle_message.await_count == 0
        assert [e["text"] for e in adapter.session_store.get_session(sid).queued_messages] == ["after the restart"]


@pytest.mark.asyncio
class TestSendingToAFinishedSession:
    def _finished(self, adapter, sid, *, source, status, resumable):
        adapter.session_store.create_session(
            Session(id=sid, source=source, user_id="u1", status=status, resumable=resumable)
        )
        return sid

    async def test_a_completed_resumable_background_session_takes_the_send(self, adapter, client):
        sid = self._finished(adapter, "s-bg", source=SessionSource.BACKGROUND.value, status="completed", resumable=True)

        resp = await _send(client, sid, "one more thing")

        assert resp.status_code == 200
        await _wait_until(lambda: "one more thing" in _sent_messages(adapter))

    async def test_a_completed_session_that_is_not_resumable_is_rejected(self, adapter, client):
        sid = self._finished(adapter, "s-web", source=SessionSource.WEB.value, status="completed", resumable=False)

        resp = await _send(client, sid, "one more thing")

        assert resp.status_code == 409
        assert resp.json()["code"] == "session_finished"

    async def test_a_failed_resumable_session_is_rejected(self, adapter, client):
        sid = self._finished(adapter, "s-bad", source=SessionSource.BACKGROUND.value, status="failed", resumable=True)

        resp = await _send(client, sid, "one more thing")

        assert resp.status_code == 409
        assert resp.json()["code"] == "session_finished"


@pytest.mark.asyncio
async def test_a_session_reply_sent_mid_turn_runs_as_the_next_turn_with_its_text(server, adapter):
    sid = _busy_session(adapter)

    result = await server.session_runner.reply_or_defer(sid, "Stuck jobs need a look.")

    assert result == {"status": "queued", "response": None}
    assert _sent_messages(adapter) == []

    adapter.session_store.end_turn(sid)

    assert await _wait_until(lambda: _sent_messages(adapter) == ["Stuck jobs need a look."])
    assert adapter.session_store.get_session(sid).queued_messages == []


@pytest.mark.asyncio
async def test_a_session_reply_sent_mid_turn_keeps_the_session_it_came_from(server, adapter):
    sid = _busy_session(adapter)

    await server.session_runner.reply_or_defer(sid, "Stuck jobs need a look.", metadata={"from_session": "lead"})
    adapter.session_store.end_turn(sid)

    assert await _wait_until(lambda: adapter.handle_message.await_count == 1)
    channel = adapter.handle_message.await_args.kwargs["channel_context"].to_dict()
    assert channel["source"] == "session"
    assert channel["from_session"] == "lead"


@pytest.mark.asyncio
async def test_a_session_reply_to_a_background_session_runs_as_the_session_itself(server, adapter):
    adapter.session_store.create_session(Session(id="bg", source=SessionSource.BACKGROUND.value))
    adapter.session_store.begin_turn("bg")

    await server.session_runner.reply_or_defer("bg", "steer")
    adapter.session_store.end_turn("bg")

    assert await _wait_until(lambda: adapter.handle_message.await_count == 1)
    kwargs = adapter.handle_message.await_args.kwargs
    assert kwargs["user_id"] == "session:bg"
    assert kwargs["channel_context"].reply_to == "session:bg"
    assert kwargs["channel_context"].metadata["session_id"] == "bg"


@pytest.mark.asyncio
async def test_a_reply_queued_to_a_run_that_never_ends_a_turn_runs_when_the_run_finishes(server, adapter):
    release = asyncio.Event()
    sent = []

    async def handle(**kwargs):
        sent.append(kwargs["message"])
        if len(sent) == 1:
            await release.wait()
        return "done"

    adapter.handle_message.side_effect = handle
    server.session_runner.start_session(Session(id="bg", prompt="work", source=SessionSource.BACKGROUND.value))
    assert await _wait_until(lambda: sent == ["work"])

    await server.session_runner.reply_or_defer("bg", "steer")
    assert sent == ["work"]
    release.set()

    assert await _wait_until(lambda: sent == ["work", "steer"])
    assert adapter.session_store.get_session("bg").queued_messages == []


@pytest.mark.asyncio
async def test_a_session_reply_waits_while_a_chat_task_is_starting(server, adapter):
    sid = _session(adapter)
    starting = asyncio.create_task(asyncio.Event().wait())
    server._active_chats[("u1", sid)] = SimpleNamespace(task=starting)

    await server.session_runner.reply_or_defer(sid, "steer")
    await asyncio.sleep(0.05)
    starting.cancel()

    assert adapter.handle_message.await_count == 0
    assert [e["text"] for e in adapter.session_store.get_session(sid).queued_messages] == ["steer"]
