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
    runner.set_queued_message_sender(server.run_queued_message)
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
