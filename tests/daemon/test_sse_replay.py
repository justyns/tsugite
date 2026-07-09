"""SSE reconnect contract: every event carries a monotonic seq and lands in a
ring buffer; a reconnecting client replays what it missed, detects a daemon
restart via the boot epoch, and a slow subscriber is told to resync instead of
silently losing events. Without this, any connection gap (sleep/wake, blip,
restart) left the client silently stale until a manual reload."""

import json
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer, SSEBroadcaster, sse_stream
from tsugite_daemon.auth import TokenStore
from tsugite_daemon.config import AgentConfig, HTTPConfig
from tsugite_daemon.session_store import SessionStore


class TestBroadcasterReplay:
    def test_events_get_monotonic_seqs_and_buffer(self):
        bus = SSEBroadcaster()
        bus.emit("a", {"n": 1})
        bus.emit("b", {"n": 2})
        assert bus.seq == 2
        assert bus.replay_since(0) == [
            {"seq": 1, "type": "a", "data": {"n": 1}},
            {"seq": 2, "type": "b", "data": {"n": 2}},
        ]

    def test_replay_since_returns_tail_only(self):
        bus = SSEBroadcaster()
        for i in range(5):
            bus.emit("e", {"i": i})
        tail = bus.replay_since(3)
        assert [m["seq"] for m in tail] == [4, 5]

    def test_replay_up_to_date_client_gets_empty(self):
        bus = SSEBroadcaster()
        bus.emit("e", {})
        assert bus.replay_since(1) == []

    def test_gap_older_than_buffer_forces_resync(self):
        bus = SSEBroadcaster()
        for i in range(SSEBroadcaster.REPLAY_BUFFER_SIZE + 10):
            bus.emit("e", {"i": i})
        assert bus.replay_since(1) is None, "unreplayable gap must signal full resync, not a partial replay"

    def test_epochs_differ_across_instances(self):
        assert SSEBroadcaster().epoch != SSEBroadcaster().epoch


class TestLaggedSubscriber:
    @pytest.mark.asyncio
    async def test_overflow_flags_and_stream_emits_resync(self):
        bus = SSEBroadcaster()
        q = bus.subscribe()
        for i in range(70):  # queue maxsize is 64
            bus.emit("e", {"i": i})
        assert q.lagged, "overflow must flag the subscriber, not silently drop"

        frames = []
        stream = sse_stream(q, keepalive_interval=0.05)
        frames.append(await stream.__anext__())
        await stream.aclose()
        payload = json.loads(frames[0][6:])
        assert payload["type"] == "resync_required"


@pytest.fixture
def server_client(tmp_path):
    from tsugite.workspace import WorkspaceNotFoundError

    workspace = tmp_path / "ws"
    workspace.mkdir()
    store = SessionStore(tmp_path / "session_store.json")
    config = AgentConfig(workspace_dir=workspace, agent_file="default")
    with patch("tsugite.workspace.Workspace") as mock_ws:
        mock_ws.load.side_effect = WorkspaceNotFoundError("nope")
        with patch("tsugite.workspace.context.build_workspace_attachments", return_value=[]):
            adapter = HTTPAgentAdapter(agent_name="test-agent", agent_config=config, session_store=store)
    token_store = TokenStore(tmp_path / "tokens.json")
    _t, raw = token_store.create_admin_token(name="t")
    server = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8374),
        adapters={"test-agent": adapter},
        webhook_store=None,
        agent_configs={"test-agent": config},
        token_store=token_store,
    )
    return server, TestClient(server.app), {"Authorization": f"Bearer {raw}"}


def _request(query: str, auth: dict):
    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/events",
        "query_string": query.encode(),
        "headers": [(b"authorization", auth["Authorization"].encode())],
    }
    return Request(scope)


async def _collect_frames(server, query, auth, count):
    """Call the handler directly and pull N data frames from its stream - the
    endpoint's response is infinite, which TestClient can't consume."""
    resp = await server._events(_request(query, auth))
    frames = []
    agen = resp.body_iterator
    try:
        async for chunk in agen:
            if chunk.startswith("data: "):
                frames.append(json.loads(chunk[6:]))
                if len(frames) >= count:
                    break
    finally:
        await agen.aclose()
    return frames


class TestEventsEndpoint:
    @pytest.mark.asyncio
    async def test_fresh_connect_gets_hello(self, server_client):
        server, _client, auth = server_client
        frames = await _collect_frames(server, "", auth, 1)
        assert frames[0]["type"] == "hello"
        assert frames[0]["data"]["epoch"] == server.event_bus.epoch
        assert frames[0]["data"]["resync"] is False

    @pytest.mark.asyncio
    async def test_reconnect_replays_missed_events(self, server_client):
        server, _client, auth = server_client
        server.event_bus.emit("session_update", {"action": "titled", "id": "s1", "title": "one"})
        server.event_bus.emit("session_update", {"action": "titled", "id": "s1", "title": "two"})

        frames = await _collect_frames(server, f"epoch={server.event_bus.epoch}&last_seq=1", auth, 2)
        assert frames[0]["type"] == "hello" and frames[0]["data"]["resync"] is False
        assert frames[1]["seq"] == 2
        assert frames[1]["data"]["title"] == "two"

    @pytest.mark.asyncio
    async def test_epoch_mismatch_says_resync(self, server_client):
        server, _client, auth = server_client
        server.event_bus.emit("e", {})
        frames = await _collect_frames(server, "epoch=stale-epoch&last_seq=1", auth, 1)
        assert frames[0]["type"] == "hello"
        assert frames[0]["data"]["resync"] is True, "a restarted daemon (new epoch) must force a full resync"

    @pytest.mark.asyncio
    async def test_live_events_dedupe_against_replay(self, server_client):
        """An event emitted between the replay snapshot and queue subscribe can
        appear in both; the stream must not deliver the same seq twice."""
        server, _client, auth = server_client
        server.event_bus.emit("e", {"n": 1})

        resp = await server._events(_request(f"epoch={server.event_bus.epoch}&last_seq=0", auth))
        agen = resp.body_iterator
        frames = []
        try:
            async for chunk in agen:
                if chunk.startswith("data: "):
                    frames.append(json.loads(chunk[6:]))
                if len(frames) == 2:
                    server.event_bus.emit("e", {"n": 2})  # live event while connected
                if len(frames) >= 3:
                    break
        finally:
            await agen.aclose()
        seqs = [f.get("seq") for f in frames if f.get("seq")]
        assert seqs == sorted(set(seqs)), f"duplicate or out-of-order seqs delivered: {seqs}"
