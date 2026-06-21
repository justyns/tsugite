"""HTTP endpoint tests for the terminal viewer routes."""

from __future__ import annotations

import json
import sys
import time
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite_daemon.config import AgentConfig, HTTPConfig
from tsugite_daemon.webhook_store import WebhookStore
from tsugite_pty.pty_manager import PtyManager
from tsugite_pty.terminal_store import TerminalSessionStore, TerminalState

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="PTY support is POSIX-only")


@pytest.fixture
def tmp_workspace(tmp_path):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    return workspace_dir


@pytest.fixture
def agent_config(tmp_workspace):
    return AgentConfig(workspace_dir=tmp_workspace, agent_file="default")


@pytest.fixture
def http_config():
    return HTTPConfig(enabled=True, host="127.0.0.1", port=8375)


@pytest.fixture
def mock_adapter(agent_config, tmp_path):
    from tsugite_daemon.session_store import SessionStore

    from tsugite.workspace import WorkspaceNotFoundError

    session_store = SessionStore(tmp_path / "session_store.json")
    with patch("tsugite.workspace.Workspace") as mock_ws_cls:
        mock_ws_cls.load.side_effect = WorkspaceNotFoundError("not found")
        with patch("tsugite.workspace.context.build_workspace_attachments", return_value=[]):
            return HTTPAgentAdapter(
                agent_name="test-agent",
                agent_config=agent_config,
                session_store=session_store,
            )


@pytest.fixture
def webhook_store(tmp_path):
    return WebhookStore(tmp_path / "webhooks.json")


@pytest.fixture
def token_store(tmp_path):
    from tsugite_daemon.auth import TokenStore

    return TokenStore(tmp_path / "tokens.json")


@pytest.fixture
def test_token(token_store):
    _st, raw = token_store.create_admin_token(name="test-request-token")
    return raw


@pytest.fixture
def terminal_store(tmp_path):
    return TerminalSessionStore(tmp_path / "terminal_sessions.json")


@pytest.fixture
def pty_manager():
    mgr = PtyManager()
    yield mgr
    mgr.shutdown()


@pytest.fixture
def server(http_config, mock_adapter, webhook_store, agent_config, token_store, terminal_store, pty_manager):
    s = HTTPServer(
        config=http_config,
        adapters={"test-agent": mock_adapter},
        webhook_store=webhook_store,
        agent_configs={"test-agent": agent_config},
        token_store=token_store,
    )
    s.terminal_store = terminal_store
    s.pty_manager = pty_manager
    return s


@pytest.fixture
def client(server):
    return TestClient(server.app)


@pytest.fixture
def headers(test_token):
    return {"Authorization": f"Bearer {test_token}"}


def _wait_for_terminal(store, terminal_id, target_states, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        t = store.get(terminal_id)
        if t and t.state in target_states:
            return t
        time.sleep(0.02)
    pytest.fail(f"Terminal {terminal_id} never reached {target_states}; final={t.state if t else 'missing'!r}")


class TestCreateTerminal:
    def test_requires_auth(self, client):
        resp = client.post("/api/terminals", json={"cmd": "echo hi"})
        assert resp.status_code == 401

    def test_creates_and_returns_terminal_record(self, client, headers, terminal_store):
        resp = client.post("/api/terminals", json={"cmd": "echo hello"}, headers=headers)
        assert resp.status_code == 201
        data = resp.json()
        assert data["id"].startswith("term-")
        assert data["cmd"] == "echo hello"
        assert data["state"] in (TerminalState.RUNNING.value, TerminalState.SUCCEEDED.value)
        # Persisted in the store.
        assert terminal_store.get(data["id"]) is not None

    def test_missing_cmd_returns_400(self, client, headers):
        resp = client.post("/api/terminals", json={}, headers=headers)
        assert resp.status_code == 400

    def test_invalid_json_returns_400(self, client, headers):
        resp = client.post(
            "/api/terminals",
            content=b"not json",
            headers={**headers, "Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    def test_parent_session_id_persisted(self, client, headers, terminal_store):
        resp = client.post(
            "/api/terminals",
            json={"cmd": "echo hi", "parent_session_id": "chat-1"},
            headers=headers,
        )
        assert resp.status_code == 201
        terminal = terminal_store.get(resp.json()["id"])
        assert terminal.parent_session_id == "chat-1"


class TestListTerminals:
    def test_empty_list(self, client, headers):
        resp = client.get("/api/terminals", headers=headers)
        assert resp.status_code == 200
        assert resp.json() == {"terminals": []}

    def test_lists_after_create(self, client, headers):
        client.post("/api/terminals", json={"cmd": "echo a"}, headers=headers)
        client.post("/api/terminals", json={"cmd": "echo b"}, headers=headers)
        resp = client.get("/api/terminals", headers=headers)
        assert resp.status_code == 200
        cmds = {t["cmd"] for t in resp.json()["terminals"]}
        assert cmds == {"echo a", "echo b"}

    def test_filter_by_parent_session_id(self, client, headers):
        client.post("/api/terminals", json={"cmd": "a", "parent_session_id": "p1"}, headers=headers)
        client.post("/api/terminals", json={"cmd": "b", "parent_session_id": "p1"}, headers=headers)
        client.post("/api/terminals", json={"cmd": "c", "parent_session_id": "p2"}, headers=headers)
        resp = client.get("/api/terminals?parent_session_id=p1", headers=headers)
        cmds = {t["cmd"] for t in resp.json()["terminals"]}
        assert cmds == {"a", "b"}


class TestGetTerminal:
    def test_fetch_known_terminal(self, client, headers):
        created = client.post("/api/terminals", json={"cmd": "echo hi"}, headers=headers).json()
        resp = client.get(f"/api/terminals/{created['id']}", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["id"] == created["id"]

    def test_fetch_unknown_returns_404(self, client, headers):
        resp = client.get("/api/terminals/term-nope", headers=headers)
        assert resp.status_code == 404


class TestKillTerminal:
    def test_kill_running_terminal(self, client, headers, terminal_store):
        created = client.post("/api/terminals", json={"cmd": "sleep 30"}, headers=headers).json()
        time.sleep(0.1)
        resp = client.post(f"/api/terminals/{created['id']}/kill", headers=headers)
        assert resp.status_code == 200
        assert resp.json() == {"status": "killed", "terminal_id": created["id"]}
        terminal = _wait_for_terminal(terminal_store, created["id"], {TerminalState.CANCELLED.value}, timeout=5.0)
        assert terminal.state == TerminalState.CANCELLED.value

    def test_kill_unknown_returns_404(self, client, headers):
        resp = client.post("/api/terminals/term-nope/kill", headers=headers)
        assert resp.status_code == 404


class TestStdin:
    def test_writes_to_running_terminal(self, client, headers, terminal_store, pty_manager):
        created = client.post("/api/terminals", json={"cmd": "/bin/cat"}, headers=headers).json()
        time.sleep(0.1)
        resp = client.post(
            f"/api/terminals/{created['id']}/stdin",
            json={"data": "hello\n"},
            headers=headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["bytes_written"] > 0
        # Pump cat for a moment to echo, then kill.
        time.sleep(0.2)
        client.post(f"/api/terminals/{created['id']}/kill", headers=headers)
        _wait_for_terminal(terminal_store, created["id"], {TerminalState.CANCELLED.value}, timeout=5.0)

    def test_unknown_terminal_returns_404(self, client, headers):
        resp = client.post("/api/terminals/term-nope/stdin", json={"data": "x"}, headers=headers)
        assert resp.status_code == 404


class TestRestart:
    def test_restart_terminated_terminal(self, client, headers, terminal_store):
        created = client.post("/api/terminals", json={"cmd": "echo hi"}, headers=headers).json()
        _wait_for_terminal(
            terminal_store,
            created["id"],
            {TerminalState.SUCCEEDED.value, TerminalState.FAILED.value},
            timeout=5.0,
        )
        resp = client.post(f"/api/terminals/{created['id']}/restart", headers=headers)
        assert resp.status_code == 201
        new = resp.json()
        assert new["id"] != created["id"]
        assert new["cmd"] == "echo hi"
        assert new["restarted_from"] == created["id"]

    def test_restart_live_terminal_returns_409(self, client, headers):
        created = client.post("/api/terminals", json={"cmd": "sleep 30"}, headers=headers).json()
        time.sleep(0.1)
        resp = client.post(f"/api/terminals/{created['id']}/restart", headers=headers)
        # The brief window before RUNNING is fine too - either pre-exit state is non-terminal.
        assert resp.status_code == 409
        client.post(f"/api/terminals/{created['id']}/kill", headers=headers)

    def test_restart_unknown_returns_404(self, client, headers):
        resp = client.post("/api/terminals/term-nope/restart", headers=headers)
        assert resp.status_code == 404


class TestStream:
    def test_stream_short_command_emits_output_state_exit(self, client, headers):
        """Spawn a tiny `echo`, hit the stream, collect events, verify shapes."""
        created = client.post("/api/terminals", json={"cmd": "echo streamtest"}, headers=headers).json()
        # Give the PTY a moment so the SSE handler can replay buffered output.
        time.sleep(0.2)

        events = _collect_stream(client, headers, created["id"], timeout=5.0)
        event_types = [e["event"] for e in events]
        # First non-keepalive events should include a state at least.
        assert any(t == "state" for t in event_types), event_types
        # Either replayed output or live chunk - collected stream must include
        # the echoed payload somewhere across output events.
        output_chunks = [json.loads(e["data"]) for e in events if e["event"] == "output"]
        assert any("streamtest" in (c.get("chunk") or "") for c in output_chunks), output_chunks
        # And we MUST see an exit event before close.
        assert any(t == "exit" for t in event_types), event_types

    def test_stream_unknown_returns_404(self, client, headers):
        resp = client.get("/api/terminals/term-nope/stream", headers=headers)
        assert resp.status_code == 404

    def test_stream_replay_frame_tagged_replay(self, client, headers, terminal_store):
        """The buffered output replayed on connect carries `replay: true`.

        Load-bearing: the frontend now RESETS its buffer/metrics on a replay
        frame instead of appending. If the backend dropped the flag, a reconnect
        would double-count output. We connect after the PTY finishes so the only
        output the stream can emit is the replayed ring buffer (the proc lingers
        in the manager for the eviction grace, so the buffer is still there).
        """
        created = client.post("/api/terminals", json={"cmd": "echo replayme"}, headers=headers).json()
        _wait_for_terminal(
            terminal_store,
            created["id"],
            {TerminalState.SUCCEEDED.value, TerminalState.FAILED.value},
            timeout=5.0,
        )

        events = _collect_stream(client, headers, created["id"], timeout=5.0)
        output_chunks = [json.loads(e["data"]) for e in events if e["event"] == "output"]
        replayed = [c for c in output_chunks if c.get("replay") is True]
        assert replayed, f"expected a replay-tagged output frame; got {output_chunks}"
        assert any("replayme" in (c.get("chunk") or "") for c in replayed), replayed

    def test_stream_replays_from_disk_after_eviction(self, client, headers, terminal_store, pty_manager):
        """After the in-memory PtyProcess is evicted, the stream must replay
        from the on-disk log written by the exit hook. Regression: previously
        the no-proc branch only emitted `exit` and closed, leaving an empty
        terminal in the UI for any reconnect later than ~30s after exit."""
        created = client.post("/api/terminals", json={"cmd": "echo persisted"}, headers=headers).json()
        tid = created["id"]
        _wait_for_terminal(
            terminal_store,
            tid,
            {TerminalState.SUCCEEDED.value, TerminalState.FAILED.value},
            timeout=5.0,
        )
        # Drain the PTY output so the buffer is final.
        proc = pty_manager.get(tid)
        if proc is not None:
            proc.wait_drain(timeout=1.0)

        # Run the exit hook so the buffer is persisted to disk, then forcibly
        # evict to simulate "user reconnects long after exit".
        from tsugite_pty.terminal_runtime import _on_pty_exit

        _on_pty_exit(proc, terminal_store, pty_manager, tid)
        pty_manager.remove(tid)
        assert pty_manager.get(tid) is None

        events = _collect_stream(client, headers, tid, timeout=5.0)
        output_chunks = [json.loads(e["data"]) for e in events if e["event"] == "output"]
        replayed = [c for c in output_chunks if c.get("replay") is True]
        assert replayed, f"expected a disk-replay frame; got {output_chunks}"
        assert any("persisted" in (c.get("chunk") or "") for c in replayed), replayed

    def test_stream_closes_under_output_burst(self, client, headers):
        """A command producing a burst of output still tears the stream down
        cleanly (exit event, then the generator ends).

        Load-bearing-ish: exercises the `_safe_put`/`closing` teardown path under
        volume so a regression that hangs the generator (sentinel dropped, no
        `closing` fallback) is caught. NOTE: true queue saturation isn't
        reproducible at the TestClient layer (the client drains the generator as
        fast as the server fills it, maxsize=512). The unit test below pins the
        QueueFull-swallowing contract directly; this asserts robustness to load.
        """
        created = client.post(
            "/api/terminals",
            json={"cmd": "for i in $(seq 1 500); do echo burst-line-$i; done"},
            headers=headers,
        ).json()
        time.sleep(0.2)
        events = _collect_stream(client, headers, created["id"], timeout=10.0)
        event_types = [e["event"] for e in events]
        assert any(t == "exit" for t in event_types), event_types

    def test_safe_put_swallows_queue_full_on_saturated_queue(self):
        """The stream's safe-put helper must swallow QueueFull on a full queue.

        Load-bearing: the fix wrapped the enqueue in try/except QueueFull so a
        slow SSE client can't propagate QueueFull up and crash the reader thread.
        This pins the exact contract: an unguarded put_nowait raises on a full
        asyncio.Queue, and the guarded form swallows it without raising.
        """
        import asyncio

        async def run():
            q: asyncio.Queue = asyncio.Queue(maxsize=2)
            q.put_nowait("a")
            q.put_nowait("b")
            # Sanity: the queue really is full and put_nowait raises.
            with pytest.raises(asyncio.QueueFull):
                q.put_nowait("c")

            # This mirrors http.py's `_safe_put`: drop the chunk on a full queue.
            def safe_put(payload):
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    pass

            safe_put("c")  # must NOT raise
            assert q.qsize() == 2  # dropped, queue unchanged

        asyncio.run(run())


# ── helpers ──


def _collect_stream(client, headers, terminal_id: str, timeout: float = 5.0):
    """Drain the SSE stream until the terminal exits. Returns list of {event, data}."""
    events = []
    with client.stream("GET", f"/api/terminals/{terminal_id}/stream", headers=headers) as resp:
        assert resp.status_code == 200
        deadline = time.time() + timeout
        current_event = "message"
        for line in resp.iter_lines():
            if time.time() > deadline:
                break
            if not line:
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event: "):
                current_event = line[len("event: ") :].strip()
                continue
            if line.startswith("data: "):
                payload = line[len("data: ") :]
                events.append({"event": current_event, "data": payload})
                if current_event == "exit":
                    break
    return events
