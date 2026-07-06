"""The server, not the UI, is the source of truth for whether a session is
busy: the sessions payload and /status must expose the authoritative
turn-in-flight flag so clients render busy state instead of inferring it from
cached progress labels (which go stale on reconnect/PWA resume, leaving a
silently-running turn that 409s new sends with no visible explanation)."""

from unittest.mock import patch

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite_daemon.auth import TokenStore
from tsugite_daemon.config import AgentConfig, HTTPConfig
from tsugite_daemon.session_store import Session, SessionSource, SessionStore


@pytest.fixture
def adapter(tmp_path):
    from tsugite.workspace import WorkspaceNotFoundError

    workspace = tmp_path / "ws"
    workspace.mkdir()
    store = SessionStore(tmp_path / "session_store.json")
    config = AgentConfig(workspace_dir=workspace, agent_file="default")
    with patch("tsugite.workspace.Workspace") as mock_ws:
        mock_ws.load.side_effect = WorkspaceNotFoundError("nope")
        with patch("tsugite.workspace.context.build_workspace_attachments", return_value=[]):
            return HTTPAgentAdapter(agent_name="test-agent", agent_config=config, session_store=store)


@pytest.fixture
def client_and_token(adapter, tmp_path):
    token_store = TokenStore(tmp_path / "tokens.json")
    _t, raw = token_store.create_admin_token(name="t")
    server = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8374),
        adapters={"test-agent": adapter},
        webhook_store=None,
        agent_configs={"test-agent": adapter.agent_config},
        token_store=token_store,
    )
    client = TestClient(server.app)
    client.app_server = server  # for tests that poke _active_chats directly
    return client, raw


def _mk_session(adapter, sid="s-busy"):
    adapter.session_store.create_session(
        Session(id=sid, agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id="u1")
    )
    return sid


def test_sessions_payload_exposes_busy_flag(adapter, client_and_token):
    client, token = client_and_token
    sid = _mk_session(adapter)
    adapter.session_store.begin_turn(sid)

    resp = client.get("/api/agents/test-agent/sessions", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    rows = {r["id"]: r for r in resp.json()["sessions"]}
    assert rows[sid]["busy"] is True, "sessions payload must carry the authoritative busy flag"

    adapter.session_store.end_turn(sid)
    resp = client.get("/api/agents/test-agent/sessions", headers={"Authorization": f"Bearer {token}"})
    rows = {r["id"]: r for r in resp.json()["sessions"]}
    assert rows[sid]["busy"] is False


def test_status_busy_reflects_turn_in_flight_without_http_chat(adapter, client_and_token):
    """A turn driven by any adapter (schedule reply, Discord, job notify) must
    surface busy=true in /status even though no HTTP chat task exists."""
    client, token = client_and_token
    sid = _mk_session(adapter, "s-status")
    adapter.session_store.begin_turn(sid)

    resp = client.get(
        f"/api/agents/test-agent/status?user_id=u1&session_id={sid}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["busy"] is True

    adapter.session_store.end_turn(sid)
    resp = client.get(
        f"/api/agents/test-agent/status?user_id=u1&session_id={sid}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.json()["busy"] is False


def test_sessions_payload_busy_from_live_http_task(adapter, client_and_token):
    """The pre-begin_turn window: an HTTP chat task exists but the durable
    marker isn't set yet. The unified predicate must still report busy so the
    sessions payload can never say idle while /chat would 409."""
    from types import SimpleNamespace

    client, token = client_and_token
    sid = _mk_session(adapter, "s-task")
    http_server = client.app_server
    http_server._active_chats[("test-agent", "u1", sid)] = SimpleNamespace(
        task=SimpleNamespace(done=lambda: False), backend=None
    )
    try:
        resp = client.get("/api/agents/test-agent/sessions", headers={"Authorization": f"Bearer {token}"})
        rows = {r["id"]: r for r in resp.json()["sessions"]}
        assert rows[sid]["busy"] is True
    finally:
        http_server._active_chats.pop(("test-agent", "u1", sid), None)
