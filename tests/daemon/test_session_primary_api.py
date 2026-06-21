"""HTTP API tests for primary-session endpoints (Tier 2 of session routing refactor)."""

from unittest.mock import patch

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite_daemon.config import AgentConfig, HTTPConfig
from tsugite_daemon.session_runner import SessionRunner
from tsugite_daemon.session_store import Session, SessionSource, SessionStore
from tsugite_daemon.webhook_store import WebhookStore


@pytest.fixture
def tmp_workspace(tmp_path):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    return workspace_dir


@pytest.fixture
def session_store(tmp_path):
    return SessionStore(tmp_path / "session_store.json")


@pytest.fixture
def session_runner(session_store):
    return SessionRunner(store=session_store, adapters={})


@pytest.fixture
def server(tmp_workspace, session_store, session_runner, tmp_path):
    agent_config = AgentConfig(workspace_dir=tmp_workspace, agent_file="default")
    http_config = HTTPConfig(enabled=True, host="127.0.0.1", port=8377)
    webhook_store = WebhookStore(tmp_path / "webhooks.json")

    from tsugite_daemon.auth import TokenStore

    token_store = TokenStore(tmp_path / "tokens.json")

    with patch("tsugite.workspace.Workspace") as mock_ws_cls:
        from tsugite.workspace import WorkspaceNotFoundError

        mock_ws_cls.load.side_effect = WorkspaceNotFoundError("not found")
        with patch("tsugite.workspace.context.build_workspace_attachments", return_value=[]):
            adapter = HTTPAgentAdapter(
                agent_name="test-agent",
                agent_config=agent_config,
                session_store=session_store,
            )

    srv = HTTPServer(
        config=http_config,
        adapters={"test-agent": adapter},
        webhook_store=webhook_store,
        agent_configs={"test-agent": agent_config},
        token_store=token_store,
    )
    srv.session_runner = session_runner
    return srv


@pytest.fixture
def test_token(server):
    _st, raw = server._token_store.create_admin_token(name="test-token")
    return raw


@pytest.fixture
def client(server):
    return TestClient(server.app)


def _create_session(store: SessionStore, sid: str, user_id: str = "web-anonymous") -> str:
    s = Session(
        id=sid,
        agent="test-agent",
        source=SessionSource.INTERACTIVE.value,
        user_id=user_id,
    )
    store.create_session(s)
    return sid


def auth(token):
    return {"Authorization": f"Bearer {token}"}


class TestSetPrimaryEndpoint:
    def test_marks_session_as_primary(self, client, test_token, session_store):
        sid = _create_session(session_store, "s-1")
        resp = client.post(f"/api/sessions/{sid}/set-primary", json={}, headers=auth(test_token))
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert session_store.get_session(sid).metadata.get("is_primary") is True

    def test_demotes_prior_primary(self, client, test_token, session_store):
        a = _create_session(session_store, "s-a")
        b = _create_session(session_store, "s-b")
        client.post(f"/api/sessions/{a}/set-primary", json={}, headers=auth(test_token))
        client.post(f"/api/sessions/{b}/set-primary", json={}, headers=auth(test_token))

        assert not session_store.get_session(a).metadata.get("is_primary")
        assert session_store.get_session(b).metadata.get("is_primary") is True

    def test_unknown_session_404(self, client, test_token):
        resp = client.post("/api/sessions/missing/set-primary", json={}, headers=auth(test_token))
        assert resp.status_code == 404


class TestClearPrimaryEndpoint:
    def test_clears_primary(self, client, test_token, session_store):
        sid = _create_session(session_store, "s-1")
        session_store.set_primary_session(sid)

        resp = client.post(
            "/api/sessions/clear-primary?agent=test-agent&user_id=web-anonymous",
            headers=auth(test_token),
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert not session_store.get_session(sid).metadata.get("is_primary")

    def test_no_op_when_nothing_primary(self, client, test_token):
        resp = client.post(
            "/api/sessions/clear-primary?agent=test-agent&user_id=web-anonymous",
            headers=auth(test_token),
        )
        assert resp.status_code == 200


class TestSessionListIncludesIsPrimary:
    def test_list_surfaces_is_primary_field(self, client, test_token, session_store):
        a = _create_session(session_store, "s-a")
        _create_session(session_store, "s-b")
        session_store.set_primary_session(a)

        resp = client.get("/api/sessions", headers=auth(test_token))
        assert resp.status_code == 200
        sessions = {s["id"]: s for s in resp.json()["sessions"]}
        assert sessions["s-a"]["is_primary"] is True
        assert sessions["s-b"]["is_primary"] is False

    def test_get_session_surfaces_is_primary(self, client, test_token, session_store):
        sid = _create_session(session_store, "s-1")
        session_store.set_primary_session(sid)

        resp = client.get(f"/api/sessions/{sid}", headers=auth(test_token))
        assert resp.status_code == 200
        assert resp.json()["is_primary"] is True
