"""Tests for session metadata HTTP API endpoints."""

from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from tsugite.daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite.daemon.config import AgentConfig, HTTPConfig
from tsugite.daemon.session_runner import SessionRunner
from tsugite.daemon.session_store import Session, SessionSource, SessionStore
from tsugite.daemon.webhook_store import WebhookStore


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
    http_config = HTTPConfig(enabled=True, host="127.0.0.1", port=8374)
    webhook_store = WebhookStore(tmp_path / "webhooks.json")

    from tsugite.daemon.auth import TokenStore

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


@pytest.fixture
def session_id(session_store):
    session = Session(
        id="",
        agent="test-agent",
        source=SessionSource.BACKGROUND.value,
        prompt="hello",
    )
    created = session_store.create_session(session)
    return created.id


def auth(token):
    return {"Authorization": f"Bearer {token}"}


class TestGetMetadata:
    def test_get_empty_metadata(self, client, test_token, session_id):
        resp = client.get(f"/api/sessions/{session_id}/metadata", headers=auth(test_token))
        assert resp.status_code == 200
        assert resp.json() == {"metadata": {}}

    def test_get_metadata_after_set(self, client, test_token, session_id, session_store):
        session_store.set_metadata(session_id, "env", "staging")
        resp = client.get(f"/api/sessions/{session_id}/metadata", headers=auth(test_token))
        assert resp.status_code == 200
        assert resp.json() == {"metadata": {"env": "staging"}}

    def test_get_metadata_not_found(self, client, test_token):
        resp = client.get("/api/sessions/nonexistent/metadata", headers=auth(test_token))
        assert resp.status_code == 404

    def test_get_metadata_no_auth(self, client, session_id):
        resp = client.get(f"/api/sessions/{session_id}/metadata")
        assert resp.status_code == 401


class TestUpdateMetadata:
    def test_update_metadata(self, client, test_token, session_id):
        resp = client.patch(
            f"/api/sessions/{session_id}/metadata",
            json={"env": "prod", "team": "backend"},
            headers=auth(test_token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["metadata"] == {"env": "prod", "team": "backend"}

    def test_update_metadata_merges(self, client, test_token, session_id, session_store):
        session_store.set_metadata(session_id, "existing", "value")
        resp = client.patch(
            f"/api/sessions/{session_id}/metadata",
            json={"new_key": "new_value"},
            headers=auth(test_token),
        )
        assert resp.status_code == 200
        assert resp.json()["metadata"] == {"existing": "value", "new_key": "new_value"}

    def test_update_metadata_rejects_read_only(self, client, test_token, session_id):
        resp = client.patch(
            f"/api/sessions/{session_id}/metadata",
            json={"source": "hacked"},
            headers=auth(test_token),
        )
        assert resp.status_code == 400
        assert "read-only" in resp.json()["error"]

    def test_update_metadata_invalid_json(self, client, test_token, session_id):
        resp = client.patch(
            f"/api/sessions/{session_id}/metadata",
            content=b"not json",
            headers={**auth(test_token), "Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    def test_update_metadata_non_object(self, client, test_token, session_id):
        resp = client.patch(
            f"/api/sessions/{session_id}/metadata",
            json=["a", "b"],
            headers=auth(test_token),
        )
        assert resp.status_code == 400
        assert "JSON object" in resp.json()["error"]

    def test_update_metadata_not_found(self, client, test_token):
        resp = client.patch(
            "/api/sessions/nonexistent/metadata",
            json={"key": "val"},
            headers=auth(test_token),
        )
        assert resp.status_code == 400

    def test_update_metadata_no_auth(self, client, session_id):
        resp = client.patch(f"/api/sessions/{session_id}/metadata", json={"k": "v"})
        assert resp.status_code == 401

    def test_update_topic_at_cap(self, client, test_token, session_id):
        topic = "x" * 160
        resp = client.patch(
            f"/api/sessions/{session_id}/metadata",
            json={"topic": topic},
            headers=auth(test_token),
        )
        assert resp.status_code == 200
        assert resp.json()["metadata"]["topic"] == topic

    def test_update_topic_over_cap_rejected(self, client, test_token, session_id):
        resp = client.patch(
            f"/api/sessions/{session_id}/metadata",
            json={"topic": "x" * 161},
            headers=auth(test_token),
        )
        assert resp.status_code == 400
        assert "160" in resp.json()["error"]

    def test_clear_topic_via_delete(self, client, test_token, session_id, session_store):
        session_store.set_metadata(session_id, "topic", "old topic")
        resp = client.delete(
            f"/api/sessions/{session_id}/metadata/topic",
            headers=auth(test_token),
        )
        assert resp.status_code == 200
        assert "topic" not in resp.json()["metadata"]


class TestDeleteMetadata:
    def test_delete_metadata(self, client, test_token, session_id, session_store):
        session_store.set_metadata(session_id, "to_delete", "bye")
        resp = client.delete(
            f"/api/sessions/{session_id}/metadata/to_delete",
            headers=auth(test_token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "to_delete" not in data["metadata"]

    def test_delete_metadata_missing_key(self, client, test_token, session_id):
        resp = client.delete(
            f"/api/sessions/{session_id}/metadata/nonexistent",
            headers=auth(test_token),
        )
        assert resp.status_code == 400
        assert "not found" in resp.json()["error"]

    def test_delete_metadata_read_only(self, client, test_token, session_id):
        resp = client.delete(
            f"/api/sessions/{session_id}/metadata/source",
            headers=auth(test_token),
        )
        assert resp.status_code == 400
        assert "read-only" in resp.json()["error"]

    def test_delete_metadata_session_not_found(self, client, test_token):
        resp = client.delete(
            "/api/sessions/nonexistent/metadata/key",
            headers=auth(test_token),
        )
        assert resp.status_code == 400

    def test_delete_metadata_no_auth(self, client, session_id):
        resp = client.delete(f"/api/sessions/{session_id}/metadata/key")
        assert resp.status_code == 401


class TestListSessionsIncludesMetadata:
    def test_api_list_sessions_has_metadata(self, client, test_token, session_id, session_store):
        session_store.set_metadata(session_id, "env", "staging")
        resp = client.get("/api/sessions", headers=auth(test_token))
        assert resp.status_code == 200
        sessions = resp.json()["sessions"]
        match = [s for s in sessions if s["id"] == session_id]
        assert len(match) == 1
        assert match[0]["metadata"] == {"env": "staging"}
