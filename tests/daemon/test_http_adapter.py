"""Tests for the HTTP API adapter."""

import json
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from tsugite.daemon.adapters.http import HTTPAgentAdapter, HTTPServer, SSEProgressHandler
from tsugite.daemon.config import AgentConfig, HTTPConfig, WebhookConfig


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
    return HTTPConfig(enabled=True, host="127.0.0.1", port=8374, auth_tokens=["test-token"])


@pytest.fixture
def http_config_no_auth():
    return HTTPConfig(enabled=True, host="127.0.0.1", port=8374, auth_tokens=[])


@pytest.fixture
def mock_adapter(agent_config):
    from tsugite.workspace import WorkspaceNotFoundError

    mock_session_mgr = MagicMock()
    mock_session_mgr.get_or_create_session.return_value = "daemon_test-agent_new-user"

    with patch("tsugite.workspace.Workspace") as mock_ws_cls:
        mock_ws_cls.load.side_effect = WorkspaceNotFoundError("not found")
        with patch("tsugite.workspace.context.build_workspace_attachments", return_value=[]):
            adapter = HTTPAgentAdapter(
                agent_name="test-agent",
                agent_config=agent_config,
                session_manager=mock_session_mgr,
            )
            return adapter


@pytest.fixture
def webhook_config():
    return WebhookConfig(token="whk_test", agent="test-agent", source="forgejo")


@pytest.fixture
def server(http_config, mock_adapter, webhook_config, agent_config):
    return HTTPServer(
        config=http_config,
        adapters={"test-agent": mock_adapter},
        webhooks=[webhook_config],
        agent_configs={"test-agent": agent_config},
    )


@pytest.fixture
def server_no_auth(http_config_no_auth, mock_adapter, webhook_config, agent_config):
    return HTTPServer(
        config=http_config_no_auth,
        adapters={"test-agent": mock_adapter},
        webhooks=[webhook_config],
        agent_configs={"test-agent": agent_config},
    )


@pytest.fixture
def client(server):
    return TestClient(server.app)


@pytest.fixture
def client_no_auth(server_no_auth):
    return TestClient(server_no_auth.app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "test-agent" in data["agents"]

    def test_health_no_auth_required(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200


class TestAgentsEndpoint:
    def test_list_agents_with_auth(self, client):
        resp = client.get("/api/agents", headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["agents"]) == 1
        assert data["agents"][0]["name"] == "test-agent"

    def test_list_agents_unauthorized(self, client):
        resp = client.get("/api/agents")
        assert resp.status_code == 401

    def test_list_agents_wrong_token(self, client):
        resp = client.get("/api/agents", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401

    def test_list_agents_no_auth_mode(self, client_no_auth):
        resp = client_no_auth.get("/api/agents")
        assert resp.status_code == 200


class TestChatEndpoint:
    def test_chat_unknown_agent(self, client):
        resp = client.post(
            "/api/agents/nonexistent/chat",
            json={"message": "hello"},
            headers={"Authorization": "Bearer test-token"},
        )
        assert resp.status_code == 404

    def test_chat_no_auth(self, client):
        resp = client.post("/api/agents/test-agent/chat", json={"message": "hello"})
        assert resp.status_code == 401

    def test_chat_empty_message(self, client):
        resp = client.post(
            "/api/agents/test-agent/chat",
            json={"message": ""},
            headers={"Authorization": "Bearer test-token"},
        )
        assert resp.status_code == 400

    def test_chat_invalid_json(self, client):
        resp = client.post(
            "/api/agents/test-agent/chat",
            content=b"not json",
            headers={"Authorization": "Bearer test-token", "Content-Type": "application/json"},
        )
        assert resp.status_code == 400


class TestWebhookEndpoint:
    def test_webhook_valid_token(self, client, tmp_workspace):
        resp = client.post(
            "/webhook/whk_test",
            json={"event": "push", "repo": "test/repo"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "accepted"

        inbox_dir = tmp_workspace / "inbox" / "webhooks"
        assert inbox_dir.exists()
        files = list(inbox_dir.glob("*.json"))
        assert len(files) == 1

        stored = json.loads(files[0].read_text())
        assert stored["source"] == "forgejo"
        assert stored["payload"]["event"] == "push"

    def test_webhook_invalid_token(self, client):
        resp = client.post("/webhook/bad-token", json={"data": 1})
        assert resp.status_code == 404

    def test_webhook_raw_body(self, client, tmp_workspace):
        resp = client.post(
            "/webhook/whk_test",
            content=b"raw payload data",
            headers={"Content-Type": "text/plain"},
        )
        assert resp.status_code == 202


class TestHistoryEndpoint:
    def test_history_empty_for_new_user(self, client):
        resp = client.get(
            "/api/agents/test-agent/history?user_id=brand-new-user",
            headers={"Authorization": "Bearer test-token"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["turns"] == []
        assert "conversation_id" in data

    def test_history_unknown_agent(self, client):
        resp = client.get(
            "/api/agents/nonexistent/history?user_id=someone",
            headers={"Authorization": "Bearer test-token"},
        )
        assert resp.status_code == 404

    def test_history_unauthorized(self, client):
        resp = client.get("/api/agents/test-agent/history?user_id=someone")
        assert resp.status_code == 401

    def test_history_no_auth_mode(self, client_no_auth):
        resp = client_no_auth.get("/api/agents/test-agent/history?user_id=someone")
        assert resp.status_code == 200
        data = resp.json()
        assert data["turns"] == []


class TestWebUI:
    def test_serve_ui(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Tsugite" in resp.text


class TestSSEProgressHandler:
    def test_emit_queues_event(self):
        handler = SSEProgressHandler()
        handler._emit("test", {"key": "value"})
        assert not handler.queue.empty()
        item = handler.queue.get_nowait()
        assert item == {"type": "test", "key": "value"}

    def test_done_flag(self):
        handler = SSEProgressHandler()
        assert handler.done is False
        handler.done = True
        assert handler.done is True


class TestHTTPConfig:
    def test_config_defaults(self):
        config = HTTPConfig()
        assert config.enabled is False
        assert config.host == "127.0.0.1"
        assert config.port == 8374
        assert config.auth_tokens == []

    def test_webhook_config(self):
        wh = WebhookConfig(token="whk_test", agent="myagent", source="github")
        assert wh.token == "whk_test"
        assert wh.agent == "myagent"
        assert wh.source == "github"
