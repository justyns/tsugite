import json
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite_daemon.auth import TokenStore
from tsugite_daemon.config import HTTPConfig, RuntimeDefaults
from tsugite_daemon.session_store import SessionStore
from tsugite_daemon.webhook_store import WebhookStore

from tsugite.workspace import WorkspaceNotFoundError


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "workspace"
    path.mkdir()
    return path


@pytest.fixture
def adapter(tmp_path, workspace):
    with patch("tsugite.workspace.Workspace") as ws_cls:
        ws_cls.load.side_effect = WorkspaceNotFoundError("not found")
        return HTTPAgentAdapter(
            runtime=RuntimeDefaults(workspace_dir=workspace, agent_file="default"),
            session_store=SessionStore(tmp_path / "state" / "session_store.json"),
        )


@pytest.fixture
def client(tmp_path, adapter):
    token_store = TokenStore(tmp_path / "tokens.json")
    _st, raw = token_store.create_admin_token(name="runtime-token")
    server = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8374),
        adapter=adapter,
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
        token_store=token_store,
    )
    return TestClient(server.app, headers={"Authorization": f"Bearer {raw}"})


def _runtime(client) -> dict:
    resp = client.get("/api/runtime")
    assert resp.status_code == 200
    return resp.json()


def test_fresh_daemon_reports_no_effective_model(client):
    assert _runtime(client)["effective_model"] is None


def test_daemon_default_model_is_the_effective_model(client, adapter):
    adapter.runtime.model = "openai:gpt-4o"
    assert _runtime(client)["effective_model"] == "openai:gpt-4o"


def test_global_default_model_is_the_effective_model(client, adapter, xdg_config_file):
    xdg_config_file.write_text(json.dumps({"default_model": "openai:gpt-4o-mini"}))
    runtime = _runtime(client)
    assert runtime["model"] is None
    assert runtime["effective_model"] == "openai:gpt-4o-mini"
