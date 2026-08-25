"""GET/PUT /api/hooks: read, validate, and write the agent
workspace's .tsugite/hooks.yaml through the daemon (saves apply on the next
hook firing - the loader reads the file fresh every time)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPServer
from tsugite_daemon.auth import TokenStore
from tsugite_daemon.config import HTTPConfig
from tsugite_daemon.webhook_store import WebhookStore

VALID = """hooks:
  post_tool:
    - name: index
      tools: [write_file]
      run: ["echo", "indexed"]
  pre_message:
    - run: "uridx search {{ message }}"
      capture_as: rag_context
"""


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


@pytest.fixture
def token(tmp_path):
    store = TokenStore(tmp_path / "tokens.json")
    _st, raw = store.create_admin_token(name="hooks-test")
    return store, raw


@pytest.fixture
def client(tmp_path, workspace, token):
    store, _raw = token
    fake_adapter = SimpleNamespace(runtime=SimpleNamespace(workspace_dir=workspace))
    server = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8586),
        adapter=fake_adapter,
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
        token_store=store,
    )
    return TestClient(server.app)


def _auth(token):
    return {"Authorization": f"Bearer {token[1]}"}


def test_requires_auth(client):
    assert client.get("/api/hooks").status_code == 401


def test_get_missing_file_reports_empty(client, token, workspace):
    resp = client.get("/api/hooks", headers=_auth(token))
    assert resp.status_code == 200
    body = resp.json()
    assert body["exists"] is False
    assert body["raw"] == ""
    assert body["phases"] == {}
    assert body["error"] is None
    assert body["path"].endswith(".tsugite/hooks.yaml")


def test_put_validates_and_writes(client, token, workspace):
    resp = client.put("/api/hooks", json={"raw": VALID}, headers=_auth(token))
    assert resp.status_code == 200
    body = resp.json()
    assert body["exists"] is True
    assert (workspace / ".tsugite" / "hooks.yaml").read_text() == VALID
    assert [r["name"] for r in body["phases"]["post_tool"]] == ["index"]
    assert body["phases"]["post_tool"][0]["run"] == "echo indexed"
    assert body["phases"]["pre_message"][0]["capture_as"] == "rag_context"


def test_put_rejects_bad_yaml_without_writing(client, token, workspace):
    resp = client.put("/api/hooks", json={"raw": "hooks: [unclosed"}, headers=_auth(token))
    assert resp.status_code == 400
    assert "invalid YAML" in resp.json()["error"]
    assert not (workspace / ".tsugite" / "hooks.yaml").exists()


def test_put_rejects_schema_violation(client, token, workspace):
    # shell hooks require `run`.
    bad = "hooks:\n  post_tool:\n    - name: broken\n"
    resp = client.put("/api/hooks", json={"raw": bad}, headers=_auth(token))
    assert resp.status_code == 400
    assert "run" in resp.json()["error"]
    assert not (workspace / ".tsugite" / "hooks.yaml").exists()


def test_put_requires_top_level_hooks_key(client, token):
    resp = client.put("/api/hooks", json={"raw": "post_tool: []"}, headers=_auth(token))
    assert resp.status_code == 400
    assert "hooks" in resp.json()["error"]


def test_get_surfaces_parse_error_of_existing_file(client, token, workspace):
    (workspace / ".tsugite").mkdir()
    (workspace / ".tsugite" / "hooks.yaml").write_text("hooks:\n  post_tool:\n    - name: broken\n")
    resp = client.get("/api/hooks", headers=_auth(token))
    assert resp.status_code == 200
    body = resp.json()
    assert body["exists"] is True
    assert body["error"] is not None
    assert body["phases"] is None
