"""Tests for the read-only registry introspection endpoints:
GET /api/plugins (thin wrapper over tsugite.plugins.discover_plugins) and
GET /api/tools (thin wrapper over the tsugite.tools registry).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPServer
from tsugite_daemon.auth import TokenStore
from tsugite_daemon.config import HTTPConfig
from tsugite_daemon.webhook_store import WebhookStore

from tsugite.plugins import PluginInfo
from tsugite.tools import ToolInfo


@pytest.fixture
def token_store(tmp_path):
    return TokenStore(tmp_path / "tokens.json")


@pytest.fixture
def test_token(token_store):
    _st, raw = token_store.create_admin_token(name="introspection-token")
    return raw


@pytest.fixture
def server(tmp_path, token_store):
    return HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8585),
        adapters={},
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
        agent_configs={},
        token_store=token_store,
    )


@pytest.fixture
def client(server):
    return TestClient(server.app)


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


class TestPluginsEndpoint:
    def test_requires_auth(self, client):
        assert client.get("/api/plugins").status_code == 401

    def test_shape_from_real_registry(self, client, test_token):
        resp = client.get("/api/plugins", headers=_auth(test_token))
        assert resp.status_code == 200
        plugins = resp.json()["plugins"]
        assert isinstance(plugins, list)
        for p in plugins:
            assert set(p.keys()) == {"name", "group", "enabled", "loaded", "error"}

    def test_wraps_discover_plugins(self, client, test_token):
        fake = [
            PluginInfo(name="pty", group="tsugite.plugins", entry_point="tsugite_pty:x", enabled=True, loaded=False),
            PluginInfo(
                name="broken", group="tsugite.tools", entry_point="pkg:f", enabled=False, loaded=False, error="boom"
            ),
        ]
        with patch("tsugite.plugins.discover_plugins", return_value=fake):
            resp = client.get("/api/plugins", headers=_auth(test_token))
        assert resp.status_code == 200
        assert resp.json() == {
            "plugins": [
                {"name": "pty", "group": "tsugite.plugins", "enabled": True, "loaded": False, "error": None},
                {"name": "broken", "group": "tsugite.tools", "enabled": False, "loaded": False, "error": "boom"},
            ]
        }


class TestToolsEndpoint:
    def test_requires_auth(self, client):
        assert client.get("/api/tools").status_code == 401

    def test_shape_from_real_registry(self, client, test_token):
        # Robust to whatever the process-global tool registry currently holds:
        # a full-suite run mutates the shared tsugite.tools._tools dict, so this
        # asserts only the serialization invariants, not specific tools. Exact
        # builtin/plugin content is pinned deterministically below.
        resp = client.get("/api/tools", headers=_auth(test_token))
        assert resp.status_code == 200
        tools = resp.json()["tools"]
        assert isinstance(tools, list)
        for t in tools:
            assert set(t.keys()) == {"name", "category", "description", "source"}
            assert t["source"] in {"builtin", "plugin"}

    def test_serializes_builtin_and_plugin(self, client, test_token):
        def _core():
            """Read a file from disk."""

        def _ext():
            """Plugin tool docstring."""

        _core.__module__ = "tsugite.tools.fs"
        _ext.__module__ = "tsugite_pty.tools"
        fake_tools = {
            "read_file": ToolInfo(
                name="read_file", func=_core, description="Read a file from disk.", parameters={}, category="fs"
            ),
            "ext_tool": ToolInfo(name="ext_tool", func=_ext, description="ext", parameters={}, category=None),
        }
        with patch("tsugite.tools._ensure_tools_loaded"), patch.dict("tsugite.tools._tools", fake_tools, clear=True):
            resp = client.get("/api/tools", headers=_auth(test_token))
        assert resp.status_code == 200
        by_name = {t["name"]: t for t in resp.json()["tools"]}
        # A core tsugite.tools.* tool serializes as builtin, keeping its category + first-line doc.
        assert by_name["read_file"] == {
            "name": "read_file",
            "category": "fs",
            "description": "Read a file from disk.",
            "source": "builtin",
        }
        # A non-core module is a plugin; category falls back to the module basename when unset.
        assert by_name["ext_tool"]["source"] == "plugin"
        assert by_name["ext_tool"]["category"] == "tools"
