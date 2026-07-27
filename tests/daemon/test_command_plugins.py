"""Plugin-contributed slash commands (`tsugite.commands` entry-point group) and
the `CommandParam.widget` autocomplete hint the frontend consumes.

Built-in commands register when `tsugite_daemon.commands` is imported; plugin
commands must be discovered + imported before the first `/api/commands` or
command-run resolves. The trigger lives in `get_commands()` so every caller
(the HTTP list/run endpoints and the Discord sync) sees plugin commands.
"""

from __future__ import annotations

import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPServer
from tsugite_daemon.auth import TokenStore
from tsugite_daemon.config import HTTPConfig
from tsugite_daemon.webhook_store import WebhookStore


def _make_entry_point(name, value, group):
    ep = MagicMock()
    ep.name = name
    ep.value = value
    ep.group = group
    return ep


# ── loader: a tsugite.commands module's command lands in get_commands() ──


def test_load_command_plugins_registers_into_get_commands():
    from tsugite_daemon.commands import adapter_command, get_commands

    from tsugite.plugins import load_command_plugins

    def _register_on_import():
        @adapter_command(name="_pytest_plugin_cmd", description="fake plugin command")
        async def _handler(adapter):
            return "ok"

        return types.ModuleType("_fake_command_plugin")

    ep = _make_entry_point("fake-cmds", "_fake_command_plugin", "tsugite.commands")
    ep.load.side_effect = _register_on_import
    try:
        with patch("tsugite.plugins.importlib.metadata.entry_points", return_value=[ep]):
            results = load_command_plugins()
        assert results[0].loaded is True
        ep.load.assert_called_once()  # module-only: import IS the registration
        assert "_pytest_plugin_cmd" in get_commands()
    finally:
        get_commands().pop("_pytest_plugin_cmd", None)


def test_get_commands_triggers_plugin_load_once():
    """get_commands() is the single chokepoint every caller hits, so the plugin
    load fires there, guarded so it only runs once."""
    import tsugite_daemon.commands as cmds

    cmds._command_plugins_loaded = False
    try:
        with patch("tsugite.plugins.load_command_plugins") as mock_load:
            cmds.get_commands()
            cmds.get_commands()
        mock_load.assert_called_once()
    finally:
        cmds._command_plugins_loaded = False


# ── widget hint serializes through GET /api/commands ──


@pytest.fixture
def token(tmp_path):
    store = TokenStore(tmp_path / "tokens.json")
    _st, raw = store.create_admin_token(name="commands-test")
    return store, raw


@pytest.fixture
def client(tmp_path, token):
    store, _raw = token
    server = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8587),
        adapters={"smokeagent": SimpleNamespace()},
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
        agent_configs={},
        token_store=store,
    )
    return TestClient(server.app)


def _auth(token):
    return {"Authorization": f"Bearer {token[1]}"}


def test_widget_hint_serialized_through_api_commands(client, token):
    resp = client.get("/api/commands", headers=_auth(token))
    assert resp.status_code == 200
    commands = {c["name"]: c for c in resp.json()["commands"]}

    model_params = {p["name"]: p for p in commands["model"]["params"]}
    assert model_params["message"]["widget"] == "model"
    # Plain params omit the key entirely (like `choices`), not widget=null.
    assert "widget" not in model_params["user_id"]

    effort_params = {p["name"]: p for p in commands["effort"]["params"]}
    assert effort_params["message"]["widget"] == "effort"

    sessions_params = {p["name"]: p for p in commands["sessions"]["params"]}
    assert sessions_params["status"]["choices"] == ["running", "completed", "failed"]
    assert "widget" not in sessions_params["status"]


# ── example plugin command: /terminals ──


class _FakeTerminalStore:
    def __init__(self, terminals):
        self._terminals = terminals

    def list_all(self):
        return list(self._terminals)


@pytest.mark.asyncio
async def test_terminals_command_registers_and_summarizes():
    import tsugite_pty.commands  # noqa: F401 -- import runs the @adapter_command decorator
    from tsugite_daemon.commands import get_commands

    commands = get_commands()
    assert "terminals" in commands
    handler = commands["terminals"].handler

    empty = SimpleNamespace(terminal_store=_FakeTerminalStore([]))
    assert await handler(empty) == "No terminals found."

    seeded = SimpleNamespace(
        terminal_store=_FakeTerminalStore(
            [
                SimpleNamespace(id="term-aaaa1111", cmd="htop", state="running"),
                SimpleNamespace(id="term-bbbb2222", cmd="psql prod", state="succeeded"),
            ]
        )
    )
    out = await handler(seeded)
    assert "term-aaaa1111" in out
    assert "htop" in out
    assert "running" in out
    assert "psql prod" in out


@pytest.mark.asyncio
async def test_terminals_command_reports_missing_runtime():
    import tsugite_pty.commands  # noqa: F401
    from tsugite_daemon.commands import get_commands

    handler = get_commands()["terminals"].handler
    out = await handler(SimpleNamespace())
    assert "terminal runtime" in out.lower()
