"""Gateway restart: preflight, drain, and the flag the CLI re-execs on.

The gateway stands in for a started daemon the way test_config_reload does: a real
Gateway with `_http_server` stubbed, so the drain can be driven by hand.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from tsugite_daemon import gateway as gateway_mod
from tsugite_daemon.config import load_daemon_config
from tsugite_daemon.gateway import Gateway, run_daemon


def _write_daemon_config(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "alpha.md").write_text("---\nname: alpha\n---\nYou are alpha.\n")
    cfg_path = tmp_path / "daemon.yaml"
    cfg_path.write_text(
        f"state_dir: {tmp_path / 'state'}\n"
        "http:\n  enabled: true\n  host: 127.0.0.1\n  port: 8321\n"
        f"agents:\n  alpha:\n    agent_file: alpha.md\n    workspace_dir: {ws}\n"
    )
    return cfg_path


@pytest.fixture
def gateway(tmp_path):
    cfg_path = _write_daemon_config(tmp_path)
    gateway = Gateway(load_daemon_config(cfg_path), config_path=cfg_path)
    gateway._http_server = SimpleNamespace(
        _active_chats={},
        _server=SimpleNamespace(force_exit=False),
        stop=AsyncMock(),
    )
    gateway._drain_poll = 0.01
    return gateway


async def _wait_until(predicate, timeout: float = 2.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            return False
        await asyncio.sleep(0.01)
    return True


class TestRejectsNewChats:
    """Nothing must be able to push the drain out indefinitely with fresh turns."""

    @pytest.fixture
    def client(self, tmp_path):
        from starlette.testclient import TestClient
        from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
        from tsugite_daemon.auth import TokenStore
        from tsugite_daemon.config import AgentConfig, HTTPConfig
        from tsugite_daemon.session_store import SessionStore

        workspace = tmp_path / "ws"
        workspace.mkdir()
        agent_config = AgentConfig(workspace_dir=workspace, agent_file="default")
        adapter = HTTPAgentAdapter(
            agent_name="alpha",
            agent_config=agent_config,
            session_store=SessionStore(tmp_path / "session_store.json"),
        )
        token_store = TokenStore(tmp_path / "tokens.json")
        _token, raw = token_store.create_admin_token(name="t")
        server = HTTPServer(
            config=HTTPConfig(enabled=True, host="127.0.0.1", port=8374),
            adapters={"alpha": adapter},
            webhook_store=None,
            agent_configs={"alpha": agent_config},
            token_store=token_store,
            gateway=SimpleNamespace(restart_requested=True, config_path=None),
        )
        return TestClient(server.app), raw

    def test_chat_is_refused_while_a_restart_is_pending(self, client):
        http_client, token = client

        resp = http_client.post(
            "/api/agents/alpha/chat",
            json={"message": "hello", "user_id": "u1"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert resp.status_code == 409
        assert resp.json()["code"] == "daemon_restarting"


class TestPreflight:
    def test_passes_on_a_loadable_config(self, gateway):
        assert gateway.preflight_restart() == []

    def test_reports_an_unloadable_daemon_config_instead_of_raising(self, gateway):
        gateway.config_path.write_text("agents: [unclosed\n")

        problems = gateway.preflight_restart()

        assert len(problems) == 1
        assert "daemon.yaml" in problems[0]


class TestDrain:
    @pytest.mark.asyncio
    async def test_waits_for_the_in_flight_turn(self, gateway):
        gateway._http_server._active_chats["chat-1"] = object()

        gateway.request_restart()

        assert gateway.restart_requested
        await asyncio.sleep(0.05)
        assert not gateway._http_server.stop.called, "must not shut down while a turn is in flight"

        gateway._http_server._active_chats.clear()

        assert await _wait_until(lambda: gateway._http_server.stop.called)
        assert gateway._http_server._server.force_exit is False
        # Draining must not clear the intent, or run_daemon reports no restart.
        assert gateway.restart_requested

    @pytest.mark.asyncio
    async def test_deadline_forces_the_shutdown(self, gateway):
        gateway._drain_deadline = 0
        gateway._http_server._active_chats["chat-1"] = object()

        gateway.request_restart()

        assert await _wait_until(lambda: gateway._http_server.stop.called)
        assert gateway._http_server._server.force_exit is True
        assert gateway.restart_requested, "giving up on the drain must still re-exec"

    @pytest.mark.asyncio
    async def test_a_signal_during_the_drain_cancels_the_restart(self, gateway, monkeypatch):
        monkeypatch.setattr(gateway, "_shutdown", AsyncMock())
        gateway._http_server._active_chats["chat-1"] = object()
        before = asyncio.all_tasks()
        gateway.request_restart()
        drain = (asyncio.all_tasks() - before).pop()
        assert gateway.restart_requested

        gateway._on_signal()

        assert not gateway.restart_requested, "Ctrl-C during the drain must not re-exec the daemon"
        # This turn never finishes: a drain that ignored the cleared flag would poll
        # until its deadline, holding the gateway alive.
        await asyncio.wait_for(drain, timeout=1)


class TestRunDaemonReturn:
    """The flag has to survive Gateway -> run_daemon, or daemon_main never re-execs."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("requested", [True, False])
    async def test_reports_whether_a_restart_was_requested(self, tmp_path, monkeypatch, requested):
        cfg_path = _write_daemon_config(tmp_path)

        async def fake_start(self):
            self.restart_requested = requested

        # Only start() is faked, so the real constructor and the real return run.
        monkeypatch.setattr(Gateway, "start", fake_start)
        # Both reconfigure process-wide state that must not leak into the session.
        monkeypatch.setattr(gateway_mod, "_configure_logging", lambda config: None)
        monkeypatch.setattr(gateway_mod, "_install_crash_hooks", lambda: None)
        # run_daemon imports this inside the function, so patch it at its source.
        monkeypatch.setattr("tsugite.secrets.configure_from_daemon", lambda config: None)

        assert await run_daemon(cfg_path) is requested
