"""Gateway restart: preflight, drain, and the flag the CLI re-execs on.

The gateway stands in for a started daemon the way test_config_reload does: a real
Gateway with `_http_server` stubbed, so the drain can be driven by hand.
"""

import asyncio
import json
import logging
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from tsugite_daemon import gateway as gateway_mod
from tsugite_daemon.config import load_daemon_config
from tsugite_daemon.gateway import Gateway, run_daemon

from .conftest import _wait_until


def _write_daemon_config(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "alpha.md").write_text("---\nname: alpha\n---\nYou are alpha.\n")
    cfg_path = tmp_path / "daemon.yaml"
    cfg_path.write_text(
        f"state_dir: {tmp_path / 'state'}\n"
        "http:\n  enabled: true\n  host: 127.0.0.1\n  port: 8321\n"
        f"default_agent_file: alpha.md\ndefault_workspace_dir: {ws}\n"
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


class TestRejectsNewChats:
    """Nothing must be able to push the drain out indefinitely with fresh turns."""

    @pytest.fixture
    def client(self, tmp_path):
        from starlette.testclient import TestClient
        from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
        from tsugite_daemon.auth import TokenStore
        from tsugite_daemon.config import HTTPConfig, RuntimeDefaults
        from tsugite_daemon.session_store import SessionStore

        workspace = tmp_path / "ws"
        workspace.mkdir()
        agent_config = RuntimeDefaults(workspace_dir=workspace, agent_file="default")
        adapter = HTTPAgentAdapter(
            runtime=agent_config,
            session_store=SessionStore(tmp_path / "session_store.json"),
        )
        token_store = TokenStore(tmp_path / "tokens.json")
        _token, raw = token_store.create_admin_token(name="t")
        server = HTTPServer(
            config=HTTPConfig(enabled=True, host="127.0.0.1", port=8374),
            adapter=adapter,
            webhook_store=None,
            token_store=token_store,
            gateway=SimpleNamespace(restart_requested=True, config_path=None),
        )
        return TestClient(server.app), raw

    def test_chat_is_refused_while_a_restart_is_pending(self, client):
        http_client, token = client

        resp = http_client.post(
            "/api/chat",
            json={"message": "hello", "user_id": "u1"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert resp.status_code == 409
        assert resp.json()["code"] == "daemon_restarting"


class TestPreflight:
    def test_passes_on_a_loadable_config(self, gateway):
        assert gateway.preflight_restart() == []

    def test_reports_an_unloadable_daemon_config_instead_of_raising(self, gateway):
        gateway.config_path.write_text("default_workspace_dir: [unclosed\n")

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
    async def test_a_second_request_starts_no_second_drain(self, gateway):
        gateway._http_server._active_chats["chat-1"] = object()
        before = asyncio.all_tasks()

        gateway.request_restart()
        gateway.request_restart()

        assert len(asyncio.all_tasks() - before) == 1

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


class TestRestartNotice:
    @pytest.mark.asyncio
    async def test_the_drain_leaves_a_notice_for_the_next_daemon(self, gateway):
        gateway.config.state_dir.mkdir()
        gateway.request_restart("new plugin", "origin-1")

        assert await _wait_until(lambda: gateway._http_server.stop.called)
        notice = json.loads(gateway._restart_notice_path.read_text())
        assert (notice["reason"], notice["session_id"]) == ("new plugin", "origin-1")

    @pytest.fixture
    def after_restart(self, gateway, monkeypatch):
        """A freshly started daemon holding a notice, with one plugin loaded and one failed."""
        import tsugite.plugins
        import tsugite.tools
        import tsugite.tools.sessions

        gateway._restart_notice_path.parent.mkdir(parents=True, exist_ok=True)
        gateway._restart_notice_path.write_text(
            json.dumps({"reason": "new plugin", "session_id": "origin-1", "requested_at": 0})
        )
        info = tsugite.plugins.PluginInfo
        monkeypatch.setattr(
            tsugite.tools,
            "plugin_results",
            [
                info("good", "g", "good.py", loaded=True),
                info("bad", "g", "bad.py", error="boom"),
                info("pkg", "g", "x"),
            ],
        )
        monkeypatch.setattr(
            tsugite.plugins,
            "local_plugin_files",
            lambda: [SimpleNamespace(name="good"), SimpleNamespace(name="bad")],
        )
        sent = []

        def reply(message, session_id):
            sent.append((session_id, message))
            return {}

        monkeypatch.setattr(tsugite.tools.sessions, "session_reply", reply)
        return sent

    @pytest.mark.asyncio
    async def test_the_new_daemon_tells_the_session(self, gateway, after_restart):
        await gateway._send_restart_notice()

        [(session_id, message)] = after_restart
        assert session_id == "origin-1"
        assert "Restart reason: new plugin" in message
        assert f"pid {os.getpid()}" in message
        assert "loaded: good" in message
        assert "- bad: boom" in message
        assert "pkg" not in message
        assert not gateway._restart_notice_path.exists()

    @pytest.mark.parametrize("failure", ["gone", "queue full"])
    @pytest.mark.asyncio
    async def test_an_undelivered_notice_is_dropped_and_logged_at_info(
        self, gateway, after_restart, monkeypatch, caplog, failure
    ):
        import tsugite.tools.sessions

        def reply(message, session_id):
            if failure == "gone":
                raise KeyError(session_id)
            return {"error": failure}

        monkeypatch.setattr(tsugite.tools.sessions, "session_reply", reply)

        with caplog.at_level(logging.INFO):
            await gateway._send_restart_notice()

        assert [r.levelno for r in caplog.records if "Dropped" in r.message] == [logging.INFO]
        assert not gateway._restart_notice_path.exists()

    @pytest.mark.asyncio
    async def test_a_corrupt_notice_is_removed(self, gateway, after_restart):
        gateway._restart_notice_path.write_text("{not json")

        with pytest.raises(json.JSONDecodeError):
            await gateway._send_restart_notice()

        assert not gateway._restart_notice_path.exists()


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
