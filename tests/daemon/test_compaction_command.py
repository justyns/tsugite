"""Tests for the /compact adapter command.

The web UI surfaces custom compaction instructions through the existing
slash-command UI: typing `/compact some instructions` is parsed by
`tsugite/daemon/web/js/views/conversation/input.js:_runCommand` into a POST
against `/api/agents/{agent}/commands/compact` with body
`{user_id, message: "some instructions"}`. The backend forwards `message`
to `_compact_session(instructions=...)`.

These tests verify the HTTP route correctly threads `message` through to
`_compact_session` and that the command list advertises the optional
instructions affordance.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

from tsugite.daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite.daemon.config import AgentConfig, HTTPConfig
from tsugite.daemon.webhook_store import WebhookStore


@pytest.fixture
def tmp_workspace(tmp_path):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    return workspace_dir


@pytest.fixture
def agent_config(tmp_workspace):
    return AgentConfig(workspace_dir=tmp_workspace, agent_file="default")


@pytest.fixture
def adapter(agent_config, tmp_path):
    from tsugite.daemon.session_store import SessionStore
    from tsugite.workspace import WorkspaceNotFoundError

    session_store = SessionStore(tmp_path / "session_store.json", context_limits={"test-agent": 128_000})

    with patch("tsugite.workspace.Workspace") as mock_ws_cls:
        mock_ws_cls.load.side_effect = WorkspaceNotFoundError("not found")
        with patch("tsugite.workspace.context.build_workspace_attachments", return_value=[]):
            return HTTPAgentAdapter(
                agent_name="test-agent",
                agent_config=agent_config,
                session_store=session_store,
            )


@pytest.fixture
def token_store(tmp_path):
    from tsugite.daemon.auth import TokenStore

    return TokenStore(tmp_path / "tokens.json")


@pytest.fixture
def admin_token(token_store):
    _, raw = token_store.create_admin_token(name="test-token")
    return raw


@pytest.fixture
def client(adapter, agent_config, tmp_path, token_store):
    server = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8374),
        adapters={"test-agent": adapter},
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
        agent_configs={"test-agent": agent_config},
        token_store=token_store,
    )
    return TestClient(server.app)


def _seed_session_with_message(adapter, user_id: str) -> str:
    """`cmd_compact` returns "No conversation to compact" early when the session
    has zero messages. Bump message_count via the public counter to bypass that.
    """
    session = adapter.session_store.get_or_create_interactive(user_id, "test-agent")
    adapter.session_store.update_token_count(session.id, 100)
    return session.id


class TestCommandList:
    def test_compact_advertised_with_instructions_hint(self, client, admin_token):
        """The command list must mention the instructions affordance so users
        discover that /compact accepts a message argument."""
        resp = client.get("/api/commands", headers={"Authorization": f"Bearer {admin_token}"})
        assert resp.status_code == 200

        commands = resp.json().get("commands", [])
        compact = next((c for c in commands if c["name"] == "compact"), None)
        assert compact is not None
        assert "instruction" in compact["description"].lower()

        params_by_name = {p["name"]: p for p in compact["params"]}
        assert "message" in params_by_name
        assert params_by_name["message"]["required"] is False


class TestCompactCommandRouting:
    def test_message_forwarded_as_instructions(self, client, adapter, admin_token):
        """`/compact <text>` (which the JS slash parser sends as
        `{message: <text>, user_id}`) reaches `_compact_session` with
        `instructions=<text>` and `reason="manual"`."""
        user_id = "compact-user-with-msg"
        _seed_session_with_message(adapter, user_id)

        captured = {}

        async def fake_compact(session_id, instructions=None, reason=None, progress_callback=None):
            captured["instructions"] = instructions
            captured["reason"] = reason

        with patch.object(adapter, "_compact_session", new=AsyncMock(side_effect=fake_compact)):
            resp = client.post(
                "/api/agents/test-agent/commands/compact",
                content=json.dumps({"user_id": user_id, "message": "focus on schema design"}),
                headers={
                    "Authorization": f"Bearer {admin_token}",
                    "Content-Type": "application/json",
                },
            )

        assert resp.status_code == 200, resp.text
        assert captured.get("instructions") == "focus on schema design"
        assert captured.get("reason") == "manual"

    def test_bare_compact_passes_no_instructions(self, client, adapter, admin_token):
        """Bare `/compact` (no message arg) reaches `_compact_session` with
        `instructions=None` so the default instructions apply."""
        user_id = "compact-user-bare"
        _seed_session_with_message(adapter, user_id)

        captured = {}

        async def fake_compact(session_id, instructions=None, reason=None, progress_callback=None):
            captured["instructions"] = instructions

        with patch.object(adapter, "_compact_session", new=AsyncMock(side_effect=fake_compact)):
            resp = client.post(
                "/api/agents/test-agent/commands/compact",
                content=json.dumps({"user_id": user_id}),
                headers={
                    "Authorization": f"Bearer {admin_token}",
                    "Content-Type": "application/json",
                },
            )

        assert resp.status_code == 200, resp.text
        assert captured.get("instructions") is None

    def test_empty_session_short_circuits_without_calling_compact(self, client, adapter, admin_token):
        """When the user has no messages yet, the command must not call
        `_compact_session` and must return a friendly no-op string."""
        user_id = "compact-user-empty"
        adapter.session_store.get_or_create_interactive(user_id, "test-agent")

        compact_mock = AsyncMock()
        with patch.object(adapter, "_compact_session", new=compact_mock):
            resp = client.post(
                "/api/agents/test-agent/commands/compact",
                content=json.dumps({"user_id": user_id, "message": "ignored"}),
                headers={
                    "Authorization": f"Bearer {admin_token}",
                    "Content-Type": "application/json",
                },
            )

        assert resp.status_code == 200, resp.text
        compact_mock.assert_not_called()
        assert "no conversation" in resp.json().get("result", "").lower()
