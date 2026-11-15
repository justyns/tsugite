"""Tests for the /compact adapter command.

The web UI surfaces custom compaction instructions through the existing
slash-command UI: typing `/compact some instructions` is parsed by
`tsugite/daemon/web/js/views/conversation/input.js:_runCommand` into a POST
against `/api/commands/compact` with body
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
from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite_daemon.config import HTTPConfig, RuntimeDefaults
from tsugite_daemon.webhook_store import WebhookStore


@pytest.fixture
def tmp_workspace(tmp_path):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    return workspace_dir


@pytest.fixture
def agent_config(tmp_workspace):
    return RuntimeDefaults(workspace_dir=tmp_workspace, agent_file="default")


@pytest.fixture
def adapter(agent_config, tmp_path):
    from tsugite_daemon.session_store import SessionStore

    from tsugite.workspace import WorkspaceNotFoundError

    session_store = SessionStore(tmp_path / "session_store.json", default_context_limit=128_000)

    with patch("tsugite.workspace.Workspace") as mock_ws_cls:
        mock_ws_cls.load.side_effect = WorkspaceNotFoundError("not found")
        return HTTPAgentAdapter(
            runtime=agent_config,
            session_store=session_store,
        )


@pytest.fixture
def token_store(tmp_path):
    from tsugite_daemon.auth import TokenStore

    return TokenStore(tmp_path / "tokens.json")


@pytest.fixture
def admin_token(token_store):
    _, raw = token_store.create_admin_token(name="test-token")
    return raw


@pytest.fixture
def client(adapter, agent_config, tmp_path, token_store):
    server = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8374),
        adapter=adapter,
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
        token_store=token_store,
    )
    return TestClient(server.app)


def _seed_session_with_message(adapter, user_id: str) -> str:
    """`cmd_compact` returns "No conversation to compact" early when the session
    has zero messages. Bump message_count via the public counter to bypass that.
    """
    session = adapter.session_store.get_or_create_interactive(user_id)
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
                "/api/commands/compact",
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
                "/api/commands/compact",
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
        adapter.session_store.get_or_create_interactive(user_id)

        compact_mock = AsyncMock()
        with patch.object(adapter, "_compact_session", new=compact_mock):
            resp = client.post(
                "/api/commands/compact",
                content=json.dumps({"user_id": user_id, "message": "ignored"}),
                headers={
                    "Authorization": f"Bearer {admin_token}",
                    "Content-Type": "application/json",
                },
            )

        assert resp.status_code == 200, resp.text
        compact_mock.assert_not_called()
        assert "no conversation" in resp.json().get("result", "").lower()


class TestCompactRouteRoutesBySessionId:
    """The web UI's compact button posts to /api/chat/compact with the
    selected session_id. Without per-session routing, _compact targets the
    user's default interactive session, which is often empty when the user is
    on a named/secondary session and 404s with 'no session to compact'.
    """

    def _seed_named_session(self, adapter, user_id: str, session_id: str):
        from tsugite_daemon.session_store import Session, SessionSource

        s = Session(
            id=session_id,
            source=SessionSource.INTERACTIVE.value,
            user_id=user_id,
        )
        adapter.session_store.create_session(s)
        adapter.session_store.update_token_count(session_id, 100)
        return s

    def test_compact_routes_to_named_session(self, client, adapter, admin_token):
        user_id = "alice"
        # Default interactive session — empty
        adapter.session_store.get_or_create_interactive(user_id)
        # Named session the user is actively viewing
        named = self._seed_named_session(adapter, user_id, "named-A")

        captured = {}

        async def fake_compact(session_id, instructions=None, reason=None, progress_callback=None, **kwargs):
            captured["session_id"] = session_id

        with patch.object(adapter, "_compact_session", new=AsyncMock(side_effect=fake_compact)):
            resp = client.post(
                "/api/chat/compact",
                content=json.dumps({"user_id": user_id, "session_id": named.id}),
                headers={
                    "Authorization": f"Bearer {admin_token}",
                    "Content-Type": "application/json",
                },
            )

        assert resp.status_code == 200, resp.text
        assert captured.get("session_id") == named.id

    def test_compact_falls_back_to_default_without_session_id(self, client, adapter, admin_token):
        """No session_id: behave like the legacy slash-command — target default."""
        user_id = "bob"
        default = adapter.session_store.get_or_create_interactive(user_id)
        adapter.session_store.update_token_count(default.id, 50)

        captured = {}

        async def fake_compact(session_id, instructions=None, reason=None, progress_callback=None, **kwargs):
            captured["session_id"] = session_id

        with patch.object(adapter, "_compact_session", new=AsyncMock(side_effect=fake_compact)):
            resp = client.post(
                "/api/chat/compact",
                content=json.dumps({"user_id": user_id}),
                headers={
                    "Authorization": f"Bearer {admin_token}",
                    "Content-Type": "application/json",
                },
            )

        assert resp.status_code == 200, resp.text
        assert captured.get("session_id") == default.id


def _post_compact(client, admin_token, body):
    return client.post(
        "/api/chat/compact",
        content=json.dumps(body),
        headers={"Authorization": f"Bearer {admin_token}", "Content-Type": "application/json"},
    )


class TestSummarizeIntoNewChat:
    def _seed_two_turn_session(self, adapter, user_id):
        from tests.history_helpers import seed_history_session

        session_id = _seed_session_with_message(adapter, user_id)
        storage = seed_history_session(session_id)
        for i in range(2):
            storage.record("user_input", text=f"message {i}")
            storage.record("model_response", raw_content=f"reply {i}")
        return session_id

    def _compact(self, client, admin_token, body, summarized_with):
        async def fake_summarize(messages, model=None, max_context_tokens=None, progress_callback=None):
            summarized_with.append((model, messages))
            return "Summary of everything"

        with (
            patch("tsugite_daemon.memory.summarize_session", new=fake_summarize),
            patch("tsugite.hooks.fire_compact_hooks", new_callable=AsyncMock, return_value=[]),
        ):
            return _post_compact(client, admin_token, body)

    def test_route_summarizes_whole_chat_into_successor(self, client, adapter, admin_token, history_dir):
        from tests.history_helpers import load_history_session

        user_id = "summarize-user"
        old_id = self._seed_two_turn_session(adapter, user_id)
        summarized_with = []

        resp = self._compact(
            client, admin_token, {"user_id": user_id, "session_id": old_id, "retain_recent": False}, summarized_with
        )

        assert resp.status_code == 200, resp.text
        new_id = resp.json()["new_conversation_id"]
        assert new_id != old_id
        new_events = load_history_session(new_id).load_events()
        assert [e.type for e in new_events if e.type != "session_start"] == ["compaction"]
        compaction = new_events[-1].data
        assert compaction["replaced_count"] == 2
        assert compaction["retained_count"] == 0
        assert len(summarized_with) == 1
        assert "reply 1" in str(summarized_with[0][1])

    def test_default_still_skips_a_short_chat(self, client, adapter, admin_token, history_dir):
        user_id = "summarize-default-user"
        old_id = self._seed_two_turn_session(adapter, user_id)
        summarized_with = []

        resp = self._compact(client, admin_token, {"user_id": user_id, "session_id": old_id}, summarized_with)

        assert resp.status_code == 200, resp.text
        assert resp.json()["new_conversation_id"] == old_id
        assert summarized_with == []

    def test_chat_without_user_input_is_not_rotated(self, client, adapter, admin_token, history_dir):
        from tests.history_helpers import seed_history_session

        user_id = "summarize-empty-user"
        old_id = _seed_session_with_message(adapter, user_id)
        seed_history_session(old_id, events=[("model_response", {"raw_content": "hello"})])
        summarized_with = []

        resp = self._compact(
            client,
            admin_token,
            {"user_id": user_id, "session_id": old_id, "retain_recent": False},
            summarized_with,
        )

        assert resp.status_code == 409, resp.text
        assert adapter.session_store.get_session(old_id).superseded_by is None
        assert summarized_with == []

    @pytest.mark.parametrize("value", [None, 0, "false"])
    def test_non_boolean_retain_recent_is_rejected(self, client, adapter, admin_token, history_dir, value):
        user_id = "summarize-bad-flag-user"
        old_id = self._seed_two_turn_session(adapter, user_id)
        summarized_with = []

        resp = self._compact(
            client, admin_token, {"user_id": user_id, "session_id": old_id, "retain_recent": value}, summarized_with
        )

        assert resp.status_code == 400, resp.text
        assert summarized_with == []


class TestCompactRouteRefusals:
    def test_superseded_session_is_refused(self, client, adapter, admin_token):
        sid = _seed_session_with_message(adapter, "superseded-user")
        adapter.session_store.update_session(sid, superseded_by="successor")

        with patch.object(adapter, "_compact_session", new=AsyncMock(return_value=None)) as compact:
            resp = _post_compact(client, admin_token, {"user_id": "superseded-user", "session_id": sid})

        assert resp.status_code == 409, resp.text
        compact.assert_not_called()

    def test_busy_session_is_refused(self, client, adapter, admin_token):
        sid = _seed_session_with_message(adapter, "busy-user")
        adapter.session_store.begin_turn(sid)

        with patch.object(adapter, "_compact_session", new=AsyncMock(return_value=None)) as compact:
            resp = _post_compact(client, admin_token, {"user_id": "busy-user", "session_id": sid})

        assert resp.status_code == 409, resp.text
        compact.assert_not_called()
