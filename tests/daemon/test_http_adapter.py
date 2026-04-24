"""Tests for the HTTP API adapter."""

import json
from datetime import timedelta
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from tsugite.daemon.adapters.http import HTTPAgentAdapter, HTTPServer, SSEProgressHandler
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
def http_config():
    return HTTPConfig(enabled=True, host="127.0.0.1", port=8374)


@pytest.fixture
def mock_adapter(agent_config, tmp_path):
    from tsugite.daemon.session_store import SessionStore
    from tsugite.workspace import WorkspaceNotFoundError

    session_store = SessionStore(tmp_path / "session_store.json")

    with patch("tsugite.workspace.Workspace") as mock_ws_cls:
        mock_ws_cls.load.side_effect = WorkspaceNotFoundError("not found")
        with patch("tsugite.workspace.context.build_workspace_attachments", return_value=[]):
            adapter = HTTPAgentAdapter(
                agent_name="test-agent",
                agent_config=agent_config,
                session_store=session_store,
            )
            return adapter


@pytest.fixture
def webhook_store(tmp_path):
    store = WebhookStore(tmp_path / "webhooks.json")
    store.add(agent="test-agent", source="forgejo", token="whk_test")
    return store


@pytest.fixture
def token_store(tmp_path):
    from tsugite.daemon.auth import TokenStore

    store = TokenStore(tmp_path / "tokens.json")
    store.create_admin_token(name="test-token")
    return store


@pytest.fixture
def test_token(token_store):
    """Return a valid raw admin token for use in test requests."""
    _st, raw = token_store.create_admin_token(name="test-request-token")
    return raw


@pytest.fixture
def server(http_config, mock_adapter, webhook_store, agent_config, token_store):
    return HTTPServer(
        config=http_config,
        adapters={"test-agent": mock_adapter},
        webhook_store=webhook_store,
        agent_configs={"test-agent": agent_config},
        token_store=token_store,
    )


@pytest.fixture
def client(server):
    return TestClient(server.app)


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
    def test_list_agents_with_auth(self, client, test_token):
        resp = client.get("/api/agents", headers={"Authorization": f"Bearer {test_token}"})
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


class TestChatEndpoint:
    def test_chat_unknown_agent(self, client, test_token):
        resp = client.post(
            "/api/agents/nonexistent/chat",
            json={"message": "hello"},
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert resp.status_code == 404

    def test_chat_no_auth(self, client):
        resp = client.post("/api/agents/test-agent/chat", json={"message": "hello"})
        assert resp.status_code == 401

    def test_chat_empty_message(self, client, test_token):
        resp = client.post(
            "/api/agents/test-agent/chat",
            json={"message": ""},
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert resp.status_code == 400

    def test_chat_invalid_json(self, client, test_token):
        resp = client.post(
            "/api/agents/test-agent/chat",
            content=b"not json",
            headers={"Authorization": f"Bearer {test_token}", "Content-Type": "application/json"},
        )
        assert resp.status_code == 400


class TestEffortLevelsEndpoint:
    """GET /api/agents/{agent}/effort-levels returns the resolved model's effort list."""

    def test_unknown_agent(self, client, test_token):
        resp = client.get(
            "/api/agents/nope/effort-levels",
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert resp.status_code == 404

    def test_returns_levels_for_claude_code_model(self, client, mock_adapter, test_token):
        mock_adapter.agent_config.model = "claude_code:opus"
        resp = client.get(
            "/api/agents/test-agent/effort-levels",
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["supported_effort_levels"] == ["low", "medium", "high", "xhigh", "max"]

    def test_returns_null_for_non_reasoning_model(self, client, mock_adapter, test_token):
        mock_adapter.agent_config.model = "anthropic:claude-3-opus-20240229"
        resp = client.get(
            "/api/agents/test-agent/effort-levels",
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["supported_effort_levels"] is None


class TestSessionSettingsEndpoint:
    """GET/PATCH /api/sessions/{session_id}/settings round-trips reasoning_effort."""

    def test_get_unknown_session(self, client, test_token):
        resp = client.get(
            "/api/sessions/does-not-exist/settings",
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert resp.status_code == 404

    def test_patch_then_get(self, client, mock_adapter, test_token):
        mock_adapter.agent_config.model = "claude_code:opus"
        session = mock_adapter.session_store.get_or_create_interactive("user-a", "test-agent")

        resp = client.patch(
            f"/api/sessions/{session.id}/settings",
            json={"reasoning_effort": "high"},
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["reasoning_effort"] == "high"

        resp = client.get(
            f"/api/sessions/{session.id}/settings",
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["reasoning_effort"] == "high"

    def test_patch_invalid_value_returns_400(self, client, mock_adapter, test_token):
        mock_adapter.agent_config.model = "claude_code:opus"
        session = mock_adapter.session_store.get_or_create_interactive("user-b", "test-agent")

        resp = client.patch(
            f"/api/sessions/{session.id}/settings",
            json={"reasoning_effort": "bogus"},
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert "supported" in body
        assert "xhigh" in body["supported"]

    def test_patch_null_clears_value(self, client, mock_adapter, test_token):
        mock_adapter.agent_config.model = "claude_code:opus"
        session = mock_adapter.session_store.get_or_create_interactive("user-c", "test-agent")
        mock_adapter.session_store.set_reasoning_effort(session.id, "max")

        resp = client.patch(
            f"/api/sessions/{session.id}/settings",
            json={"reasoning_effort": None},
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["reasoning_effort"] is None


class TestChatReasoningEffortOverride:
    """POST /api/agents/{agent}/chat accepts and validates reasoning_effort."""

    def test_invalid_value_returns_400(self, client, mock_adapter, test_token):
        mock_adapter.agent_config.model = "claude_code:opus"
        resp = client.post(
            "/api/agents/test-agent/chat",
            json={"message": "hi", "reasoning_effort": "ultra"},
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert resp.status_code == 400
        assert "supported" in resp.json()


class TestUnloadSkillEndpoint:
    """POST /api/agents/{agent}/unload-skill suppresses a skill for the session."""

    def _headers(self, token):
        return {"Authorization": f"Bearer {token}"}

    def test_unload_skill_happy_path(self, client, test_token, mock_adapter):
        resp = client.post(
            "/api/agents/test-agent/unload-skill",
            json={"user_id": "alice", "name": "skill-a"},
            headers=self._headers(test_token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["name"] == "skill-a"
        # Resolves the interactive session and stores the suppression there.
        session = mock_adapter.session_store.get_or_create_interactive("alice", "test-agent")
        assert data["session_id"] == session.id
        assert mock_adapter.session_store.get_suppressed_skills(session.id) == {"skill-a"}

    def test_unload_skill_is_idempotent(self, client, test_token, mock_adapter):
        for _ in range(3):
            resp = client.post(
                "/api/agents/test-agent/unload-skill",
                json={"user_id": "alice", "name": "skill-a"},
                headers=self._headers(test_token),
            )
            assert resp.status_code == 200
        session = mock_adapter.session_store.get_or_create_interactive("alice", "test-agent")
        assert mock_adapter.session_store.get_suppressed_skills(session.id) == {"skill-a"}

    def test_unload_multiple_skills(self, client, test_token, mock_adapter):
        for name in ("skill-a", "skill-b", "skill-c"):
            resp = client.post(
                "/api/agents/test-agent/unload-skill",
                json={"user_id": "alice", "name": name},
                headers=self._headers(test_token),
            )
            assert resp.status_code == 200
        session = mock_adapter.session_store.get_or_create_interactive("alice", "test-agent")
        assert mock_adapter.session_store.get_suppressed_skills(session.id) == {"skill-a", "skill-b", "skill-c"}

    def test_unload_skill_isolated_per_user(self, client, test_token, mock_adapter):
        client.post(
            "/api/agents/test-agent/unload-skill",
            json={"user_id": "alice", "name": "skill-a"},
            headers=self._headers(test_token),
        )
        alice_session = mock_adapter.session_store.get_or_create_interactive("alice", "test-agent")
        bob_session = mock_adapter.session_store.get_or_create_interactive("bob", "test-agent")
        assert alice_session.id != bob_session.id
        assert mock_adapter.session_store.get_suppressed_skills(bob_session.id) == set()

    def test_unload_skill_missing_name(self, client, test_token):
        resp = client.post(
            "/api/agents/test-agent/unload-skill",
            json={"user_id": "alice"},
            headers=self._headers(test_token),
        )
        assert resp.status_code == 400
        assert "name" in resp.json()["error"]

    def test_unload_skill_empty_name(self, client, test_token):
        resp = client.post(
            "/api/agents/test-agent/unload-skill",
            json={"user_id": "alice", "name": ""},
            headers=self._headers(test_token),
        )
        assert resp.status_code == 400

    def test_unload_skill_non_string_name(self, client, test_token):
        resp = client.post(
            "/api/agents/test-agent/unload-skill",
            json={"user_id": "alice", "name": 42},
            headers=self._headers(test_token),
        )
        assert resp.status_code == 400

    def test_unload_skill_invalid_json(self, client, test_token):
        resp = client.post(
            "/api/agents/test-agent/unload-skill",
            content=b"not json",
            headers={**self._headers(test_token), "Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    def test_unload_skill_unknown_agent(self, client, test_token):
        resp = client.post(
            "/api/agents/nonexistent/unload-skill",
            json={"user_id": "alice", "name": "skill-a"},
            headers=self._headers(test_token),
        )
        assert resp.status_code == 404

    def test_unload_skill_no_auth(self, client):
        resp = client.post(
            "/api/agents/test-agent/unload-skill",
            json={"user_id": "alice", "name": "skill-a"},
        )
        assert resp.status_code == 401

    def test_unload_skill_invokes_manager(self, client, test_token):
        """The handler also tells the global SkillManager to drop the skill so any
        in-flight read of its state reflects the removal immediately."""
        with patch("tsugite.tools.skills.get_skill_manager") as get_mgr:
            client.post(
                "/api/agents/test-agent/unload-skill",
                json={"user_id": "alice", "name": "skill-a"},
                headers=self._headers(test_token),
            )
        get_mgr.return_value.unload_skill.assert_called_once_with("skill-a")


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

    def test_webhook_logs_receipt(self, client, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="tsugite.daemon.adapters.http"):
            client.post("/webhook/whk_test", json={"event": "push"})

        assert any("Received webhook [whk_test" in r.message for r in caplog.records)
        assert any("source: forgejo | event: push" in r.message for r in caplog.records)
        assert any("saved to inbox" in r.message for r in caplog.records)

    def test_webhook_logs_invalid_token(self, client, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="tsugite.daemon.adapters.http"):
            client.post("/webhook/bad-token", json={"data": 1})

        assert any("invalid token [bad-toke" in r.message for r in caplog.records)


class TestHistoryEndpoint:
    def test_history_empty_for_new_user(self, client, test_token):
        resp = client.get(
            "/api/agents/test-agent/history?user_id=brand-new-user",
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["events"] == []
        assert "conversation_id" in data

    def test_history_unknown_agent(self, client, test_token):
        resp = client.get(
            "/api/agents/nonexistent/history?user_id=someone",
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert resp.status_code == 404

    def test_history_unauthorized(self, client):
        resp = client.get("/api/agents/test-agent/history?user_id=someone")
        assert resp.status_code == 401

    def test_history_returns_event_stream(self, client, test_token, mock_adapter, tmp_path):
        """The history endpoint returns the raw event log (session_start, user_input, model_response, ...)."""
        from tsugite.history.storage import SessionStorage

        session = mock_adapter.session_store.get_or_create_interactive("web-anonymous", "test-agent")
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        session_path = history_dir / f"{session.id}.jsonl"

        storage = SessionStorage.create("test-agent", model="test", session_path=session_path)
        storage.record("user_input", text="hello")
        storage.record("model_response", raw_content="hi", usage={"total_tokens": 5})
        storage.record("session_end", status="success")

        with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
            resp = client.get(
                "/api/agents/test-agent/history?user_id=web-anonymous",
                headers={"Authorization": f"Bearer {test_token}"},
            )

        assert resp.status_code == 200
        data = resp.json()
        types = [e["type"] for e in data["events"]]
        assert types == ["session_start", "user_input", "model_response", "session_end"]
        responses = [e for e in data["events"] if e["type"] == "model_response"]
        assert responses[0]["data"]["raw_content"] == "hi"

    def test_history_includes_reactions(self, client, test_token, mock_adapter, tmp_path):
        """Reactions from the live session event log appear as their own events in history."""
        from tsugite.history.storage import SessionStorage

        session = mock_adapter.session_store.get_or_create_interactive("web-anonymous", "test-agent")
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        session_path = history_dir / f"{session.id}.jsonl"

        storage = SessionStorage.create("test-agent", model="test", session_path=session_path)
        storage.record("user_input", text="hello")
        storage.record("model_response", raw_content="hi")

        mock_adapter.session_store.append_event(
            session.id,
            {"type": "reaction", "emoji": "👍", "timestamp": "2026-01-01T00:00:00+00:00"},
        )

        with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
            resp = client.get(
                "/api/agents/test-agent/history?user_id=web-anonymous",
                headers={"Authorization": f"Bearer {test_token}"},
            )

        assert resp.status_code == 200
        events = resp.json()["events"]
        reactions = [e for e in events if e["type"] == "reaction"]
        assert len(reactions) == 1
        assert reactions[0]["data"]["emoji"] == "👍"

    def test_history_preserves_execution_result(self, client, test_token, mock_adapter, tmp_path):
        """Code execution events round-trip through the endpoint with output, error, duration."""
        from tsugite.history.storage import SessionStorage

        session = mock_adapter.session_store.get_or_create_interactive("web-anonymous", "test-agent")
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        session_path = history_dir / f"{session.id}.jsonl"

        storage = SessionStorage.create("test-agent", model="test", session_path=session_path)
        storage.record("user_input", text="go")
        storage.record("model_response", raw_content="```python\nprint('hi')\n```")
        storage.record("code_execution", code="print('hi')", output="hi\n", duration_ms=12)
        storage.record("model_response", raw_content="Done.")

        with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
            resp = client.get(
                "/api/agents/test-agent/history?user_id=web-anonymous",
                headers={"Authorization": f"Bearer {test_token}"},
            )

        assert resp.status_code == 200
        events = resp.json()["events"]
        code_events = [e for e in events if e["type"] == "code_execution"]
        assert len(code_events) == 1
        assert code_events[0]["data"]["output"] == "hi\n"
        assert code_events[0]["data"]["duration_ms"] == 12


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

    def test_cross_thread_emit_then_signal_done_preserves_event(self):
        """H2: `_emit` from a worker thread schedules `queue.put_nowait` via
        `call_soon_threadsafe`. If `signal_done()` runs on the loop before the
        scheduled callback fires, the final event may never reach the drain
        loop.  Reproduce this by emitting from a thread and *immediately*
        signalling done on the loop, repeated many times.
        """
        import asyncio
        import threading

        async def _run() -> list[dict]:
            loop = asyncio.get_running_loop()
            missed: list[int] = []

            for i in range(200):
                handler = SSEProgressHandler()
                handler.set_loop(loop)

                emit_scheduled = threading.Event()

                def worker(idx: int = i):
                    handler._emit("final_result", {"result": f"answer-{idx}"})
                    emit_scheduled.set()

                thread = threading.Thread(target=worker)
                thread.start()
                # Wait for the thread to finish calling _emit — the
                # call_soon_threadsafe callback may still be pending on the
                # loop at this point.
                emit_scheduled.wait()
                thread.join()
                handler.signal_done()

                seen: list[dict] = []
                async for chunk in handler.event_generator():
                    if chunk.startswith("data: ") and "final_result" in chunk:
                        seen.append(chunk)

                if not seen:
                    missed.append(i)

            return missed

        missed = asyncio.run(_run())
        assert not missed, (
            f"SSE race dropped {len(missed)}/200 final_result events under "
            f"cross-thread emit + signal_done contention"
        )


class TestHTTPConfig:
    def test_config_defaults(self):
        config = HTTPConfig()
        assert config.enabled is False
        assert config.host == "127.0.0.1"
        assert config.port == 8374

    def test_webhook_entry(self):
        from tsugite.daemon.webhook_store import WebhookEntry

        wh = WebhookEntry(token="whk_test", agent="myagent", source="github")
        assert wh.token == "whk_test"
        assert wh.agent == "myagent"
        assert wh.source == "github"
