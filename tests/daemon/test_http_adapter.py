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
        assert data["turns"] == []
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

    def test_history_includes_reactions(self, client, test_token, mock_adapter, tmp_path):
        """Reactions from session event log should appear in history turns."""
        from tsugite.history.models import Turn
        from tsugite.history.storage import SessionStorage

        session = mock_adapter.session_store.get_or_create_interactive("web-anonymous", "test-agent")
        session_id = session.id
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        session_path = history_dir / f"{session_id}.jsonl"

        storage = SessionStorage.create("test-agent", model="test", session_path=session_path)
        storage.record_turn(
            messages=[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
            final_answer="hi",
        )

        turn = next(r for r in storage.load_records() if isinstance(r, Turn))
        reaction_dt = turn.timestamp - timedelta(milliseconds=500)
        reaction_ts = reaction_dt.isoformat()

        mock_adapter.session_store.append_event(
            session_id,
            {
                "type": "reaction",
                "emoji": "👍",
                "message_id": None,
                "timestamp": reaction_ts,
            },
        )

        # The session_id from create() differs from session.id, so we need to
        # rename the file to match the session_id the adapter expects
        session_path.rename(history_dir / f"{session_id}.jsonl")

        with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
            resp = client.get(
                "/api/agents/test-agent/history?user_id=web-anonymous",
                headers={"Authorization": f"Bearer {test_token}"},
            )

        assert resp.status_code == 200
        data = resp.json()
        turn_data = [t for t in data["turns"] if t.get("user")]
        assert len(turn_data) == 1
        assert turn_data[0].get("reactions") == ["👍"]

    def test_history_no_reactions_when_none(self, client, test_token, mock_adapter, tmp_path):
        """Turns without reactions should not have a reactions field."""
        from tsugite.history.storage import SessionStorage

        session = mock_adapter.session_store.get_or_create_interactive("web-anonymous", "test-agent")
        session_id = session.id
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        session_path = history_dir / f"{session_id}.jsonl"

        storage = SessionStorage.create("test-agent", model="test", session_path=session_path)
        storage.record_turn(
            messages=[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
            final_answer="hi",
        )
        session_path.rename(history_dir / f"{session_id}.jsonl")

        with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
            resp = client.get(
                "/api/agents/test-agent/history?user_id=web-anonymous",
                headers={"Authorization": f"Bearer {test_token}"},
            )

        assert resp.status_code == 200
        data = resp.json()
        turn_data = [t for t in data["turns"] if t.get("user")]
        assert len(turn_data) == 1
        assert "reactions" not in turn_data[0]

    def test_history_detail_attaches_content_blocks_per_message(self, client, test_token, mock_adapter, tmp_path):
        """When detail=true, each assistant message carries its own content_blocks dict."""
        from tsugite.history.storage import SessionStorage

        session = mock_adapter.session_store.get_or_create_interactive("web-anonymous", "test-agent")
        session_id = session.id
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        session_path = history_dir / f"{session_id}.jsonl"

        storage = SessionStorage.create("test-agent", model="test", session_path=session_path)
        storage.record_turn(
            messages=[
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "```python\ninspect()\n```"},
                {
                    "role": "user",
                    "content": '<tsugite_execution_result status="success" duration_ms="3"><output>ok</output></tsugite_execution_result>',
                },
                {
                    "role": "assistant",
                    "content": (
                        '```python\nfinal_answer(result="done")\n```\n\n'
                        '<content name="reply">Stopped before creating anything.</content>'
                    ),
                },
            ],
            final_answer="done",
        )
        session_path.rename(history_dir / f"{session_id}.jsonl")

        with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
            resp = client.get(
                "/api/agents/test-agent/history?user_id=web-anonymous&detail=true",
                headers={"Authorization": f"Bearer {test_token}"},
            )

        assert resp.status_code == 200
        data = resp.json()
        turn = next(t for t in data["turns"] if t.get("user"))

        # Turn-level aggregate still available for backward compat
        assert turn.get("content_blocks") == {"reply": "Stopped before creating anything."}

        # Per-message attachment is the new source of truth for rendering
        assistant_msgs = [m for m in turn["messages"] if m.get("role") == "assistant"]
        assert assistant_msgs[0].get("content_blocks", {}) == {}
        assert assistant_msgs[1]["content_blocks"] == {"reply": "Stopped before creating anything."}

    def test_history_detail_preserves_execution_result_with_attributes(
        self, client, test_token, mock_adapter, tmp_path
    ):
        """Execution-result tags with attributes must survive the payload unmodified."""
        from tsugite.history.storage import SessionStorage

        session = mock_adapter.session_store.get_or_create_interactive("web-anonymous", "test-agent")
        session_id = session.id
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        session_path = history_dir / f"{session_id}.jsonl"

        observation = (
            '<tsugite_execution_result status="success" duration_ms="12">'
            "<output>ran</output></tsugite_execution_result>"
        )
        storage = SessionStorage.create("test-agent", model="test", session_path=session_path)
        storage.record_turn(
            messages=[
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "```python\nprint('hi')\n```"},
                {"role": "user", "content": observation},
            ],
            final_answer="done",
        )
        session_path.rename(history_dir / f"{session_id}.jsonl")

        with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
            resp = client.get(
                "/api/agents/test-agent/history?user_id=web-anonymous&detail=true",
                headers={"Authorization": f"Bearer {test_token}"},
            )

        assert resp.status_code == 200
        turn = next(t for t in resp.json()["turns"] if t.get("user"))
        user_msgs = [m for m in turn["messages"] if m.get("role") == "user"]
        assert any(m.get("content") == observation for m in user_msgs)


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
