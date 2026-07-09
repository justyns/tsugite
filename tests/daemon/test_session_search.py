"""Session search must scan ALL sessions (not the recency window the list
endpoint serves), match topic/metadata text, and resolve superseded bodies to
their live successors - the sidebar search box must not return false 'no
results' for sessions that exist."""

from unittest.mock import patch

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite_daemon.auth import TokenStore
from tsugite_daemon.config import AgentConfig, HTTPConfig
from tsugite_daemon.session_store import Session, SessionSource, SessionStore


def _mk(store, sid, title="", last_active="", metadata=None, superseded_by=None):
    store.create_session(
        Session(
            id=sid,
            agent="test-agent",
            source=SessionSource.INTERACTIVE.value,
            user_id="u1",
            title=title,
            metadata=metadata or {},
        )
    )
    if last_active:
        store._sessions[sid].last_active = last_active
    if superseded_by:
        store._sessions[sid].superseded_by = superseded_by


class TestSearchSessions:
    @pytest.fixture
    def store(self, tmp_path):
        return SessionStore(tmp_path / "session_store.json")

    def test_finds_title_outside_recency_window(self, store):
        for i in range(5):
            _mk(store, f"s-recent-{i}", title=f"recent {i}", last_active=f"2026-07-0{i + 3}T00:00:00+00:00")
        _mk(store, "s-old", title="Incremental Game Design", last_active="2026-06-01T00:00:00+00:00")

        window = store.list_sessions(agent="test-agent", limit=3)
        assert all(s.id != "s-old" for s in window), "precondition: target is outside the recency window"

        hits = store.search_sessions("test-agent", "design")
        assert [s.id for s in hits] == ["s-old"]

    def test_matches_topic_metadata(self, store):
        _mk(store, "s-topic", title="Untitled", metadata={"topic": "Idle infra-provider game design"})
        hits = store.search_sessions("test-agent", "infra-provider")
        assert [s.id for s in hits] == ["s-topic"]

    def test_superseded_hit_resolves_to_live_head(self, store):
        _mk(store, "s-head", title="continuation", last_active="2026-07-05T00:00:00+00:00")
        _mk(store, "s-body", title="Incremental Game Design", superseded_by="s-head")

        hits = store.search_sessions("test-agent", "incremental")
        assert [s.id for s in hits] == ["s-head"], "a compacted body must surface its live successor"

    def test_head_and_body_hits_dedupe(self, store):
        _mk(store, "s-head2", title="Game Design continued", last_active="2026-07-05T00:00:00+00:00")
        _mk(store, "s-body2", title="Game Design original", superseded_by="s-head2")
        hits = store.search_sessions("test-agent", "game design")
        assert [s.id for s in hits] == ["s-head2"]

    def test_empty_query_returns_nothing(self, store):
        _mk(store, "s-x", title="anything")
        assert store.search_sessions("test-agent", "  ") == []


class TestSearchEndpoint:
    @pytest.fixture
    def client_and_token(self, tmp_path):
        from tsugite.workspace import WorkspaceNotFoundError

        workspace = tmp_path / "ws"
        workspace.mkdir()
        store = SessionStore(tmp_path / "session_store.json")
        config = AgentConfig(workspace_dir=workspace, agent_file="default")
        with patch("tsugite.workspace.Workspace") as mock_ws:
            mock_ws.load.side_effect = WorkspaceNotFoundError("nope")
            with patch("tsugite.workspace.context.build_workspace_attachments", return_value=[]):
                adapter = HTTPAgentAdapter(agent_name="test-agent", agent_config=config, session_store=store)
        token_store = TokenStore(tmp_path / "tokens.json")
        _t, raw = token_store.create_admin_token(name="t")
        server = HTTPServer(
            config=HTTPConfig(enabled=True, host="127.0.0.1", port=8374),
            adapters={"test-agent": adapter},
            webhook_store=None,
            agent_configs={"test-agent": config},
            token_store=token_store,
        )
        return TestClient(server.app), raw, store

    def test_q_param_searches_past_the_window(self, client_and_token):
        client, token, store = client_and_token
        for i in range(5):
            _mk(store, f"api-recent-{i}", title=f"recent {i}", last_active=f"2026-07-0{i + 3}T00:00:00+00:00")
        _mk(store, "api-old", title="Incremental Game Design", last_active="2026-06-01T00:00:00+00:00")

        resp = client.get(
            "/api/agents/test-agent/sessions?q=design&limit=3",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        ids = [r["id"] for r in resp.json()["sessions"]]
        assert ids == ["api-old"], f"?q= must search all sessions, got {ids}"
