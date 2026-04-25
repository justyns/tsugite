"""HTTP API tests for session pinning, viewing, and supersession (#217, #218, #219)."""

from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from tsugite.daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite.daemon.config import AgentConfig, HTTPConfig
from tsugite.daemon.session_runner import SessionRunner
from tsugite.daemon.session_store import Session, SessionSource, SessionStore
from tsugite.daemon.webhook_store import WebhookStore


@pytest.fixture
def tmp_workspace(tmp_path):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    return workspace_dir


@pytest.fixture
def session_store(tmp_path):
    return SessionStore(tmp_path / "session_store.json")


@pytest.fixture
def session_runner(session_store):
    return SessionRunner(store=session_store, adapters={})


@pytest.fixture
def server(tmp_workspace, session_store, session_runner, tmp_path):
    agent_config = AgentConfig(workspace_dir=tmp_workspace, agent_file="default")
    http_config = HTTPConfig(enabled=True, host="127.0.0.1", port=8374)
    webhook_store = WebhookStore(tmp_path / "webhooks.json")

    from tsugite.daemon.auth import TokenStore

    token_store = TokenStore(tmp_path / "tokens.json")

    with patch("tsugite.workspace.Workspace") as mock_ws_cls:
        from tsugite.workspace import WorkspaceNotFoundError

        mock_ws_cls.load.side_effect = WorkspaceNotFoundError("not found")
        with patch("tsugite.workspace.context.build_workspace_attachments", return_value=[]):
            adapter = HTTPAgentAdapter(
                agent_name="test-agent",
                agent_config=agent_config,
                session_store=session_store,
            )

    srv = HTTPServer(
        config=http_config,
        adapters={"test-agent": adapter},
        webhook_store=webhook_store,
        agent_configs={"test-agent": agent_config},
        token_store=token_store,
    )
    srv.session_runner = session_runner
    return srv


@pytest.fixture
def test_token(server):
    _st, raw = server._token_store.create_admin_token(name="test-token")
    return raw


@pytest.fixture
def client(server):
    return TestClient(server.app)


def _create_session(store: SessionStore, *, title: str | None = None, suffix: str = "") -> str:
    sid = f"s-{title or 'anon'}-{suffix}"
    s = Session(
        id=sid,
        agent="test-agent",
        source=SessionSource.INTERACTIVE.value,
        user_id="web-anonymous",
        title=title,
    )
    store.create_session(s)
    return sid


def auth(token):
    return {"Authorization": f"Bearer {token}"}


class TestPatchSessionPinFields:
    def test_patch_pinned_true(self, client, test_token, session_store):
        sid = _create_session(session_store, title="t", suffix="1")
        resp = client.patch(f"/api/sessions/{sid}", json={"pinned": True}, headers=auth(test_token))
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["pinned"] is True
        assert body["pin_position"] == 0
        assert session_store.get_session(sid).pinned is True

    def test_patch_pinned_false(self, client, test_token, session_store):
        sid = _create_session(session_store, title="t", suffix="1")
        session_store.set_pin(sid, True)
        resp = client.patch(f"/api/sessions/{sid}", json={"pinned": False}, headers=auth(test_token))
        assert resp.status_code == 200
        assert session_store.get_session(sid).pinned is False

    def test_patch_last_viewed_at(self, client, test_token, session_store):
        sid = _create_session(session_store, title="t", suffix="1")
        ts = "2026-04-25T12:00:00+00:00"
        resp = client.patch(f"/api/sessions/{sid}", json={"last_viewed_at": ts}, headers=auth(test_token))
        assert resp.status_code == 200
        assert resp.json()["last_viewed_at"] == ts

    def test_patch_no_fields_400(self, client, test_token, session_store):
        sid = _create_session(session_store, title="t", suffix="1")
        resp = client.patch(f"/api/sessions/{sid}", json={}, headers=auth(test_token))
        assert resp.status_code == 400


class TestPinEndpoint:
    def test_pin_appends(self, client, test_token, session_store):
        a = _create_session(session_store, title="a", suffix="1")
        b = _create_session(session_store, title="b", suffix="2")
        client.post(f"/api/sessions/{a}/pin", json={}, headers=auth(test_token))
        resp = client.post(f"/api/sessions/{b}/pin", json={}, headers=auth(test_token))
        assert resp.status_code == 200
        body = resp.json()
        assert body["pinned"] is True
        assert body["pin_position"] == 1

    def test_pin_explicit_position(self, client, test_token, session_store):
        a = _create_session(session_store, title="a", suffix="1")
        b = _create_session(session_store, title="b", suffix="2")
        client.post(f"/api/sessions/{a}/pin", json={}, headers=auth(test_token))
        resp = client.post(f"/api/sessions/{b}/pin", json={"position": 0}, headers=auth(test_token))
        assert resp.status_code == 200
        assert resp.json()["pin_position"] == 0
        assert session_store.get_session(a).pin_position == 1

    def test_pin_unknown_session_404(self, client, test_token):
        resp = client.post("/api/sessions/missing/pin", json={}, headers=auth(test_token))
        assert resp.status_code == 404


class TestUnpinEndpoint:
    def test_unpin(self, client, test_token, session_store):
        a = _create_session(session_store, title="a", suffix="1")
        session_store.set_pin(a, True)
        resp = client.post(f"/api/sessions/{a}/unpin", headers=auth(test_token))
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert session_store.get_session(a).pinned is False

    def test_unpin_unknown_404(self, client, test_token):
        resp = client.post("/api/sessions/missing/unpin", headers=auth(test_token))
        assert resp.status_code == 404


class TestReorderEndpoint:
    def test_reorder(self, client, test_token, session_store):
        a = _create_session(session_store, title="a", suffix="1")
        b = _create_session(session_store, title="b", suffix="2")
        c = _create_session(session_store, title="c", suffix="3")
        for sid in (a, b, c):
            session_store.set_pin(sid, True)

        resp = client.post(
            "/api/sessions/pinned/reorder",
            json={"ids": [c, a, b]},
            headers=auth(test_token),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["ordered"] == [c, a, b]
        assert session_store.get_session(c).pin_position == 0
        assert session_store.get_session(a).pin_position == 1
        assert session_store.get_session(b).pin_position == 2

    def test_reorder_invalid_body(self, client, test_token):
        resp = client.post(
            "/api/sessions/pinned/reorder",
            json={"ids": "not-a-list"},
            headers=auth(test_token),
        )
        assert resp.status_code == 400


class TestMarkViewedEndpoint:
    def test_mark_viewed(self, client, test_token, session_store):
        sid = _create_session(session_store, title="t", suffix="1")
        resp = client.post(f"/api/sessions/{sid}/mark-viewed", json={}, headers=auth(test_token))
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert session_store.get_session(sid).last_viewed_at != ""

    def test_mark_viewed_explicit_ts(self, client, test_token, session_store):
        sid = _create_session(session_store, title="t", suffix="1")
        ts = "2026-04-25T01:02:03+00:00"
        resp = client.post(
            f"/api/sessions/{sid}/mark-viewed",
            json={"ts": ts},
            headers=auth(test_token),
        )
        assert resp.status_code == 200
        assert session_store.get_session(sid).last_viewed_at == ts

    def test_mark_viewed_unknown_404(self, client, test_token):
        resp = client.post("/api/sessions/missing/mark-viewed", json={}, headers=auth(test_token))
        assert resp.status_code == 404


class TestNewSessionAcceptsTitle:
    def test_new_session_with_title(self, client, test_token, session_store):
        resp = client.post(
            "/api/agents/test-agent/sessions/new",
            json={"user_id": "web-anonymous", "title": "Reading"},
            headers=auth(test_token),
        )
        assert resp.status_code == 201
        sid = resp.json()["id"]
        assert session_store.get_session(sid).title == "Reading"

    def test_new_session_without_title_falls_back(self, client, test_token, session_store):
        resp = client.post(
            "/api/agents/test-agent/sessions/new",
            json={"user_id": "web-anonymous"},
            headers=auth(test_token),
        )
        assert resp.status_code == 201
        sid = resp.json()["id"]
        assert session_store.get_session(sid).title is None


class TestListSessionsExposesNewFields:
    def test_response_includes_pinned_and_unread_fields(self, client, test_token, session_store):
        sid = _create_session(session_store, title="t", suffix="1")
        session_store.set_pin(sid, True)
        resp = client.get("/api/agents/test-agent/sessions", headers=auth(test_token))
        assert resp.status_code == 200
        rows = resp.json()["sessions"]
        match = next(r for r in rows if r["id"] == sid)
        assert match["pinned"] is True
        assert match["pin_position"] == 0
        assert "last_viewed_at" in match
        assert "unread" in match
        assert "superseded_by" in match

    def test_unread_true_when_no_view_recorded(self, client, test_token, session_store):
        sid = _create_session(session_store, title="t", suffix="1")
        resp = client.get("/api/agents/test-agent/sessions", headers=auth(test_token))
        match = next(r for r in resp.json()["sessions"] if r["id"] == sid)
        assert match["unread"] is True

    def test_unread_false_after_mark_viewed(self, client, test_token, session_store):
        sid = _create_session(session_store, title="t", suffix="1")
        # mark_viewed BEFORE last_active changes; since no further mutations happen,
        # last_active <= last_viewed_at and unread should be false.
        session_store.mark_viewed(sid, ts="2099-01-01T00:00:00+00:00")
        resp = client.get("/api/agents/test-agent/sessions", headers=auth(test_token))
        match = next(r for r in resp.json()["sessions"] if r["id"] == sid)
        assert match["unread"] is False

    def test_superseded_filtered_by_default(self, client, test_token, session_store):
        sid = _create_session(session_store, title="t", suffix="1")
        new = session_store.compact_session(sid)
        resp = client.get("/api/agents/test-agent/sessions", headers=auth(test_token))
        ids = [r["id"] for r in resp.json()["sessions"]]
        assert new.id in ids
        assert sid not in ids

    def test_superseded_included_when_requested(self, client, test_token, session_store):
        sid = _create_session(session_store, title="t", suffix="1")
        new = session_store.compact_session(sid)
        resp = client.get(
            "/api/agents/test-agent/sessions?include_superseded=true",
            headers=auth(test_token),
        )
        ids = [r["id"] for r in resp.json()["sessions"]]
        assert new.id in ids
        assert sid in ids
