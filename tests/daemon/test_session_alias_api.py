"""HTTP API for claiming and releasing a session's alias."""

from unittest.mock import patch

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite_daemon.config import HTTPConfig, RuntimeDefaults
from tsugite_daemon.session_runner import SessionRunner
from tsugite_daemon.session_store import Session, SessionSource, SessionStore
from tsugite_daemon.webhook_store import WebhookStore


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
    return SessionRunner(store=session_store, adapter=None)


@pytest.fixture
def server(tmp_workspace, session_store, session_runner, tmp_path):
    runtime = RuntimeDefaults(workspace_dir=tmp_workspace, agent_file="default")
    http_config = HTTPConfig(enabled=True, host="127.0.0.1", port=8378)
    webhook_store = WebhookStore(tmp_path / "webhooks.json")

    from tsugite_daemon.auth import TokenStore

    token_store = TokenStore(tmp_path / "tokens.json")

    with patch("tsugite.workspace.Workspace") as mock_ws_cls:
        from tsugite.workspace import WorkspaceNotFoundError

        mock_ws_cls.load.side_effect = WorkspaceNotFoundError("not found")
        adapter = HTTPAgentAdapter(runtime=runtime, session_store=session_store)

    srv = HTTPServer(
        config=http_config,
        adapter=adapter,
        webhook_store=webhook_store,
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


def _create_session(store: SessionStore, sid: str, user_id: str = "web-anonymous") -> str:
    store.create_session(Session(id=sid, source=SessionSource.INTERACTIVE.value, user_id=user_id))
    return sid


def auth(token):
    return {"Authorization": f"Bearer {token}"}


class TestClaimAlias:
    def test_claiming_an_alias_makes_the_session_findable_by_name(self, client, test_token, session_store):
        sid = _create_session(session_store, "s-1")

        resp = client.put(f"/api/sessions/{sid}/alias", json={"alias": "daily"}, headers=auth(test_token))

        assert resp.status_code == 200
        assert resp.json()["alias"] == "daily"
        assert session_store.find_named_session("daily").id == sid

    def test_a_taken_alias_is_a_conflict(self, client, test_token, session_store):
        held = _create_session(session_store, "s-1")
        session_store.set_alias(held, "daily")
        other = _create_session(session_store, "s-2")

        resp = client.put(f"/api/sessions/{other}/alias", json={"alias": "daily"}, headers=auth(test_token))

        assert resp.status_code == 409
        assert session_store.find_named_session("daily").id == held

    def test_reclaiming_its_own_alias_succeeds(self, client, test_token, session_store):
        sid = _create_session(session_store, "s-1")
        session_store.set_alias(sid, "daily")

        resp = client.put(f"/api/sessions/{sid}/alias", json={"alias": "daily"}, headers=auth(test_token))

        assert resp.status_code == 200

    def test_renaming_releases_the_old_alias(self, client, test_token, session_store):
        sid = _create_session(session_store, "s-1")
        session_store.set_alias(sid, "daily")

        resp = client.put(f"/api/sessions/{sid}/alias", json={"alias": "weekly"}, headers=auth(test_token))

        assert resp.status_code == 200
        assert session_store.find_named_session("daily") is None
        assert session_store.find_named_session("weekly").id == sid

    @pytest.mark.parametrize("bad", ["", "has space", "-leading", "a" * 65])
    def test_a_malformed_alias_is_rejected(self, client, test_token, session_store, bad):
        sid = _create_session(session_store, "s-1")

        resp = client.put(f"/api/sessions/{sid}/alias", json={"alias": bad}, headers=auth(test_token))

        assert resp.status_code == 400

    def test_a_non_string_alias_is_rejected(self, client, test_token, session_store):
        sid = _create_session(session_store, "s-1")

        resp = client.put(f"/api/sessions/{sid}/alias", json={"alias": 7}, headers=auth(test_token))

        assert resp.status_code == 400

    def test_an_unknown_session_is_not_found(self, client, test_token):
        resp = client.put("/api/sessions/nope/alias", json={"alias": "daily"}, headers=auth(test_token))

        assert resp.status_code == 404

    def test_claiming_requires_auth(self, client, session_store):
        sid = _create_session(session_store, "s-1")

        resp = client.put(f"/api/sessions/{sid}/alias", json={"alias": "daily"})

        assert resp.status_code == 401


class TestReleaseAlias:
    def test_releasing_frees_the_alias(self, client, test_token, session_store):
        sid = _create_session(session_store, "s-1")
        session_store.set_alias(sid, "daily")

        resp = client.delete(f"/api/sessions/{sid}/alias", headers=auth(test_token))

        assert resp.status_code == 200
        assert resp.json()["alias"] is None
        assert session_store.find_named_session("daily") is None

    def test_releasing_an_alias_the_session_never_held_succeeds(self, client, test_token, session_store):
        sid = _create_session(session_store, "s-1")

        resp = client.delete(f"/api/sessions/{sid}/alias", headers=auth(test_token))

        assert resp.status_code == 200

    def test_releasing_an_unknown_session_is_not_found(self, client, test_token):
        resp = client.delete("/api/sessions/nope/alias", headers=auth(test_token))

        assert resp.status_code == 404

    def test_releasing_requires_auth(self, client, session_store):
        sid = _create_session(session_store, "s-1")

        resp = client.delete(f"/api/sessions/{sid}/alias")

        assert resp.status_code == 401


class TestAliasInPayloads:
    def test_the_session_list_carries_the_alias(self, client, test_token, session_store):
        sid = _create_session(session_store, "s-1")
        session_store.set_alias(sid, "daily")

        rows = client.get("/api/sessions/", headers=auth(test_token)).json()["sessions"]

        assert {r["id"]: r["alias"] for r in rows} == {sid: "daily"}

    def test_the_session_detail_carries_the_alias(self, client, test_token, session_store):
        sid = _create_session(session_store, "s-1")
        session_store.set_alias(sid, "daily")

        detail = client.get(f"/api/sessions/{sid}", headers=auth(test_token)).json()

        assert detail["alias"] == "daily"

    def test_a_session_with_no_alias_reports_none(self, client, test_token, session_store):
        sid = _create_session(session_store, "s-1")

        detail = client.get(f"/api/sessions/{sid}", headers=auth(test_token)).json()

        assert detail["alias"] is None

    def test_the_sidebar_list_carries_the_alias(self, client, test_token, session_store):
        sid = _create_session(session_store, "s-1")
        session_store.set_alias(sid, "daily")

        rows = client.get("/api/chat/sessions", headers=auth(test_token)).json()["sessions"]

        assert {r["id"]: r["alias"] for r in rows} == {sid: "daily"}
