"""The session events endpoint serves incremental deltas (``after_id=``) and tail
windows (``limit=`` / ``before_id=``) so the chat surface never refetches a
multi-MB event log on every reconnect or session open. The bare call (no query
params) still returns the whole log for other consumers."""

from unittest.mock import patch

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite_daemon.auth import TokenStore
from tsugite_daemon.config import HTTPConfig, RuntimeDefaults
from tsugite_daemon.session_runner import SessionRunner
from tsugite_daemon.session_store import Session, SessionSource, SessionStore


@pytest.fixture
def adapter(tmp_path):
    from tsugite.workspace import WorkspaceNotFoundError

    workspace = tmp_path / "ws"
    workspace.mkdir()
    store = SessionStore(tmp_path / "session_store.json")
    config = RuntimeDefaults(workspace_dir=workspace, agent_file="default")
    with patch("tsugite.workspace.Workspace") as mock_ws:
        mock_ws.load.side_effect = WorkspaceNotFoundError("nope")
        return HTTPAgentAdapter(runtime=config, session_store=store)


@pytest.fixture
def client_and_token(adapter, tmp_path):
    token_store = TokenStore(tmp_path / "tokens.json")
    _t, raw = token_store.create_admin_token(name="t")
    server = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8375),
        adapter=adapter,
        webhook_store=None,
        token_store=token_store,
    )
    server.session_runner = SessionRunner(store=adapter.session_store, adapter=adapter)
    return TestClient(server.app), raw


def _seed(adapter, sid, n):
    adapter.session_store.create_session(Session(id=sid, source=SessionSource.INTERACTIVE.value, user_id="u1"))
    for i in range(n):
        adapter.session_store.append_event(sid, {"type": "info", "message": f"m{i}"})
    return sid


def _get(client, token, sid, query=""):
    resp = client.get(f"/api/sessions/{sid}/events{query}", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_bare_call_returns_all_events_and_no_windowing_fields(adapter, client_and_token):
    client, token = client_and_token
    sid = _seed(adapter, "s-all", 5)
    body = _get(client, token, sid)
    assert [e["message"] for e in body["events"]] == [f"m{i}" for i in range(5)]
    # Opt-in windowing only: the bare shape stays exactly {events: [...]}.
    assert set(body.keys()) == {"events"}


def test_after_id_returns_only_newer_events(adapter, client_and_token):
    client, token = client_and_token
    sid = _seed(adapter, "s-delta", 5)
    ids = [e["id"] for e in _get(client, token, sid)["events"]]
    cursor = ids[2]  # after the 3rd event
    body = _get(client, token, sid, f"?after_id={cursor}")
    assert [e["message"] for e in body["events"]] == ["m3", "m4"]
    assert all(e["id"] > cursor for e in body["events"])


def test_after_id_at_head_is_an_empty_delta(adapter, client_and_token):
    client, token = client_and_token
    sid = _seed(adapter, "s-head", 3)
    newest = _get(client, token, sid)["events"][-1]["id"]
    body = _get(client, token, sid, f"?after_id={newest}")
    assert body["events"] == []


def test_limit_returns_newest_window_with_has_more_and_oldest_id(adapter, client_and_token):
    client, token = client_and_token
    sid = _seed(adapter, "s-window", 6)
    body = _get(client, token, sid, "?limit=2")
    # Newest two, still chronological (newest-last).
    assert [e["message"] for e in body["events"]] == ["m4", "m5"]
    assert body["has_more"] is True
    assert body["oldest_id"] == body["events"][0]["id"]


def test_before_id_pages_the_chunk_immediately_earlier(adapter, client_and_token):
    client, token = client_and_token
    sid = _seed(adapter, "s-earlier", 6)
    tail = _get(client, token, sid, "?limit=2")
    cursor = tail["oldest_id"]
    earlier = _get(client, token, sid, f"?before_id={cursor}&limit=2")
    assert [e["message"] for e in earlier["events"]] == ["m2", "m3"]
    assert earlier["has_more"] is True  # m0/m1 remain


def test_limit_larger_than_log_has_no_more(adapter, client_and_token):
    client, token = client_and_token
    sid = _seed(adapter, "s-small", 3)
    body = _get(client, token, sid, "?limit=50")
    assert [e["message"] for e in body["events"]] == ["m0", "m1", "m2"]
    assert body["has_more"] is False
