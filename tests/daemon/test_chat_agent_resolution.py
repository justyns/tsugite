"""A reply to a session must land on the session's OWNING agent, even when the
client points the request at an agent that isn't a live HTTP adapter.

The web UI's Jobs view opens a job's session (parent/host/worker) carrying the
job's worker agent-file (e.g. ``job_worker``) as the URL agent. ``job_worker`` is
not an HTTP adapter, so ``POST /api/agents/job_worker/chat`` used to 404 with
"unknown agent: job_worker" - the user launched a job, it finished, they tried to
reply, and the send dead-ended. The session_id is authoritative: the route must
resolve the session's real owner and proceed."""

from unittest.mock import patch

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite_daemon.auth import TokenStore
from tsugite_daemon.config import AgentConfig, HTTPConfig
from tsugite_daemon.session_store import Session, SessionSource, SessionStore


@pytest.fixture
def adapter(tmp_path):
    from tsugite.workspace import WorkspaceNotFoundError

    workspace = tmp_path / "ws"
    workspace.mkdir()
    store = SessionStore(tmp_path / "session_store.json")
    config = AgentConfig(workspace_dir=workspace, agent_file="default")
    with patch("tsugite.workspace.Workspace") as mock_ws:
        mock_ws.load.side_effect = WorkspaceNotFoundError("nope")
        return HTTPAgentAdapter(agent_name="test-agent", agent_config=config, session_store=store)


@pytest.fixture
def client_and_token(adapter, tmp_path):
    token_store = TokenStore(tmp_path / "tokens.json")
    _t, raw = token_store.create_admin_token(name="t")
    server = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8374),
        adapters={"test-agent": adapter},
        webhook_store=None,
        agent_configs={"test-agent": adapter.agent_config},
        token_store=token_store,
    )
    client = TestClient(server.app)
    client.app_server = server
    return client, raw


def _mk_session(adapter, sid):
    adapter.session_store.create_session(
        Session(id=sid, agent="test-agent", source=SessionSource.SPAWNED.value, user_id="u1")
    )
    return sid


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_chat_resolves_session_owner_when_url_agent_not_adapter(adapter, client_and_token):
    """The reported bug: a reply posted to a non-adapter agent (the job's worker
    agent-file) with a valid session_id must resolve the session's owner rather
    than 404. An empty message then trips normal validation (400), proving the
    route got PAST adapter resolution without starting a real turn."""
    client, token = client_and_token
    sid = _mk_session(adapter, "s-jobparent")

    resp = client.post(
        "/api/agents/job_worker/chat",
        headers=_auth(token),
        json={"user_id": "u1", "message": "", "session_id": sid},
    )

    assert resp.status_code != 404, resp.text
    assert "unknown agent" not in resp.text
    assert resp.status_code == 400  # empty message -> reached normal validation


def test_chat_unknown_agent_still_404_without_resolvable_session(adapter, client_and_token):
    """The fallback must not blanket-accept bogus agents: with no resolvable
    session it stays a 404 unknown-agent."""
    client, token = client_and_token

    resp = client.post(
        "/api/agents/job_worker/chat",
        headers=_auth(token),
        json={"user_id": "u1", "message": "hi", "session_id": "no-such-session"},
    )

    assert resp.status_code == 404
    assert "unknown agent: job_worker" in resp.text


def test_cancel_resolves_session_owner_when_url_agent_not_adapter(adapter, client_and_token):
    """Stop shares the mechanism: after the reply fix routes the turn through the
    session's owner, a cancel posted to the worker agent-file must resolve the
    same owner and report the honest 'no active chat' instead of 'unknown agent'."""
    client, token = client_and_token
    sid = _mk_session(adapter, "s-cancel")

    resp = client.post(
        "/api/agents/job_worker/chat/cancel",
        headers=_auth(token),
        json={"user_id": "u1", "session_id": sid},
    )

    assert resp.status_code == 404
    assert "unknown agent" not in resp.text
    assert "no active chat" in resp.text
