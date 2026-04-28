"""If a chat turn is already running for (agent, user_id), the server must
not start a second `run_agent` task on top of it. Double-dispatch is how we
get duplicate side effects (e.g. the same POST issued by both runs).
"""

import time
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from tsugite.daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite.daemon.config import AgentConfig, HTTPConfig
from tsugite.daemon.webhook_store import WebhookStore


@pytest.fixture
def tmp_workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def agent_config(tmp_workspace):
    return AgentConfig(workspace_dir=tmp_workspace, agent_file="default")


@pytest.fixture
def mock_adapter(agent_config, tmp_path):
    from tsugite.daemon.session_store import SessionStore
    from tsugite.workspace import WorkspaceNotFoundError

    session_store = SessionStore(tmp_path / "session_store.json")
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

    store = TokenStore(tmp_path / "tokens.json")
    return store


@pytest.fixture
def test_token(token_store):
    _st, raw = token_store.create_admin_token(name="test-token")
    return raw


@pytest.fixture
def server(agent_config, mock_adapter, tmp_path, token_store):
    webhook_store = WebhookStore(tmp_path / "webhooks.json")
    return HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8374),
        adapters={"test-agent": mock_adapter},
        webhook_store=webhook_store,
        agent_configs={"test-agent": agent_config},
        token_store=token_store,
    )


@pytest.fixture
def client(server):
    return TestClient(server.app)


def test_second_chat_for_same_user_is_rejected_while_first_runs(client, mock_adapter, test_token):
    """Second POST /chat for the same (agent, user_id) while the first is
    still running returns 409 and does not spawn a second agent run.
    """
    call_count = 0

    async def slow_handle(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # Simulate a slow in-flight agent turn so the second request overlaps.
        import asyncio

        await asyncio.sleep(0.5)
        return "ok"

    with patch.object(mock_adapter, "handle_message", side_effect=slow_handle):
        # Kick off the first request in a thread so we can issue a second while
        # it's still streaming.
        import threading

        first_done = threading.Event()
        first_status = {}

        def first_request():
            try:
                with client.stream(
                    "POST",
                    "/api/agents/test-agent/chat",
                    json={"message": "hello", "user_id": "alice"},
                    headers={"Authorization": f"Bearer {test_token}"},
                ) as resp:
                    first_status["code"] = resp.status_code
                    # Drain so the server actually runs.
                    for _ in resp.iter_bytes():
                        pass
            finally:
                first_done.set()

        t = threading.Thread(target=first_request, daemon=True)
        t.start()

        # Wait briefly so the first request begins running.
        time.sleep(0.1)

        resp2 = client.post(
            "/api/agents/test-agent/chat",
            json={"message": "second", "user_id": "alice"},
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert resp2.status_code == 409, f"expected 409 Conflict, got {resp2.status_code}: {resp2.text}"

        first_done.wait(timeout=5)

    assert call_count == 1, f"handle_message fired {call_count} times; should be 1"


def test_sequential_chats_for_same_user_both_run(client, mock_adapter, test_token):
    """Sanity: once the first chat finishes, a follow-up still works."""
    call_count = 0

    async def quick_handle(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return f"reply {call_count}"

    with patch.object(mock_adapter, "handle_message", side_effect=quick_handle):
        for _ in range(2):
            with client.stream(
                "POST",
                "/api/agents/test-agent/chat",
                json={"message": "hi", "user_id": "bob"},
                headers={"Authorization": f"Bearer {test_token}"},
            ) as resp:
                assert resp.status_code == 200
                for _ in resp.iter_bytes():
                    pass

    assert call_count == 2


def _make_session(mock_adapter, sid: str, user_id: str):
    """Pre-create an interactive session with an explicit id."""
    from tsugite.daemon.session_store import Session, SessionSource

    session = Session(
        id=sid,
        agent=mock_adapter.agent_name,
        source=SessionSource.INTERACTIVE.value,
        user_id=user_id,
    )
    mock_adapter.session_store.create_session(session)
    return session


def _start_chat_in_thread(client, test_token, *, session_id: str, user_id: str, message: str = "hi"):
    """Start a streaming /chat in a daemon thread; returns (thread, status_holder, done_event)."""
    import threading

    done = threading.Event()
    status: dict = {}

    def run():
        try:
            with client.stream(
                "POST",
                "/api/agents/test-agent/chat",
                json={"message": message, "user_id": user_id, "session_id": session_id},
                headers={"Authorization": f"Bearer {test_token}"},
            ) as resp:
                status["code"] = resp.status_code
                for _ in resp.iter_bytes():
                    pass
        finally:
            done.set()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, status, done


def test_distinct_sessions_same_user_run_in_parallel(client, mock_adapter, test_token, server):
    """Two POSTs with same (agent, user_id) but distinct session_id values must
    BOTH start streaming. The 409 guard should be per-session, not per-user.
    """
    _make_session(mock_adapter, "sess-A", "alice")
    _make_session(mock_adapter, "sess-B", "alice")

    call_count = 0
    seen_sessions: list[str] = []

    async def slow_handle(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        cc = kwargs["channel_context"]
        sid = (cc.metadata or {}).get("conv_id_override")
        seen_sessions.append(sid)
        import asyncio

        await asyncio.sleep(0.5)
        return "ok"

    with patch.object(mock_adapter, "handle_message", side_effect=slow_handle):
        _t1, st1, d1 = _start_chat_in_thread(client, test_token, session_id="sess-A", user_id="alice")
        time.sleep(0.1)
        _t2, st2, d2 = _start_chat_in_thread(client, test_token, session_id="sess-B", user_id="alice")

        d1.wait(timeout=5)
        d2.wait(timeout=5)

    assert st1.get("code") == 200, f"sess-A: expected 200, got {st1}"
    assert st2.get("code") == 200, f"sess-B: expected 200, got {st2}"
    assert call_count == 2, f"handle_message fired {call_count} times; expected 2"
    assert sorted(seen_sessions) == ["sess-A", "sess-B"]


def test_same_session_double_send_still_409s(client, mock_adapter, test_token):
    """Preserve original safety: two POSTs with the SAME session_id while the
    first is in flight must still return 409. Per-session keying must not
    weaken the same-session guard.
    """
    _make_session(mock_adapter, "sess-X", "alice")

    call_count = 0

    async def slow_handle(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        import asyncio

        await asyncio.sleep(0.5)
        return "ok"

    with patch.object(mock_adapter, "handle_message", side_effect=slow_handle):
        _t1, st1, d1 = _start_chat_in_thread(client, test_token, session_id="sess-X", user_id="alice")
        time.sleep(0.1)

        resp2 = client.post(
            "/api/agents/test-agent/chat",
            json={"message": "second", "user_id": "alice", "session_id": "sess-X"},
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert resp2.status_code == 409, f"expected 409, got {resp2.status_code}: {resp2.text}"

        d1.wait(timeout=5)

    assert call_count == 1


def test_cancel_chat_routes_by_session(client, mock_adapter, test_token, server):
    """POST /chat/cancel with session_id must cancel only that session's task.
    The peer session continues running.
    """
    import asyncio

    _make_session(mock_adapter, "sess-A", "alice")
    _make_session(mock_adapter, "sess-B", "alice")

    cancelled_sessions: list[str] = []
    completed_sessions: list[str] = []

    async def slow_handle(*args, **kwargs):
        cc = kwargs["channel_context"]
        sid = (cc.metadata or {}).get("conv_id_override")
        try:
            await asyncio.sleep(2.0)
            completed_sessions.append(sid)
            return "ok"
        except asyncio.CancelledError:
            cancelled_sessions.append(sid)
            raise

    with patch.object(mock_adapter, "handle_message", side_effect=slow_handle):
        _t1, _st1, d1 = _start_chat_in_thread(client, test_token, session_id="sess-A", user_id="alice")
        _t2, _st2, d2 = _start_chat_in_thread(client, test_token, session_id="sess-B", user_id="alice")
        time.sleep(0.2)

        resp = client.post(
            "/api/agents/test-agent/chat/cancel",
            json={"user_id": "alice", "session_id": "sess-A"},
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert resp.status_code == 200, f"cancel failed: {resp.text}"

        d1.wait(timeout=5)
        d2.wait(timeout=5)

    assert "sess-A" in cancelled_sessions, f"sess-A should be cancelled, got cancelled={cancelled_sessions}"
    assert "sess-B" in completed_sessions, f"sess-B should complete, got completed={completed_sessions}"
    assert "sess-B" not in cancelled_sessions


def test_cancel_chat_requires_session_id(client, test_token):
    """Cancel without session_id is ambiguous under per-session keying — 400."""
    resp = client.post(
        "/api/agents/test-agent/chat/cancel",
        json={"user_id": "alice"},
        headers={"Authorization": f"Bearer {test_token}"},
    )
    assert resp.status_code == 400, f"expected 400, got {resp.status_code}: {resp.text}"


def test_respond_routes_by_session(client, mock_adapter, test_token, server):
    """POST /respond with session_id routes to the correct backend when two
    sessions are awaiting concurrently. Without per-session keying, the
    request would land on whichever backend last registered at (agent, user).
    """
    import asyncio

    _make_session(mock_adapter, "sess-A", "alice")
    _make_session(mock_adapter, "sess-B", "alice")

    async def slow_handle(*args, **kwargs):
        await asyncio.sleep(2.0)
        return "ok"

    with patch.object(mock_adapter, "handle_message", side_effect=slow_handle):
        _t1, _st1, d1 = _start_chat_in_thread(client, test_token, session_id="sess-A", user_id="alice")
        _t2, _st2, d2 = _start_chat_in_thread(client, test_token, session_id="sess-B", user_id="alice")
        time.sleep(0.3)

        resolved_user = mock_adapter.resolve_http_user("alice")
        agent_name = mock_adapter.agent_name
        backend_a = server._active_backends.get((agent_name, resolved_user, "sess-A"))
        backend_b = server._active_backends.get((agent_name, resolved_user, "sess-B"))
        assert backend_a is not None, "backend for sess-A should be registered"
        assert backend_b is not None, "backend for sess-B should be registered"

        resp = client.post(
            "/api/agents/test-agent/respond",
            json={"user_id": "alice", "session_id": "sess-A", "response": "answer-A"},
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert resp.status_code == 200, f"respond failed: {resp.text}"
        assert backend_a._response == "answer-A"
        assert backend_b._response is None, "sess-B's backend should be untouched"

        client.post(
            "/api/agents/test-agent/chat/cancel",
            json={"user_id": "alice", "session_id": "sess-A"},
            headers={"Authorization": f"Bearer {test_token}"},
        )
        client.post(
            "/api/agents/test-agent/chat/cancel",
            json={"user_id": "alice", "session_id": "sess-B"},
            headers={"Authorization": f"Bearer {test_token}"},
        )
        d1.wait(timeout=5)
        d2.wait(timeout=5)


def test_status_returns_correct_session_busy(client, mock_adapter, test_token):
    """While S1 is mid-turn, /status?session_id=S1 reports busy=true and
    /status?session_id=S2 reports busy=false. On master, both report busy=true
    because the (agent, user) backend lookup ignores session_id.
    """
    import asyncio

    _make_session(mock_adapter, "sess-A", "alice")
    _make_session(mock_adapter, "sess-B", "alice")

    async def slow_handle(*args, **kwargs):
        await asyncio.sleep(1.5)
        return "ok"

    with patch.object(mock_adapter, "handle_message", side_effect=slow_handle):
        _t1, _st1, d1 = _start_chat_in_thread(client, test_token, session_id="sess-A", user_id="alice", message="probe")
        time.sleep(0.2)

        resp_a = client.get(
            "/api/agents/test-agent/status?user_id=alice&session_id=sess-A",
            headers={"Authorization": f"Bearer {test_token}"},
        )
        resp_b = client.get(
            "/api/agents/test-agent/status?user_id=alice&session_id=sess-B",
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert resp_a.status_code == 200
        assert resp_b.status_code == 200
        body_a = resp_a.json()
        body_b = resp_b.json()
        assert body_a.get("busy") is True, f"sess-A status: {body_a}"
        assert body_b.get("busy") is False, f"sess-B status: {body_b}"

        d1.wait(timeout=5)
