"""If a chat turn is already running for a (user_id, session_id) pair, the server
must not start a second `run_agent` task on top of it. Double-dispatch is how we
get duplicate side effects (e.g. the same POST issued by both runs).
"""

import asyncio
import threading
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite_daemon.config import HTTPConfig, RuntimeDefaults
from tsugite_daemon.webhook_store import WebhookStore


@pytest.fixture
def tmp_workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def agent_config(tmp_workspace):
    return RuntimeDefaults(workspace_dir=tmp_workspace, agent_file="default")


@pytest.fixture
def mock_adapter(agent_config, tmp_path):
    from tsugite_daemon.session_store import SessionStore

    from tsugite.workspace import WorkspaceNotFoundError

    session_store = SessionStore(tmp_path / "session_store.json")
    with patch("tsugite.workspace.Workspace") as mock_ws_cls:
        mock_ws_cls.load.side_effect = WorkspaceNotFoundError("not found")
        return HTTPAgentAdapter(
            runtime=agent_config,
            session_store=session_store,
        )


@pytest.fixture
def token_store(tmp_path):
    from tsugite_daemon.auth import TokenStore

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
        adapter=mock_adapter,
        webhook_store=webhook_store,
        token_store=token_store,
    )


@pytest.fixture
def client(server):
    return TestClient(server.app)


class _HeldTurn:
    """Fake handle_message that stays in flight until the test releases it,
    plus the /chat callers holding it open.

    A sleeping handler races the test. Under load the turn finishes before the
    second request lands, and the in-flight assertions stop meaning anything.
    """

    def __init__(self, client, test_token):
        self._client = client
        self._token = test_token
        self._threads: list[threading.Thread] = []
        self._running = threading.Semaphore(0)
        self.release = threading.Event()
        self.started: list[str | None] = []
        self.completed: list[str | None] = []
        self.cancelled: list[str | None] = []

    async def handle(self, *args, **kwargs):
        session_id = (kwargs["channel_context"].metadata or {}).get("conv_id_override")
        self.started.append(session_id)
        self._running.release()
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self.release.wait, 5)
        except asyncio.CancelledError:
            self.cancelled.append(session_id)
            raise
        self.completed.append(session_id)
        return "ok"

    def start(self, *, user_id: str, session_id: str | None = None, message: str = "hi") -> dict:
        """POST /chat from a daemon thread. The returned dict gets the
        response status once the stream closes."""
        status: dict = {}
        payload = {"message": message, "user_id": user_id}
        if session_id is not None:
            payload["session_id"] = session_id

        def run():
            with self._client.stream(
                "POST",
                "/api/chat",
                json=payload,
                headers={"Authorization": f"Bearer {self._token}"},
            ) as resp:
                status["code"] = resp.status_code
                for _ in resp.iter_bytes():
                    pass

        t = threading.Thread(target=run, daemon=True)
        t.start()
        self._threads.append(t)
        return status

    def wait_until_running(self, count: int = 1, timeout: float = 5.0):
        """Block until `count` turns have entered the handler. A turn inside the
        handler is already registered in `_active_chats`, which the busy check and
        /status read."""
        for _ in range(count):
            assert self._running.acquire(timeout=timeout), f"fewer than {count} turns reached handle_message"

    def finish(self):
        self.release.set()
        for t in self._threads:
            t.join(timeout=5)
            assert not t.is_alive(), "a /chat request never finished streaming"


@contextmanager
def _held_turn(client, mock_adapter, test_token):
    """Patch handle_message with a held turn and, whatever the test does,
    release it and drain its requests before the fixtures tear down. A turn
    left blocked outlives the test and writes events into a history database
    the teardown has already closed."""
    turn = _HeldTurn(client, test_token)
    with patch.object(mock_adapter, "handle_message", side_effect=turn.handle):
        try:
            yield turn
        finally:
            turn.finish()


def _make_session(mock_adapter, sid: str, user_id: str):
    """Pre-create an interactive session with an explicit id."""
    from tsugite_daemon.session_store import Session, SessionSource

    session = Session(
        id=sid,
        source=SessionSource.INTERACTIVE.value,
        user_id=user_id,
    )
    mock_adapter.session_store.create_session(session)
    return session


def test_second_chat_for_same_user_is_queued_while_first_runs(client, mock_adapter, test_token):
    """Second POST /chat for the same (agent, user_id) while the first is
    still running is queued (202) and does not spawn a second agent run.
    """
    with _held_turn(client, mock_adapter, test_token) as turn:
        turn.start(user_id="alice", message="hello")
        turn.wait_until_running()

        resp2 = client.post(
            "/api/chat",
            json={"message": "second", "user_id": "alice"},
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert resp2.status_code == 202, f"expected 202 Accepted, got {resp2.status_code}: {resp2.text}"

    assert len(turn.started) == 1, f"handle_message fired {len(turn.started)} times; should be 1"


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
                "/api/chat",
                json={"message": "hi", "user_id": "bob"},
                headers={"Authorization": f"Bearer {test_token}"},
            ) as resp:
                assert resp.status_code == 200
                for _ in resp.iter_bytes():
                    pass

    assert call_count == 2


def test_distinct_sessions_same_user_run_in_parallel(client, mock_adapter, test_token):
    """Two POSTs with same (agent, user_id) but distinct session_id values must
    BOTH start streaming. The busy check should be per-session, not per-user.
    """
    _make_session(mock_adapter, "sess-A", "alice")
    _make_session(mock_adapter, "sess-B", "alice")

    with _held_turn(client, mock_adapter, test_token) as turn:
        st1 = turn.start(user_id="alice", session_id="sess-A")
        turn.wait_until_running()
        st2 = turn.start(user_id="alice", session_id="sess-B")
        turn.wait_until_running()

    assert st1.get("code") == 200, f"sess-A: expected 200, got {st1}"
    assert st2.get("code") == 200, f"sess-B: expected 200, got {st2}"
    assert sorted(turn.started) == ["sess-A", "sess-B"]


def test_same_session_double_send_is_queued(client, mock_adapter, test_token):
    """Two POSTs with the same session_id must not run two turns at once. The
    second is queued and runs when the first finishes.
    """
    _make_session(mock_adapter, "sess-X", "alice")

    with _held_turn(client, mock_adapter, test_token) as turn:
        turn.start(user_id="alice", session_id="sess-X")
        turn.wait_until_running()

        resp2 = client.post(
            "/api/chat",
            json={"message": "second", "user_id": "alice", "session_id": "sess-X"},
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert resp2.status_code == 202, f"expected 202, got {resp2.status_code}: {resp2.text}"

    assert len(turn.started) == 1


def test_cancel_chat_routes_by_session(client, mock_adapter, test_token):
    """POST /chat/cancel with session_id must cancel only that session's task.
    The peer session continues running.
    """
    _make_session(mock_adapter, "sess-A", "alice")
    _make_session(mock_adapter, "sess-B", "alice")

    with _held_turn(client, mock_adapter, test_token) as turn:
        turn.start(user_id="alice", session_id="sess-A")
        turn.start(user_id="alice", session_id="sess-B")
        turn.wait_until_running(2)

        resp = client.post(
            "/api/chat/cancel",
            json={"user_id": "alice", "session_id": "sess-A"},
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert resp.status_code == 200, f"cancel failed: {resp.text}"

    assert "sess-A" in turn.cancelled, f"sess-A should be cancelled, got cancelled={turn.cancelled}"
    assert "sess-B" in turn.completed, f"sess-B should complete, got completed={turn.completed}"
    assert "sess-B" not in turn.cancelled


def test_cancel_chat_requires_session_id(client, test_token):
    """Cancel without session_id is ambiguous under per-session keying, so 400."""
    resp = client.post(
        "/api/chat/cancel",
        json={"user_id": "alice"},
        headers={"Authorization": f"Bearer {test_token}"},
    )
    assert resp.status_code == 400, f"expected 400, got {resp.status_code}: {resp.text}"


def test_cancel_chat_sets_cooperative_cancel_event(client, mock_adapter, test_token, server):
    """Cancel must set the per-chat cooperative cancel Event, not only cancel the
    awaiting coroutine. task.cancel alone tears down the SSE stream but leaves the
    agent loop running in its to_thread worker; the Event is what the worker checks.
    """
    _make_session(mock_adapter, "sess-COOP", "alice")

    with _held_turn(client, mock_adapter, test_token) as turn:
        turn.start(user_id="alice", session_id="sess-COOP")
        turn.wait_until_running()

        chats = list(server._active_chats.values())
        assert len(chats) == 1, f"expected one active chat, got {len(chats)}"
        chat = chats[0]
        assert not chat.cancel_event.is_set()

        resp = client.post(
            "/api/chat/cancel",
            json={"user_id": "alice", "session_id": "sess-COOP"},
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert resp.status_code == 200, resp.text
        assert chat.cancel_event.is_set()


def test_status_returns_correct_session_busy(client, mock_adapter, test_token):
    """While S1 is mid-turn, /status?session_id=S1 reports busy=true and
    /status?session_id=S2 reports busy=false.
    """
    _make_session(mock_adapter, "sess-A", "alice")
    _make_session(mock_adapter, "sess-B", "alice")

    with _held_turn(client, mock_adapter, test_token) as turn:
        turn.start(user_id="alice", session_id="sess-A", message="probe")
        turn.wait_until_running()

        resp_a = client.get(
            "/api/chat/status?user_id=alice&session_id=sess-A",
            headers={"Authorization": f"Bearer {test_token}"},
        )
        resp_b = client.get(
            "/api/chat/status?user_id=alice&session_id=sess-B",
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert resp_a.status_code == 200
        assert resp_b.status_code == 200
        body_a = resp_a.json()
        body_b = resp_b.json()
        assert body_a.get("busy") is True, f"sess-A status: {body_a}"
        assert body_b.get("busy") is False, f"sess-B status: {body_b}"
