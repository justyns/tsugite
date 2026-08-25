"""Durable, resumable chat ask (approval / ask_user).

Two halves:

- Message persistence: the user_input is recorded up front, BEFORE the context
  detector runs, so a turn parked on a blocking approval still persists the
  message and survives a reload. The runner's own later record_user_input is
  deduped to a no-op (exactly one user_input event).
- Answer-by-id: ask_user carries a durable ``ask_id`` (emitted, persisted,
  replayed), and POST /respond resolves the blocking ask by that id. An absent
  ask_id is rejected (400).
"""

import asyncio
import threading
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite_daemon.config import HTTPConfig, RuntimeDefaults
from tsugite_daemon.session_store import Session, SessionSource, SessionStore
from tsugite_daemon.webhook_store import WebhookStore


class _StubAdapter(BaseAdapter):
    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


def _fake_result():
    return MagicMock(
        token_count=0,
        cost=0,
        execution_steps=[],
        provider_state={},
        last_input_tokens=0,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        __str__=lambda self: "ok",
    )


def _cc(**metadata) -> ChannelContext:
    return ChannelContext(source="http", channel_id=None, user_id="alice", reply_to="http:alice", metadata=metadata)


@pytest.fixture
def clean_registry(monkeypatch):
    """Only explicitly registered detectors run (no real URL/webfetch detector
    that could block on an approval we didn't set up)."""
    from tsugite import context as ctx_module

    monkeypatch.setattr(ctx_module, "ensure_loaded", lambda: None)
    ctx_module.reset_context_providers()
    yield
    ctx_module.reset_context_providers()


@pytest.fixture
def persist_adapter(tmp_path, monkeypatch):
    """A BaseAdapter whose heavy internals are stubbed but whose up-front history
    recording (open_or_create_session + record_user_input) runs for real against a
    temp history backend."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "agent.md").write_text("---\nname: test-agent\n---\n\nHi.\n")
    store = SessionStore(tmp_path / "store.json")
    config = RuntimeDefaults(workspace_dir=ws, agent_file=str(ws / "agent.md"))
    adapter = _StubAdapter(config, store)

    async def _noop_auto_title(*_a, **_k):
        return None

    monkeypatch.setattr(adapter, "_resolve_agent_path", lambda *a, **k: ws / "agent.md")
    monkeypatch.setattr(
        adapter, "_build_message_context", lambda msg, *a, **k: f"<message_context>x</message_context>\n\n{msg}"
    )
    monkeypatch.setattr(adapter, "_build_agent_context", lambda *a, **k: {})
    monkeypatch.setattr(adapter, "_save_history", lambda **k: None)
    monkeypatch.setattr(adapter, "_update_skill_ttl", lambda *a, **k: None)
    monkeypatch.setattr(adapter, "_auto_title_session", _noop_auto_title)
    monkeypatch.setattr("tsugite_daemon.adapters.base.run_agent", lambda *a, **k: _fake_result())
    return adapter, store


class TestMessagePersistence:
    @pytest.mark.asyncio
    async def test_message_persisted_before_blocking_detector(self, persist_adapter, clean_registry):
        """The message is durable the instant it arrives: it is recorded before the
        detector, so a turn parked on an approval keeps it across a reload."""
        from tsugite.attachments.base import Attachment
        from tsugite.context import ContextProvider, register_context_provider

        adapter, store = persist_adapter
        at_detect = threading.Event()
        release = threading.Event()

        def detect(message, ctx):
            at_detect.set()
            release.wait(timeout=5)  # stand in for a parked approval prompt
            return [Attachment.context("gate", "Gate", "gated")]

        register_context_provider(ContextProvider(key="gate", label="Gate", detect=detect))
        session = adapter.session_store.get_or_create_interactive("alice")

        task = asyncio.create_task(
            adapter.handle_message(user_id="alice", message="hello there", channel_context=_cc())
        )
        for _ in range(500):
            if at_detect.is_set():
                break
            await asyncio.sleep(0.01)
        assert at_detect.is_set(), "detector never ran"

        # Parked on the approval, yet the message is already recorded.
        user_inputs = [e for e in store.read_events(session.id) if e.get("type") == "user_input"]
        assert len(user_inputs) == 1
        assert user_inputs[0]["text"] == "hello there"

        release.set()
        await task

    @pytest.mark.asyncio
    async def test_runner_second_record_is_deduped_to_one_user_input(self, persist_adapter, monkeypatch):
        """The runner's own record_user_input (via user_input_for_history) must be a
        no-op after the up-front recording: exactly one user_input event lands, and
        the common-case recorded text is unchanged."""
        adapter, store = persist_adapter

        def fake_run_agent_like_runner(*args, **kwargs):
            from tsugite.agent_runner.history_integration import open_or_create_session, record_user_input

            storage = open_or_create_session(
                agent_path=kwargs["agent_path"],
                agent_name="test-agent",
                model="test-model",
                continue_conversation_id=kwargs["continue_conversation_id"],
            )
            if storage is not None:
                record_user_input(
                    storage, kwargs["user_input_for_history"], channel_metadata=kwargs.get("channel_metadata")
                )
            return _fake_result()

        monkeypatch.setattr("tsugite_daemon.adapters.base.run_agent", fake_run_agent_like_runner)
        session = adapter.session_store.get_or_create_interactive("bob")

        await adapter.handle_message(user_id="bob", message="just a message", channel_context=_cc())

        user_inputs = [e for e in store.read_events(session.id) if e.get("type") == "user_input"]
        assert len(user_inputs) == 1
        assert user_inputs[0]["text"] == "just a message"


def test_ask_events_persist_with_ask_id_top_level(tmp_path):
    """ask_user / ask_answered land in history (they are in _PERSIST_EVENT_TYPES),
    and the read path flattens their data so ask_id/question/options arrive as
    top-level fields on reload, like every other event."""
    from tsugite_daemon.adapters.http import SSEProgressHandler
    from tsugite_daemon.adapters.http.helpers import build_session_event_persister

    store = SessionStore(tmp_path / "store.json")
    sid = "sess-ask"
    progress = SSEProgressHandler()
    progress.set_event_persister(build_session_event_persister(store, sid))

    progress._emit(
        "ask_user",
        {
            "ask_id": "ask-abc12345",
            "question": "Fetch content from evil.test?",
            "question_type": "approval",
            "options": ["Approve", "Deny"],
        },
    )
    progress._emit("ask_answered", {"ask_id": "ask-abc12345", "answer": "Approve"})

    events = store.read_events(sid)
    ask = next(e for e in events if e["type"] == "ask_user")
    assert ask["ask_id"] == "ask-abc12345"
    assert ask["question"] == "Fetch content from evil.test?"
    assert ask["question_type"] == "approval"
    assert ask["options"] == ["Approve", "Deny"]
    answered = next(e for e in events if e["type"] == "ask_answered")
    assert answered["ask_id"] == "ask-abc12345"
    assert answered["answer"] == "Approve"


# ── Answer-by-id over the real HTTP endpoint ──


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
    from unittest.mock import patch

    from tsugite.workspace import WorkspaceNotFoundError

    session_store = SessionStore(tmp_path / "session_store.json")
    with patch("tsugite.workspace.Workspace") as mock_ws_cls:
        mock_ws_cls.load.side_effect = WorkspaceNotFoundError("not found")
        return HTTPAgentAdapter(
            runtime=agent_config,
            session_store=session_store,
        )


@pytest.fixture
def test_token(tmp_path):
    from tsugite_daemon.auth import TokenStore

    store = TokenStore(tmp_path / "tokens.json")
    _st, raw = store.create_admin_token(name="test-token")
    return store, raw


@pytest.fixture
def server(agent_config, mock_adapter, tmp_path, test_token):
    token_store, _raw = test_token
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


def _auth(test_token):
    return {"Authorization": f"Bearer {test_token[1]}"}


def _make_session(mock_adapter, sid: str, user_id: str = "alice"):
    session = Session(id=sid, source=SessionSource.INTERACTIVE.value, user_id=user_id)
    mock_adapter.session_store.create_session(session)
    return session


class TestRespondByAskId:
    def test_resolves_by_ask_id_when_triple_would_miss(self, client, mock_adapter, test_token, server):
        """ask_id resolves the still-blocking backend even though no ActiveChat is
        registered for the request's (agent, user, session) triple."""
        from tsugite_daemon.adapters.http.sse import _PENDING_ASKS, HTTPInteractionBackend, SSEProgressHandler

        backend = HTTPInteractionBackend(SSEProgressHandler())
        backend._ask_id = "ask-live0001"
        _PENDING_ASKS["ask-live0001"] = backend
        try:
            resp = client.post(
                "/api/chat/respond",
                json={
                    "user_id": "alice",
                    "session_id": "sess-none",
                    "ask_id": "ask-live0001",
                    "response": "Approve",
                },
                headers=_auth(test_token),
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["status"] == "ok"
            assert backend._response == "Approve"
        finally:
            _PENDING_ASKS.pop("ask-live0001", None)

    def test_absent_ask_id_returns_400(self, client, mock_adapter, test_token):
        """ask_id is required: the legacy (agent, user, session) fallback is gone,
        so a respond with no ask_id is a 400, not a triple-keyed lookup or 404."""
        resp = client.post(
            "/api/chat/respond",
            json={"user_id": "alice", "session_id": "sess-Z", "response": "hi"},
            headers=_auth(test_token),
        )
        assert resp.status_code == 400, resp.text

    def test_stale_ask_id_persists_ask_answered_and_non_404(self, client, mock_adapter, test_token, server):
        """A durably-pending ask whose backend is gone (timeout / restart): the
        answer records ask_answered so a reload stops re-prompting, and the client
        gets a clear non-404 status instead of a bare 404."""
        sid = "sess-stale"
        _make_session(mock_adapter, sid)
        mock_adapter.session_store.append_event(
            sid, {"type": "ask_user", "ask_id": "ask-stale001", "question": "Fetch?", "question_type": "approval"}
        )

        resp = client.post(
            "/api/chat/respond",
            json={"user_id": "alice", "session_id": sid, "ask_id": "ask-stale001", "response": "Deny"},
            headers=_auth(test_token),
        )
        assert resp.status_code != 404, resp.text
        assert resp.json()["status"] == "expired"

        events = mock_adapter.session_store.read_events(sid)
        answered = [e for e in events if e.get("type") == "ask_answered" and e.get("ask_id") == "ask-stale001"]
        assert len(answered) == 1
        assert answered[0]["answer"] == "Deny"

    def test_unknown_ask_id_writes_nothing_and_non_404(self, client, mock_adapter, test_token, server):
        """An ask_id that is not durably pending must not spam the log, and still
        returns a non-404 the UI can clear on."""
        sid = "sess-unknown"
        _make_session(mock_adapter, sid)

        resp = client.post(
            "/api/chat/respond",
            json={"user_id": "alice", "session_id": sid, "ask_id": "ask-nope0001", "response": "x"},
            headers=_auth(test_token),
        )
        assert resp.status_code != 404, resp.text
        events = mock_adapter.session_store.read_events(sid)
        assert not [e for e in events if e.get("type") == "ask_answered"]
