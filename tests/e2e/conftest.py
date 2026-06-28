"""Fixtures for Playwright E2E tests."""

import asyncio
import socket
import threading
import time
from unittest.mock import AsyncMock, patch

import pytest
import uvicorn
from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite_daemon.auth import TokenStore
from tsugite_daemon.config import AgentConfig, HTTPConfig
from tsugite_daemon.session_store import SessionStore
from tsugite_daemon.webhook_store import WebhookStore

from .helpers import (
    open_conversations,
    reload_conversations_view,
    select_session_in_view,
    wait_for_alpine_ready,
    wait_for_session_in_list,
)


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def e2e_tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("e2e")


@pytest.fixture(scope="session")
def e2e_workspace(e2e_tmp):
    ws = e2e_tmp / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture(scope="session")
def e2e_session_store(e2e_tmp):
    return SessionStore(e2e_tmp / "sessions.json", context_limits={"test-agent": 128000})


@pytest.fixture(scope="session")
def e2e_token_store(e2e_tmp):
    return TokenStore(e2e_tmp / "tokens.json")


@pytest.fixture(scope="session")
def e2e_auth_token(e2e_token_store):
    _meta, raw = e2e_token_store.create_admin_token(name="e2e-test")
    return raw


@pytest.fixture(scope="session")
def e2e_adapter(e2e_workspace, e2e_session_store):
    agent_config = AgentConfig(workspace_dir=e2e_workspace, agent_file="default")

    with (
        patch("tsugite.workspace.Workspace") as mock_ws,
        patch("tsugite.workspace.context.build_workspace_attachments", return_value=[]),
    ):
        from tsugite.workspace import WorkspaceNotFoundError

        mock_ws.load.side_effect = WorkspaceNotFoundError("not found")
        adapter = HTTPAgentAdapter(
            agent_name="test-agent",
            agent_config=agent_config,
            session_store=e2e_session_store,
        )

    # Tripwire: every e2e test that triggers chat must use mock_chat() first.
    # No real provider calls are allowed from this suite. Replace the default
    # handle_message with a raiser; mock_chat() swaps it for a configured fake.
    async def _require_mock_chat(*args, **kwargs):
        raise AssertionError(
            "e2e tests must call mock_chat(...) before sending a message. "
            "Real handle_message was invoked without the fixture; this would "
            "hit a real LLM provider."
        )

    adapter._original_handle_message = _require_mock_chat
    adapter.handle_message = _require_mock_chat
    return adapter


@pytest.fixture(scope="session")
def e2e_server(e2e_tmp, e2e_workspace, e2e_adapter, e2e_token_store):
    port = _free_port()
    config = HTTPConfig(enabled=True, host="127.0.0.1", port=port)
    agent_config = AgentConfig(workspace_dir=e2e_workspace, agent_file="default")

    server = HTTPServer(
        config=config,
        adapters={"test-agent": e2e_adapter},
        webhook_store=WebhookStore(e2e_tmp / "webhooks.json"),
        agent_configs={"test-agent": agent_config},
        token_store=e2e_token_store,
    )

    uvi_config = uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="warning")
    uvi_server = uvicorn.Server(uvi_config)
    uvi_server.install_signal_handlers = lambda: None

    thread = threading.Thread(target=asyncio.run, args=(uvi_server.serve(),), daemon=True)
    thread.start()

    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.1)

    yield f"http://127.0.0.1:{port}", server

    uvi_server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="session")
def base_url(e2e_server):
    url, _ = e2e_server
    return url


@pytest.fixture(autouse=True)
def _reset_daemon_state(e2e_session_store, e2e_server):
    """Drop every session and any live server state between tests.

    The daemon (uvicorn) and its session store are `scope="session"` because
    standing up a fresh server per test would be too slow. Clearing the
    in-memory dicts at fixture setup is cheap and gives each test a clean
    sidebar plus a clean live-progress map.
    """
    _url, server = e2e_server
    with e2e_session_store._lock:
        e2e_session_store._sessions.clear()
        e2e_session_store._thread_index.clear()
        e2e_session_store._channel_index.clear()
    server._active_chats.clear()
    yield


@pytest.fixture(autouse=True)
def _e2e_jsonl_history(reset_history_backend_fixture):
    """Pin the JSONL history backend for the e2e suite.

    These tests were written for the JSONL era: they seed `<id>.jsonl` files and
    patch `tsugite.history.storage.get_history_dir`. The daemon now defaults to the
    SQLite backend, so without this the in-process daemon would never read the
    seeded files. Depends on the global (autouse) reset_history_backend_fixture so
    that reset runs first on setup and the default backend is restored on teardown.
    """
    from tsugite.history import JsonlHistoryBackend, set_history_backend

    set_history_backend(JsonlHistoryBackend())


@pytest.fixture
def authenticated_page(page, base_url, e2e_auth_token):
    """Page with auth token pre-injected into localStorage."""
    page.goto(base_url + "/api/health")
    page.evaluate(f"localStorage.setItem('tsugite_token', '{e2e_auth_token}')")
    page.goto(base_url)
    wait_for_alpine_ready(page)
    return page


@pytest.fixture
def chat_page(authenticated_page, e2e_session_store):
    """Authenticated page with conversations view open and a session selected."""
    page = authenticated_page

    user_id = page.evaluate("Alpine.store('app').userId")
    session = e2e_session_store.get_or_create_interactive(user_id, "test-agent")

    open_conversations(page)
    reload_conversations_view(page)
    wait_for_session_in_list(page, session.id)
    select_session_in_view(page, session.id)
    page.wait_for_selector("textarea#message-input", timeout=5000)
    return page


@pytest.fixture
def mock_chat(e2e_adapter):
    """Factory to configure what the mock agent returns during chat.

    Usage:
        mock_chat("Hello!", events=[("reaction", {"emoji": "👍"})])
    """
    _original = getattr(e2e_adapter, "_original_handle_message", None)
    if _original is None:
        e2e_adapter._original_handle_message = e2e_adapter.handle_message

    def _configure(response="Test response", events=None):
        async def fake_handle(user_id, message, channel_context, custom_logger=None):
            if custom_logger and events:
                handler = custom_logger.ui_handler
                for ev_type, ev_data in events:
                    handler._emit(ev_type, ev_data)
            return response

        e2e_adapter.handle_message = AsyncMock(side_effect=fake_handle)

    yield _configure

    # Restore original (or previous mock) after test
    if _original:
        e2e_adapter.handle_message = _original
