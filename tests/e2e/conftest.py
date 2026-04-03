"""Fixtures for Playwright E2E tests."""

import asyncio
import socket
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import uvicorn

from tsugite.daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite.daemon.auth import TokenStore
from tsugite.daemon.config import AgentConfig, HTTPConfig
from tsugite.daemon.session_store import SessionStore
from tsugite.daemon.webhook_store import WebhookStore


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


@pytest.fixture
def authenticated_page(page, base_url, e2e_auth_token):
    """Page with auth token pre-injected into localStorage."""
    page.goto(base_url + "/api/health")
    page.evaluate(f"localStorage.setItem('tsugite_token', '{e2e_auth_token}')")
    page.goto(base_url)
    page.wait_for_function("!Alpine.store('app').authRequired", timeout=5000)
    return page


@pytest.fixture
def chat_page(authenticated_page, e2e_session_store):
    """Authenticated page with conversations view open and a session selected."""
    page = authenticated_page

    # Ensure an interactive session exists for the default user
    user_id = page.evaluate("Alpine.store('app').userId")
    e2e_session_store.get_or_create_interactive(user_id, "test-agent")

    page.locator("nav button", has_text="Conversations").click()
    page.wait_for_function("Alpine.store('app').view === 'conversations'", timeout=3000)
    page.wait_for_selector("textarea", timeout=5000)
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
