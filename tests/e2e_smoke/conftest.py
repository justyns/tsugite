"""Fixtures for the real-LLM e2e smoke tier.

Standalone from `tests/e2e/conftest.py` because the mock-chat tripwire there
is what guarantees the main suite never hits a real provider. Reusing those
fixtures here would either bypass the tripwire (dangerous if a test bleeds
back to `tests/e2e/`) or fight against it.

Auto-skips the whole directory if `TSUGITE_E2E_REAL_LLM` is unset or no
provider key is present.
"""

import asyncio
import os
import socket
import threading
import time
from unittest.mock import patch

import pytest
import uvicorn

from tsugite.daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite.daemon.auth import TokenStore
from tsugite.daemon.config import AgentConfig, HTTPConfig
from tsugite.daemon.session_store import SessionStore
from tsugite.daemon.webhook_store import WebhookStore


_REQUIRED_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY")


def pytest_collection_modifyitems(config, items):
    """Skip the entire directory unless TSUGITE_E2E_REAL_LLM=1 + a provider key."""
    if not os.environ.get("TSUGITE_E2E_REAL_LLM"):
        skip = pytest.mark.skip(reason="TSUGITE_E2E_REAL_LLM=1 not set; real-LLM smoke tier is opt-in")
        for item in items:
            item.add_marker(skip)
        return
    if not any(os.environ.get(k) for k in _REQUIRED_KEYS):
        skip = pytest.mark.skip(reason=f"No provider key set; one of {_REQUIRED_KEYS} required")
        for item in items:
            item.add_marker(skip)


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


SMOKE_MODEL = os.environ.get("TSUGITE_SMOKE_MODEL", "openai:gpt-4o-mini")


@pytest.fixture(scope="session")
def smoke_tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("e2e_smoke")


@pytest.fixture(scope="session")
def smoke_workspace(smoke_tmp):
    ws = smoke_tmp / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture(scope="session")
def smoke_session_store(smoke_tmp):
    return SessionStore(smoke_tmp / "sessions.json", context_limits={"smoke-agent": 128000})


@pytest.fixture(scope="session")
def smoke_token_store(smoke_tmp):
    return TokenStore(smoke_tmp / "tokens.json")


@pytest.fixture(scope="session")
def smoke_auth_token(smoke_token_store):
    _meta, raw = smoke_token_store.create_admin_token(name="e2e-smoke")
    return raw


@pytest.fixture(scope="session")
def smoke_agent_file(smoke_tmp):
    """Tiny agent file that fits the test_agent expectations."""
    path = smoke_tmp / "smoke-agent.md"
    path.write_text(
        f"""---
name: smoke-agent
extends: none
model: {SMOKE_MODEL}
max_turns: 3
tools: []
---

You are a terse helper. Answer with a single short sentence.
"""
    )
    return path


@pytest.fixture(scope="session")
def smoke_adapter(smoke_workspace, smoke_session_store, smoke_agent_file):
    agent_config = AgentConfig(workspace_dir=smoke_workspace, agent_file=str(smoke_agent_file))

    with (
        patch("tsugite.workspace.Workspace") as mock_ws,
        patch("tsugite.workspace.context.build_workspace_attachments", return_value=[]),
    ):
        from tsugite.workspace import WorkspaceNotFoundError

        mock_ws.load.side_effect = WorkspaceNotFoundError("not found")
        adapter = HTTPAgentAdapter(
            agent_name="smoke-agent",
            agent_config=agent_config,
            session_store=smoke_session_store,
        )
    # Crucial: NO mock-chat tripwire here. Real handle_message stays in place.
    return adapter


@pytest.fixture(scope="session")
def smoke_server(smoke_tmp, smoke_workspace, smoke_adapter, smoke_token_store, smoke_agent_file):
    port = _free_port()
    config = HTTPConfig(enabled=True, host="127.0.0.1", port=port)
    agent_config = AgentConfig(workspace_dir=smoke_workspace, agent_file=str(smoke_agent_file))

    server = HTTPServer(
        config=config,
        adapters={"smoke-agent": smoke_adapter},
        webhook_store=WebhookStore(smoke_tmp / "webhooks.json"),
        agent_configs={"smoke-agent": agent_config},
        token_store=smoke_token_store,
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
def smoke_base_url(smoke_server):
    url, _ = smoke_server
    return url


@pytest.fixture
def smoke_authenticated_page(page, smoke_base_url, smoke_auth_token):
    page.goto(smoke_base_url + "/api/health")
    page.evaluate(f"localStorage.setItem('tsugite_token', '{smoke_auth_token}')")
    page.goto(smoke_base_url)
    page.wait_for_function(
        "typeof Alpine !== 'undefined' && Alpine.store('app') && !Alpine.store('app').authRequired",
        timeout=10000,
    )
    return page
