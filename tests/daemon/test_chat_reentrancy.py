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
        assert resp2.status_code == 409, (
            f"expected 409 Conflict, got {resp2.status_code}: {resp2.text}"
        )

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
