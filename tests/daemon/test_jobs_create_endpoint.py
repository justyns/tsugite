"""Tests for POST /api/jobs - the structured job-creation endpoint.

Distinct from POST /api/agents/{agent}/commands/job (the generic command
dispatcher, which returns a free-text string with no job_id). This route calls
JobsOrchestrator.create_and_start_job directly and returns job.to_payload() so
the caller gets the job_id synchronously.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite_daemon.auth import TokenStore
from tsugite_daemon.config import AgentConfig, HTTPConfig
from tsugite_daemon.job_store import Job
from tsugite_daemon.session_store import SessionStore
from tsugite_daemon.webhook_store import WebhookStore


@pytest.fixture
def tmp_workspace(tmp_path):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    return workspace_dir


@pytest.fixture
def agent_config(tmp_workspace):
    return AgentConfig(workspace_dir=tmp_workspace, agent_file="default")


@pytest.fixture
def mock_adapter(agent_config, tmp_path):
    from tsugite.workspace import WorkspaceNotFoundError

    session_store = SessionStore(tmp_path / "session_store.json")
    with patch("tsugite.workspace.Workspace") as mock_ws_cls:
        mock_ws_cls.load.side_effect = WorkspaceNotFoundError("not found")
        return HTTPAgentAdapter(
            agent_name="test-agent",
            agent_config=agent_config,
            session_store=session_store,
        )


@pytest.fixture
def token_store(tmp_path):
    return TokenStore(tmp_path / "tokens.json")


@pytest.fixture
def test_token(token_store):
    _st, raw = token_store.create_admin_token(name="jobs-create-token")
    return raw


class _FakeOrchestrator:
    """Records create_and_start_job kwargs and returns a preset (job, started)."""

    def __init__(self, job: Job | None = None, raise_exc: Exception | None = None):
        self.job = job
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    async def create_and_start_job(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_exc is not None:
            raise self.raise_exc
        return (self.job, None)


@pytest.fixture
def server(mock_adapter, agent_config, token_store, tmp_path):
    s = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8486),
        adapters={"test-agent": mock_adapter},
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
        agent_configs={"test-agent": agent_config},
        token_store=token_store,
    )
    s.jobs_orchestrator = _FakeOrchestrator(
        job=Job(id="job-new1", parent_session_id="sess-host-1", prompt="do the thing", state="queued")
    )
    return s


@pytest.fixture
def client(server):
    return TestClient(server.app)


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


class TestCreateJobAuth:
    def test_requires_auth(self, client):
        resp = client.post("/api/jobs", json={"agent": "test-agent", "user_id": "u1", "task": "x"})
        assert resp.status_code == 401

    def test_503_when_orchestrator_unavailable(self, mock_adapter, agent_config, token_store, test_token, tmp_path):
        s = HTTPServer(
            config=HTTPConfig(enabled=True, host="127.0.0.1", port=8487),
            adapters={"test-agent": mock_adapter},
            webhook_store=WebhookStore(tmp_path / "webhooks.json"),
            agent_configs={"test-agent": agent_config},
            token_store=token_store,
        )
        # jobs_orchestrator left None
        c = TestClient(s.app)
        resp = c.post(
            "/api/jobs", headers=_auth(test_token), json={"agent": "test-agent", "user_id": "u1", "task": "x"}
        )
        assert resp.status_code == 503


class TestCreateJobHappyPath:
    def test_returns_201_with_job_payload(self, client, test_token, server):
        with patch("tsugite_daemon.commands._create_job_host_session", return_value="sess-host-1"):
            resp = client.post(
                "/api/jobs",
                headers=_auth(test_token),
                json={
                    "agent": "test-agent",
                    "user_id": "u1",
                    "task": "do the thing",
                    "acceptance_criteria": "tests pass|PR open",
                    "max_attempts": 5,
                    "executor": "agent",
                    "model": "gpt",
                },
            )
        assert resp.status_code == 201
        body = resp.json()
        assert body["job_id"] == "job-new1"
        assert body["state"] == "queued"

        call = server.jobs_orchestrator.calls[-1]
        assert call["parent_session_id"] == "sess-host-1"
        assert call["prompt"] == "do the thing"
        assert call["acceptance_criteria"] == ["tests pass", "PR open"]
        assert call["max_attempts"] == 5
        assert call["executor"] == "agent"
        assert call["model"] == "gpt"

    def test_minimal_body_defaults(self, client, test_token, server):
        with patch("tsugite_daemon.commands._create_job_host_session", return_value="sess-host-1"):
            resp = client.post(
                "/api/jobs",
                headers=_auth(test_token),
                json={"agent": "test-agent", "user_id": "u1", "task": "just run"},
            )
        assert resp.status_code == 201
        call = server.jobs_orchestrator.calls[-1]
        assert call["acceptance_criteria"] == []
        assert call["executor"] == "agent"
        assert call["max_attempts"] is None
        assert call["model"] is None


class TestCreateJobValidation:
    def test_missing_task_is_400(self, client, test_token):
        resp = client.post("/api/jobs", headers=_auth(test_token), json={"agent": "test-agent", "user_id": "u1"})
        assert resp.status_code == 400
        assert "task" in resp.json()["error"]

    def test_missing_user_id_is_400(self, client, test_token):
        resp = client.post("/api/jobs", headers=_auth(test_token), json={"agent": "test-agent", "task": "x"})
        assert resp.status_code == 400
        assert "user_id" in resp.json()["error"]

    def test_unknown_agent_is_404(self, client, test_token):
        resp = client.post("/api/jobs", headers=_auth(test_token), json={"agent": "nope", "user_id": "u1", "task": "x"})
        assert resp.status_code == 404

    def test_orchestrator_valueerror_is_400(self, server, test_token):
        server.jobs_orchestrator = _FakeOrchestrator(raise_exc=ValueError("Unknown job executor: 'bogus'"))
        c = TestClient(server.app)
        with patch("tsugite_daemon.commands._create_job_host_session", return_value="sess-host-1"):
            resp = c.post(
                "/api/jobs",
                headers=_auth(test_token),
                json={"agent": "test-agent", "user_id": "u1", "task": "x", "executor": "bogus"},
            )
        assert resp.status_code == 400
        assert "executor" in resp.json()["error"]
