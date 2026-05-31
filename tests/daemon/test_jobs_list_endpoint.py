"""Tests for the GET /api/jobs list endpoint used by the Jobs tab."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from tsugite.daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite.daemon.auth import TokenStore
from tsugite.daemon.config import AgentConfig, HTTPConfig
from tsugite.daemon.job_store import Job, JobStore
from tsugite.daemon.session_store import SessionStore
from tsugite.daemon.webhook_store import WebhookStore


@pytest.fixture
def tmp_workspace(tmp_path):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    return workspace_dir


@pytest.fixture
def agent_config(tmp_workspace):
    return AgentConfig(workspace_dir=tmp_workspace, agent_file="default")


@pytest.fixture
def http_config():
    return HTTPConfig(enabled=True, host="127.0.0.1", port=8484)


@pytest.fixture
def mock_adapter(agent_config, tmp_path):
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
def webhook_store(tmp_path):
    return WebhookStore(tmp_path / "webhooks.json")


@pytest.fixture
def token_store(tmp_path):
    return TokenStore(tmp_path / "tokens.json")


@pytest.fixture
def test_token(token_store):
    _st, raw = token_store.create_admin_token(name="jobs-list-token")
    return raw


@pytest.fixture
def job_store(tmp_path):
    store = JobStore(tmp_path / "jobs.json")
    # Seed a realistic spread across all states so filter logic is exercised.
    store.add(Job(id="job-r1", parent_session_id="parent-1", prompt="running task", state="running", agent="odyn"))
    store.add(Job(id="job-v1", parent_session_id="parent-1", prompt="verifying", state="verifying", agent="odyn"))
    store.add(Job(id="job-q1", parent_session_id="parent-1", prompt="queued task", state="queued", agent="assistant"))
    store.add(Job(id="job-s1", parent_session_id="parent-1", prompt="stuck task", state="stuck", agent="odyn"))
    store.add(Job(id="job-e1", parent_session_id="parent-1", prompt="errored", state="errored", agent="odyn"))
    store.add(Job(id="job-d1", parent_session_id="parent-1", prompt="done!", state="done", agent="assistant"))
    store.add(Job(id="job-c1", parent_session_id="parent-1", prompt="cancelled", state="cancelled", agent="assistant"))
    return store


@pytest.fixture
def server(http_config, mock_adapter, webhook_store, agent_config, token_store, job_store):
    s = HTTPServer(
        config=http_config,
        adapters={"test-agent": mock_adapter},
        webhook_store=webhook_store,
        agent_configs={"test-agent": agent_config},
        token_store=token_store,
    )
    s.job_store = job_store
    # Set a truthy placeholder so any orchestrator-requiring routes don't 503.
    # GET /api/jobs reads only from job_store, but we keep this consistent.
    s.jobs_orchestrator = object()
    return s


@pytest.fixture
def client(server):
    return TestClient(server.app)


class TestListJobsEndpoint:
    def test_requires_auth(self, client):
        resp = client.get("/api/jobs")
        assert resp.status_code == 401

    def test_returns_503_when_job_store_unavailable(
        self, http_config, mock_adapter, webhook_store, agent_config, token_store, test_token
    ):
        s = HTTPServer(
            config=http_config,
            adapters={"test-agent": mock_adapter},
            webhook_store=webhook_store,
            agent_configs={"test-agent": agent_config},
            token_store=token_store,
        )
        # Intentionally do not set s.job_store
        c = TestClient(s.app)
        resp = c.get("/api/jobs", headers={"Authorization": f"Bearer {test_token}"})
        assert resp.status_code == 503

    def test_lists_all_jobs(self, client, test_token):
        resp = client.get("/api/jobs", headers={"Authorization": f"Bearer {test_token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert "jobs" in data
        assert len(data["jobs"]) == 7
        ids = {j["job_id"] for j in data["jobs"]}
        assert {"job-r1", "job-v1", "job-q1", "job-s1", "job-e1", "job-d1", "job-c1"} == ids

    def test_job_payload_shape_matches_emit_event(self, client, test_token):
        resp = client.get("/api/jobs", headers={"Authorization": f"Bearer {test_token}"})
        job = next(j for j in resp.json()["jobs"] if j["job_id"] == "job-r1")
        expected_keys = {
            "job_id",
            "parent_session_id",
            "worker_session_id",
            "verifier_session_id",
            "state",
            "prompt",
            "verify_attempts",
            "error",
            "attempts",
            "acceptance_criteria",
            "agent",
            "model",
            "created_at",
            "updated_at",
            "resolved_at",
            "spawned_by",
        }
        assert expected_keys.issubset(set(job.keys()))
        assert job["state"] == "running"
        assert job["agent"] == "odyn"

    def test_state_filter_exact_match(self, client, test_token):
        resp = client.get("/api/jobs?state=queued", headers={"Authorization": f"Bearer {test_token}"})
        assert resp.status_code == 200
        jobs = resp.json()["jobs"]
        assert {j["job_id"] for j in jobs} == {"job-q1"}

    def test_state_alias_stuck_includes_errored(self, client, test_token):
        resp = client.get("/api/jobs?state=stuck", headers={"Authorization": f"Bearer {test_token}"})
        assert resp.status_code == 200
        ids = {j["job_id"] for j in resp.json()["jobs"]}
        assert ids == {"job-s1", "job-e1"}

    def test_state_alias_active_includes_running_and_verifying(self, client, test_token):
        resp = client.get("/api/jobs?state=active", headers={"Authorization": f"Bearer {test_token}"})
        assert resp.status_code == 200
        ids = {j["job_id"] for j in resp.json()["jobs"]}
        assert ids == {"job-r1", "job-v1"}

    def test_state_alias_resolved_includes_done_and_cancelled(self, client, test_token):
        resp = client.get("/api/jobs?state=resolved", headers={"Authorization": f"Bearer {test_token}"})
        assert resp.status_code == 200
        ids = {j["job_id"] for j in resp.json()["jobs"]}
        assert ids == {"job-d1", "job-c1"}

    def test_limit_caps_results(self, client, test_token):
        resp = client.get("/api/jobs?limit=2", headers={"Authorization": f"Bearer {test_token}"})
        assert resp.status_code == 200
        assert len(resp.json()["jobs"]) == 2

    def test_empty_job_store_returns_empty_list(
        self, http_config, mock_adapter, webhook_store, agent_config, token_store, test_token, tmp_path
    ):
        empty_store = JobStore(tmp_path / "empty_jobs.json")
        s = HTTPServer(
            config=http_config,
            adapters={"test-agent": mock_adapter},
            webhook_store=webhook_store,
            agent_configs={"test-agent": agent_config},
            token_store=token_store,
        )
        s.job_store = empty_store
        c = TestClient(s.app)
        resp = c.get("/api/jobs", headers={"Authorization": f"Bearer {test_token}"})
        assert resp.status_code == 200
        assert resp.json() == {"jobs": []}
