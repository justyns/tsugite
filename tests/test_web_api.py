"""Tests for web API endpoints."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tsugite.web.server import app, executions


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_agents(tmp_path):
    """Create mock agent files."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    # Create test agent files
    agent1 = agents_dir / "test_agent.md"
    agent1.write_text(
        """---
name: test_agent
model: ollama:qwen2.5-coder:7b
tools: []
---

Test agent
"""
    )

    agent2 = agents_dir / "helper.md"
    agent2.write_text(
        """---
name: helper
model: openai:gpt-4o-mini
---

Helper agent
"""
    )

    return tmp_path


class TestWebAPI:
    """Test web API endpoints."""

    def test_index_endpoint(self, client):
        """Test that index serves HTML."""
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_list_agents_empty(self, client, monkeypatch, tmp_path):
        """Test listing agents when none exist."""
        monkeypatch.chdir(tmp_path)

        response = client.get("/api/agents")

        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert data["agents"] == []

    def test_list_agents_with_agents(self, client, monkeypatch, mock_agents):
        """Test listing available agents."""
        monkeypatch.chdir(mock_agents)

        response = client.get("/api/agents")

        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert len(data["agents"]) == 2

        # Check agent structure
        agent_names = [a["name"] for a in data["agents"]]
        assert "test_agent" in agent_names
        assert "helper" in agent_names

    def test_run_agent_endpoint(self, client, monkeypatch, mock_agents):
        """Test running an agent."""
        monkeypatch.chdir(mock_agents)

        # Mock the run_agent_async to avoid actual execution
        with patch("tsugite.web.server.asyncio.create_task") as mock_task:
            response = client.post(
                "/api/run",
                data={
                    "agent_path": "agents/test_agent.md",
                    "prompt": "Test prompt",
                    "model": "openai:gpt-4o-mini",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "execution_id" in data
            assert isinstance(data["execution_id"], str)
            assert len(data["execution_id"]) > 0

            # Verify task was created
            mock_task.assert_called_once()

    def test_run_agent_stores_execution(self, client, monkeypatch, mock_agents):
        """Test that execution is stored."""
        monkeypatch.chdir(mock_agents)

        with patch("tsugite.web.server.asyncio.create_task"):
            response = client.post(
                "/api/run",
                data={
                    "agent_path": "agents/test_agent.md",
                    "prompt": "Test prompt",
                },
            )

            execution_id = response.json()["execution_id"]
            assert execution_id in executions

            # Clean up
            del executions[execution_id]

    @pytest.mark.asyncio
    async def test_stream_endpoint_not_found(self, client):
        """Test streaming with invalid execution ID."""
        response = client.get("/api/stream/nonexistent-id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_stream_endpoint_with_events(self, client):
        """Test SSE streaming with events."""
        from tsugite.web.ui_handler import SSEUIHandler

        # Create handler and execution
        handler = SSEUIHandler()
        execution_id = "test-execution-123"
        executions[execution_id] = handler

        # Queue some events
        from tsugite.custom_ui import UIEvent

        handler.handle_event(UIEvent.TASK_START, {"task": "Test task", "model": "test-model"})
        handler.handle_event(UIEvent.STEP_START, {"step": 1})
        handler.handle_event(UIEvent.FINAL_ANSWER, {"answer": "Done"})

        # Stream events
        with client.stream("GET", f"/api/stream/{execution_id}") as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

            # Collect events
            events = []
            for line in response.iter_lines():
                if line.startswith("event:"):
                    events.append(line.split(":", 1)[1].strip())

            # Should have task_start, step_start, final_answer, done
            assert "task_start" in events
            assert "step_start" in events
            assert "final_answer" in events
            assert "done" in events

        # Clean up
        if execution_id in executions:
            del executions[execution_id]

    def test_static_files_mounted(self, client):
        """Test that static files are accessible."""
        # Try to access CSS file
        response = client.get("/static/style.css")

        assert response.status_code == 200
        assert "text/css" in response.headers["content-type"]

    def test_run_agent_missing_fields(self, client):
        """Test running agent with missing required fields."""
        response = client.post("/api/run", data={"agent_path": "test.md"})

        # Should fail due to missing prompt
        assert response.status_code == 422  # Validation error

    def test_run_agent_optional_model(self, client, monkeypatch, mock_agents):
        """Test running agent without model parameter."""
        monkeypatch.chdir(mock_agents)

        with patch("tsugite.web.server.asyncio.create_task"):
            response = client.post(
                "/api/run",
                data={
                    "agent_path": "agents/test_agent.md",
                    "prompt": "Test prompt",
                    # model is optional
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "execution_id" in data


class TestExecutionCleanup:
    """Test execution cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_execution(self):
        """Test that cleanup removes execution."""
        from tsugite.web.server import cleanup_execution
        from tsugite.web.ui_handler import SSEUIHandler

        # Create execution
        execution_id = "test-cleanup"
        executions[execution_id] = SSEUIHandler()

        # Cleanup with short delay
        await cleanup_execution(execution_id, delay=0)

        assert execution_id not in executions

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_execution(self):
        """Test cleanup of nonexistent execution."""
        from tsugite.web.server import cleanup_execution

        # Should not raise error
        await cleanup_execution("nonexistent-id", delay=0)


class TestFindAgents:
    """Test agent discovery."""

    def test_find_agents_in_tsugite_dir(self, tmp_path):
        """Test finding agents in .tsugite directory."""
        from tsugite.web.server import find_agents

        tsugite_dir = tmp_path / ".tsugite"
        tsugite_dir.mkdir()
        (tsugite_dir / "custom.md").write_text("# Custom agent")

        agents = find_agents(tmp_path)

        assert len(agents) == 1
        assert agents[0]["name"] == "custom"

    def test_find_agents_skips_readme(self, tmp_path):
        """Test that README files are skipped."""
        from tsugite.web.server import find_agents

        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "README.md").write_text("# Readme")
        (agents_dir / "actual_agent.md").write_text("# Agent")

        agents = find_agents(tmp_path)

        assert len(agents) == 1
        assert agents[0]["name"] == "actual_agent"

    def test_find_agents_nested(self, tmp_path):
        """Test finding agents in nested directories."""
        from tsugite.web.server import find_agents

        nested = tmp_path / "agents" / "subdir"
        nested.mkdir(parents=True)
        (nested / "nested_agent.md").write_text("# Nested")

        agents = find_agents(tmp_path)

        assert len(agents) == 1
        assert agents[0]["name"] == "nested_agent"
        assert "subdir" in agents[0]["path"]
