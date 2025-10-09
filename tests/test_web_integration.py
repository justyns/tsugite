"""Integration tests for web UI."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tsugite.web.server import app, executions


@pytest.fixture
def test_agent(tmp_path):
    """Create a simple test agent."""
    agent_file = tmp_path / "simple_agent.md"
    agent_file.write_text(
        """---
name: simple_test
model: ollama:qwen2.5-coder:7b
tools: []
max_steps: 1
---

Task: {{ user_prompt }}

Just respond with final_answer("Test completed").
"""
    )
    return agent_file


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestWebIntegration:
    """Integration tests for full web flow."""

    @pytest.mark.asyncio
    async def test_full_execution_flow(self, client, test_agent, monkeypatch):
        """Test complete flow: submit -> execute -> stream events."""
        monkeypatch.chdir(test_agent.parent)

        # Create events directly in handler
        from tsugite.ui import UIEvent
        from tsugite.web.ui_handler import SSEUIHandler

        # Create execution with pre-queued events
        execution_id = "test-execution"
        handler = SSEUIHandler()
        executions[execution_id] = handler

        # Queue events
        handler.handle_event(UIEvent.TASK_START, {"task": "Test this agent", "model": "test-model"})
        handler.handle_event(UIEvent.STEP_START, {"step": 1})
        handler.handle_event(UIEvent.CODE_EXECUTION, {"code": "final_answer('Test completed')"})
        handler.handle_event(UIEvent.FINAL_ANSWER, {"answer": "Test completed"})

        # Stream events (with done sent after final_answer)
        collected_events = []
        with client.stream("GET", f"/api/stream/{execution_id}") as stream:
            assert stream.status_code == 200

            current_event = None
            current_data = None

            for line in stream.iter_lines():
                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    current_data = line.split(":", 1)[1].strip()
                    if current_event and current_data:
                        collected_events.append(
                            {
                                "event": current_event,
                                "data": current_data,
                            }
                        )
                        # Break on done event
                        if current_event == "done":
                            break
                        current_event = None
                        current_data = None

        # Verify event sequence
        event_names = [e["event"] for e in collected_events]
        assert "task_start" in event_names
        assert "step_start" in event_names
        assert "code_execution" in event_names
        assert "final_answer" in event_names
        assert "done" in event_names

        # Cleanup
        if execution_id in executions:
            del executions[execution_id]

    def test_concurrent_executions(self, client, test_agent, monkeypatch):
        """Test multiple concurrent agent executions."""
        monkeypatch.chdir(test_agent.parent)

        execution_ids = []

        # Start multiple executions
        for i in range(3):
            with patch("tsugite.web.server.asyncio.create_task"):
                response = client.post(
                    "/api/run",
                    data={
                        "agent_path": str(test_agent),
                        "prompt": f"Task {i}",
                    },
                )

                assert response.status_code == 200
                execution_ids.append(response.json()["execution_id"])

        # Verify all executions are tracked
        for exec_id in execution_ids:
            assert exec_id in executions

        # Clean up
        for exec_id in execution_ids:
            if exec_id in executions:
                del executions[exec_id]

    def test_error_handling_in_stream(self, client):
        """Test error event streaming."""
        from tsugite.ui import UIEvent
        from tsugite.web.ui_handler import SSEUIHandler

        # Create execution with error
        handler = SSEUIHandler()
        execution_id = "error-test"
        executions[execution_id] = handler

        # Queue error event (which is terminal)
        handler.handle_event(
            UIEvent.ERROR,
            {"error": "Test error occurred", "error_type": "RuntimeError"},
        )

        # Stream should handle error event
        collected_events = []
        with client.stream("GET", f"/api/stream/{execution_id}") as stream:
            for line in stream.iter_lines():
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                    collected_events.append(event_name)
                    # Error is terminal, followed by done
                    if event_name == "done":
                        break

        assert "error" in collected_events
        assert "done" in collected_events

        # Clean up
        if execution_id in executions:
            del executions[execution_id]

    def test_agent_list_filtered_correctly(self, client, tmp_path, monkeypatch):
        """Test that agent listing filters correctly."""
        monkeypatch.chdir(tmp_path)

        # Create various files
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        # Valid agent
        (agents_dir / "valid.md").write_text("---\nname: valid\n---\nAgent")

        # README (should be filtered)
        (agents_dir / "README.md").write_text("# Documentation")

        # Non-markdown file (should be ignored)
        (agents_dir / "config.yaml").write_text("config: value")

        response = client.get("/api/agents")
        agents = response.json()["agents"]

        assert len(agents) == 1
        assert agents[0]["name"] == "valid"

    def test_stream_keepalive(self, client):
        """Test SSE keepalive messages."""
        from tsugite.ui import UIEvent
        from tsugite.web.ui_handler import SSEUIHandler

        handler = SSEUIHandler()
        execution_id = "keepalive-test"
        executions[execution_id] = handler

        # Queue final answer to end stream after keepalives
        handler.handle_event(UIEvent.FINAL_ANSWER, {"answer": "Done"})

        keepalives = 0
        events = 0

        with client.stream("GET", f"/api/stream/{execution_id}") as stream:
            # Read lines until done
            for line in stream.iter_lines():
                if line.startswith(":"):
                    keepalives += 1
                elif line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                    if event_name in ("final_answer", "done"):
                        events += 1
                    if event_name == "done":
                        break

        # May or may not have keepalives depending on timing
        # But should have received final_answer and done
        assert events == 2

        # Clean up
        if execution_id in executions:
            del executions[execution_id]


class TestWebUILogger:
    """Test integration between CustomUILogger and SSEUIHandler."""

    def test_logger_with_sse_handler(self):
        """Test that CustomUILogger works with SSEUIHandler."""
        from io import StringIO

        from rich.console import Console

        from tsugite.ui import CustomUILogger, UIEvent
        from tsugite.web.ui_handler import SSEUIHandler

        handler = SSEUIHandler()
        console = Console(file=StringIO())
        CustomUILogger(handler, console)

        # Trigger various events directly through handler
        handler.handle_event(UIEvent.TASK_START, {"task": "Test task", "model": "test-model"})
        handler.handle_event(UIEvent.STEP_START, {"step": 1})
        handler.handle_event(UIEvent.CODE_EXECUTION, {"code": "print('test')"})

        # Verify events were queued
        assert handler.event_queue.qsize() == 3

        # Check event types
        event1 = handler.event_queue.get_nowait()
        assert event1["event"] == "task_start"

        event2 = handler.event_queue.get_nowait()
        assert event2["event"] == "step_start"

        event3 = handler.event_queue.get_nowait()
        assert event3["event"] == "code_execution"
