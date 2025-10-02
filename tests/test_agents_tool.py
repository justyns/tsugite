"""Tests for agent orchestration tools."""

from unittest.mock import MagicMock, patch

import pytest

from tsugite.tools import call_tool, get_tool, tool
from tsugite.tools.agents import spawn_agent


@pytest.fixture
def register_spawn_agent():
    """Register spawn_agent tool for testing."""
    tool(spawn_agent)
    return spawn_agent


class TestSpawnAgentTool:
    """Test the spawn_agent tool function."""

    @pytest.fixture(autouse=True)
    def setup_test_agent(self, temp_dir, simple_agent_content):
        """Set up test fixtures."""
        from .conftest import create_agent_file

        self.temp_dir = temp_dir
        self.agent_file = create_agent_file(temp_dir, simple_agent_content, "test_agent.md")

    def test_spawn_agent_tool_registered(self, register_spawn_agent):
        """Test that spawn_agent tool is registered."""
        tool_info = get_tool("spawn_agent")
        assert tool_info.name == "spawn_agent"
        assert "Spawn a sub-agent and return its result" in tool_info.description

    @patch("tsugite.agent_runner.run_agent")
    def test_spawn_agent_basic(self, mock_run_agent):
        """Test basic spawn_agent functionality."""
        mock_run_agent.return_value = "Sub-agent completed successfully"

        result = spawn_agent(agent_path=str(self.agent_file), prompt="Test task")

        assert result == "Sub-agent completed successfully"
        mock_run_agent.assert_called_once()

        # Check the call arguments
        call_args = mock_run_agent.call_args
        assert call_args[1]["agent_path"] == self.agent_file
        assert call_args[1]["prompt"] == "Test task"
        assert call_args[1]["context"] == {}
        assert call_args[1]["model_override"] is None
        assert call_args[1]["debug"] is False

    @patch("tsugite.agent_runner.run_agent")
    def test_spawn_agent_with_context(self, mock_run_agent):
        """Test spawn_agent with context variables."""
        mock_run_agent.return_value = "Context passed successfully"

        context = {"key1": "value1", "key2": 42}
        result = spawn_agent(agent_path=str(self.agent_file), prompt="Test with context", context=context)

        assert result == "Context passed successfully"
        call_args = mock_run_agent.call_args
        assert call_args[1]["context"] == context

    @patch("tsugite.agent_runner.run_agent")
    def test_spawn_agent_with_model_override(self, mock_run_agent):
        """Test spawn_agent with model override."""
        mock_run_agent.return_value = "Model overridden"

        result = spawn_agent(agent_path=str(self.agent_file), prompt="Test with model override", model_override="gpt-4")

        assert result == "Model overridden"
        call_args = mock_run_agent.call_args
        assert call_args[1]["model_override"] == "gpt-4"

    def test_spawn_agent_nonexistent_file(self):
        """Test spawn_agent with non-existent file."""
        with pytest.raises(ValueError, match="Invalid agent file.*not found"):
            spawn_agent(agent_path="nonexistent.md", prompt="Test task")

    def test_spawn_agent_non_markdown_file(self):
        """Test spawn_agent with non-markdown file."""
        txt_file = self.temp_dir / "test.txt"
        txt_file.write_text("Not a markdown file")

        with pytest.raises(ValueError, match="Invalid agent file.*must be a .md file"):
            spawn_agent(agent_path=str(txt_file), prompt="Test task")

    def test_spawn_agent_relative_path(self):
        """Test spawn_agent with relative path."""
        # Change to temp directory to test relative paths
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)

            with patch("tsugite.agent_runner.run_agent") as mock_run_agent:
                mock_run_agent.return_value = "Relative path works"

                result = spawn_agent(agent_path="test_agent.md", prompt="Test relative path")

                assert result == "Relative path works"
                call_args = mock_run_agent.call_args
                assert call_args[1]["agent_path"] == self.agent_file
        finally:
            os.chdir(original_cwd)

    @patch("tsugite.agent_runner.run_agent")
    def test_spawn_agent_execution_failure(self, mock_run_agent):
        """Test spawn_agent when sub-agent execution fails."""
        mock_run_agent.side_effect = RuntimeError("Agent execution failed")

        with pytest.raises(RuntimeError, match="Sub-agent execution failed"):
            spawn_agent(agent_path=str(self.agent_file), prompt="Test failure")

    def test_spawn_agent_via_call_tool(self, register_spawn_agent):
        """Test spawn_agent called via the tool registry."""
        with patch("tsugite.agent_runner.run_agent") as mock_run_agent:
            mock_run_agent.return_value = "Called via registry"

            result = call_tool("spawn_agent", agent_path=str(self.agent_file), prompt="Test via registry")

            assert result == "Called via registry"


class TestAgentOrchestrationIntegration:
    """Integration tests for agent orchestration."""

    @pytest.fixture(autouse=True)
    def setup_integration_agents(self, temp_dir, spawn_agent_content, simple_agent_content):
        """Set up test fixtures."""
        from .conftest import create_agent_file

        self.temp_dir = temp_dir
        self.parent_file = create_agent_file(temp_dir, spawn_agent_content, "parent_agent.md")
        self.child_file = create_agent_file(temp_dir, simple_agent_content, "child_agent.md")

    @patch("tsugite.agent_runner.CodeAgent")
    @patch("tsugite.agent_runner.get_model")
    @patch("tsugite.agent_runner.get_smolagents_tools")
    def test_nested_agent_execution(self, mock_get_tools, mock_get_model, mock_code_agent):
        """Test that agents can spawn other agents."""
        # Mock the CodeAgent to simulate tool calls
        mock_agent_instance = MagicMock()
        mock_code_agent.return_value = mock_agent_instance

        # Mock the child agent execution
        def mock_run_side_effect(*args, **kwargs):
            if "child_agent" in str(kwargs.get("agent_path", "")):
                return "Child agent completed task"
            return "Parent agent coordinated successfully"

        mock_agent_instance.run.side_effect = mock_run_side_effect

        # Mock other dependencies
        mock_get_model.return_value = MagicMock()
        mock_get_tools.return_value = []

        # This would be a real integration test if we had full mocking
        # For now, we verify the structure is in place
        assert self.parent_file.exists()
        assert self.child_file.exists()

        # Verify spawn_agent tool is available
        from tsugite.tools import list_tools

        tool(spawn_agent)  # Register for this test
        assert "spawn_agent" in list_tools()


class TestSpawnAgentParameters:
    """Test spawn_agent parameter validation and types."""

    def test_spawn_agent_parameter_info(self, register_spawn_agent):
        """Test that spawn_agent has correct parameter information."""
        tool_info = get_tool("spawn_agent")

        params = tool_info.parameters
        assert "agent_path" in params
        assert "prompt" in params
        assert "context" in params
        assert "model_override" in params

        # Check required parameters
        assert params["agent_path"]["required"] is True
        assert params["prompt"]["required"] is True
        assert params["context"]["required"] is False
        assert params["model_override"]["required"] is False

        # Check defaults
        assert params["context"]["default"] is None
        assert params["model_override"]["default"] is None

    def test_spawn_agent_missing_required_params(self, register_spawn_agent):
        """Test spawn_agent with missing required parameters."""
        with pytest.raises(ValueError, match="Invalid parameter 'agent_path'.*missing"):
            call_tool("spawn_agent", prompt="Test")

        with pytest.raises(ValueError, match="Invalid parameter 'prompt'.*missing"):
            call_tool("spawn_agent", agent_path="test.md")
