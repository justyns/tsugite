"""Tests for agent orchestration tools."""

import pytest

from tsugite.tools import call_tool, get_tool, tool
from tsugite.tools.agents import spawn_agent


@pytest.fixture
def register_spawn_agent():
    """Register spawn_agent tool for testing."""
    tool(parent_only=True)(spawn_agent)
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
        assert "Spawn subagent as subprocess" in tool_info.description

    def test_spawn_agent_nonexistent_file(self):
        """Test spawn_agent with non-existent file."""
        with pytest.raises(ValueError, match="Agent not found"):
            spawn_agent(agent_path="nonexistent.md", prompt="Test task")


class TestSpawnAgentContext:
    def test_spawn_agent_rejects_non_serializable_context(self, temp_dir, simple_agent_content):
        from .conftest import create_agent_file

        agent_file = create_agent_file(temp_dir, simple_agent_content, "test_agent.md")
        with pytest.raises(ValueError, match="non-JSON-serializable"):
            spawn_agent(agent_path=str(agent_file), prompt="Test task", context={"bad": object()})


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
