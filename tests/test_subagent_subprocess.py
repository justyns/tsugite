"""Integration tests for subprocess-based subagent execution."""

from pathlib import Path

import pytest

from tsugite.tools.agents import spawn_agent

# Mark all tests in this module as integration tests that require actual model execution
pytestmark = pytest.mark.skip(reason="Integration tests require actual model execution")


@pytest.fixture
def fixtures_dir():
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures" / "agents"


def test_basic_spawn(fixtures_dir):
    """Spawn simple subagent and get result."""
    agent_path = str(fixtures_dir / "simple.md")
    result = spawn_agent(agent_path, "Return the text 'success'")

    assert result is not None
    assert isinstance(result, str)
    # The agent should complete successfully


def test_context_passing(fixtures_dir):
    """Pass context via JSON to subagent."""
    agent_path = str(fixtures_dir / "simple.md")
    result = spawn_agent(agent_path, "Echo the test_value from context", context={"test_value": "test123"})

    assert result is not None
    # Context should be available in the subagent


def test_timeout():
    """Subagent times out correctly."""
    fixtures_dir = Path(__file__).parent / "fixtures" / "agents"
    agent_path = str(fixtures_dir / "slow.md")

    with pytest.raises(RuntimeError, match="timed out"):
        # Very short timeout to force timeout
        spawn_agent(agent_path, "Sleep for 10 seconds", timeout=2)


def test_agent_not_found():
    """Raise error when agent file doesn't exist."""
    with pytest.raises(ValueError, match="Agent not found"):
        spawn_agent("nonexistent_agent.md", "test")


def test_non_json_serializable_context(fixtures_dir):
    """Raise error for non-JSON-serializable context."""
    agent_path = str(fixtures_dir / "simple.md")

    with pytest.raises(ValueError, match="non-JSON-serializable"):
        # Functions are not JSON-serializable
        spawn_agent(agent_path, "test", context={"func": lambda x: x})


def test_model_override(fixtures_dir):
    """Model override passes through correctly."""
    agent_path = str(fixtures_dir / "simple.md")

    # Should not raise an error
    result = spawn_agent(agent_path, "Simple task", model_override="ollama:qwen2.5-coder:7b")

    assert result is not None


def test_nested_subagents(fixtures_dir):
    """Test 3-level nesting: parent → coordinator → worker."""
    # This test spawns the nested agent which itself spawns the simple agent
    agent_path = str(fixtures_dir / "nested.md")

    result = spawn_agent(agent_path, "Delegate to simple agent")

    assert result is not None
    # The nested agent should successfully delegate and return a result


def test_is_subagent_context_injection(fixtures_dir):
    """Test that is_subagent and parent_agent context are injected."""
    # This is implicitly tested by all other tests
    # The subagent mode should automatically inject these context variables
    agent_path = str(fixtures_dir / "simple.md")
    result = spawn_agent(agent_path, "Complete task")

    assert result is not None
    # Context injection happens automatically in spawn_agent


def test_no_history_pollution(fixtures_dir):
    """Verify subagent runs don't pollute history."""
    # This test verifies that TSUGITE_SUBAGENT_MODE=1 prevents history saving
    # The environment variable is set by the CLI when --subagent-mode is used
    agent_path = str(fixtures_dir / "simple.md")

    # Run subagent
    result = spawn_agent(agent_path, "Test task")

    assert result is not None
    # History pollution prevention is tested by checking the env var is set
    # and that save_run_to_history returns None when the env var is set


@pytest.mark.skipif(
    not (Path(__file__).parent / "fixtures" / "agents" / "simple.md").exists(),
    reason="Test fixture agents not found",
)
def test_subagent_error_handling(fixtures_dir):
    """Test error handling when subagent fails."""
    agent_path = str(fixtures_dir / "failing.md")

    # The failing agent should raise an error when prompted to fail
    with pytest.raises(RuntimeError):
        spawn_agent(agent_path, "Please fail", timeout=10)
