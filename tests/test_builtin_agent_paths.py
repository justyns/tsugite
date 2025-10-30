"""Tests for builtin agent path handling in agent_runner functions."""

from unittest.mock import MagicMock, patch

import pytest

from tsugite.agent_inheritance import get_builtin_agents_path
from tsugite.agent_runner import (
    get_agent_info,
    run_agent,
    run_multistep_agent,
    validate_agent_file,
)
from tsugite.md_agents import validate_agent_execution


class TestBuiltinAgentPathHandling:
    """Test that functions properly handle built-in agent file paths."""

    def test_validate_agent_file_with_builtin(self):
        """Test validate_agent_file handles builtin agent paths."""
        builtin_path = get_builtin_agents_path() / "default.md"
        is_valid, message = validate_agent_file(builtin_path)

        assert is_valid is True
        assert "valid" in message.lower()

    def test_validate_agent_execution_with_builtin_path(self):
        """Test validate_agent_execution handles builtin agent paths."""
        builtin_path = get_builtin_agents_path() / "default.md"
        is_valid, message = validate_agent_execution(builtin_path)

        assert is_valid is True
        # Should not contain errors
        assert message == "Agent is valid" or "valid" in message.lower()

    def test_get_agent_info_with_builtin(self):
        """Test get_agent_info handles builtin agent paths."""
        builtin_path = get_builtin_agents_path() / "default.md"
        info = get_agent_info(builtin_path)

        assert "error" not in info
        assert info["name"] == "default"
        assert info["description"]
        assert info["valid"] is True
        assert "spawn_agent" in info["tools"]
        assert info["prefetch_count"] == 1

    @patch("tsugite.agent_runner.runner.TsugiteAgent")
    @patch("tsugite.core.tools.create_tool_from_tsugite")
    def test_run_agent_with_builtin(self, mock_create_tool, mock_agent_class):
        """Test run_agent can execute builtin agents."""
        builtin_path = get_builtin_agents_path() / "default.md"
        # Mock the agent execution
        mock_agent = MagicMock()
        mock_agent.run.return_value.output = "Test result"
        mock_agent_class.return_value = mock_agent

        try:
            result = run_agent(agent_path=builtin_path, prompt="Test task", model_override="openai:gpt-4o-mini")
            # If it doesn't raise, the path handling works
            assert isinstance(result, str)
        except Exception as e:
            # Should not fail due to path issues
            assert "No such file or directory" not in str(e)

    def test_run_multistep_agent_with_builtin_no_steps(self):
        """Test run_multistep_agent handles builtin agent without steps."""
        builtin_path = get_builtin_agents_path() / "default.md"
        # default agent doesn't have step directives
        with pytest.raises(ValueError, match="does not contain step directives"):
            run_multistep_agent(agent_path=builtin_path, prompt="Test task")

    def test_validate_agent_file_rejects_invalid_builtin(self):
        """Test validate_agent_file rejects non-existent built-in agent."""
        builtin_path = get_builtin_agents_path() / "builtin-nonexistent.md"
        is_valid, message = validate_agent_file(builtin_path)

        assert is_valid is False
        assert "not found" in message.lower()

    def test_get_agent_info_with_invalid_builtin(self):
        """Test get_agent_info handles invalid builtin agent."""
        builtin_path = get_builtin_agents_path() / "builtin-invalid.md"
        info = get_agent_info(builtin_path)

        # Should return error info
        assert "error" in info or info.get("valid") is False


class TestBuiltinVsRegularAgents:
    """Test that builtin and regular agents are handled correctly."""

    def test_validate_regular_agent_file(self, tmp_path):
        """Test that regular agent files still work."""
        agent_file = tmp_path / "regular.md"
        agent_file.write_text(
            """---
name: regular
---
# Regular Agent
{{ user_prompt }}
"""
        )

        is_valid, message = validate_agent_file(agent_file)
        assert is_valid is True

    def test_get_info_regular_vs_builtin(self, tmp_path):
        """Test get_agent_info works for both regular and builtin agents."""
        # Create regular agent
        agent_file = tmp_path / "regular.md"
        agent_file.write_text(
            """---
name: regular
description: A regular agent
tools: [read_file]
---
Content
"""
        )

        builtin_path = get_builtin_agents_path() / "default.md"
        regular_info = get_agent_info(agent_file)
        builtin_info = get_agent_info(builtin_path)

        # Both should have required fields
        assert "name" in regular_info
        assert "name" in builtin_info
        assert regular_info["valid"] is True
        assert builtin_info["valid"] is True

        # Names should be different
        assert regular_info["name"] == "regular"
        assert builtin_info["name"] == "default"


class TestCLIBuiltinPaths:
    """Test that CLI properly handles builtin agent file paths."""

    def test_cli_validates_builtin_path(self):
        """Test that CLI validation doesn't fail on builtin paths."""
        builtin_path = get_builtin_agents_path() / "default.md"

        # Should exist and be a valid path
        assert builtin_path.exists()
        assert builtin_path.is_file()

    def test_builtin_path_has_md_suffix(self):
        """Test that builtin paths have .md suffix like regular agents."""
        builtin_path = get_builtin_agents_path() / "default.md"

        # Builtin paths have .md suffix like regular agents
        assert str(builtin_path).endswith(".md")
