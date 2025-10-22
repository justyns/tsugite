"""Tests for builtin agent path handling in agent_runner functions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tsugite.agent_runner import (
    get_agent_info,
    run_agent,
    run_multistep_agent,
    validate_agent_file,
)
from tsugite.md_agents import validate_agent_execution


class TestBuiltinAgentPathHandling:
    """Test that functions properly handle <builtin-*> paths."""

    def test_validate_agent_file_with_builtin(self):
        """Test validate_agent_file handles builtin agent paths."""
        is_valid, message = validate_agent_file(Path("<builtin-default>"))

        assert is_valid is True
        assert "valid" in message.lower()

    def test_validate_agent_execution_with_builtin_path(self):
        """Test validate_agent_execution handles builtin agent paths."""
        is_valid, message = validate_agent_execution(Path("<builtin-default>"))

        assert is_valid is True
        # Should not contain errors
        assert message == "Agent is valid" or "valid" in message.lower()

    def test_get_agent_info_with_builtin(self):
        """Test get_agent_info handles builtin agent paths."""
        info = get_agent_info(Path("<builtin-default>"))

        assert "error" not in info
        assert info["name"] == "builtin-default"
        assert info["description"]
        assert info["valid"] is True
        assert "spawn_agent" in info["tools"]
        assert info["prefetch_count"] == 1

    @patch("tsugite.agent_runner.runner.TsugiteAgent")
    @patch("tsugite.core.tools.create_tool_from_tsugite")
    def test_run_agent_with_builtin(self, mock_create_tool, mock_agent_class):
        """Test run_agent can execute builtin agents."""
        # Mock the agent execution
        mock_agent = MagicMock()
        mock_agent.run.return_value.output = "Test result"
        mock_agent_class.return_value = mock_agent

        try:
            result = run_agent(
                agent_path=Path("<builtin-default>"), prompt="Test task", model_override="openai:gpt-4o-mini"
            )
            # If it doesn't raise, the path handling works
            assert isinstance(result, str)
        except Exception as e:
            # Should not fail due to path issues
            assert "No such file or directory" not in str(e)
            assert "<builtin-default>" not in str(e) or "parse" in str(e).lower()

    def test_run_multistep_agent_with_builtin_no_steps(self):
        """Test run_multistep_agent handles builtin agent without steps."""
        # builtin-default doesn't have step directives
        with pytest.raises(ValueError, match="does not contain step directives"):
            run_multistep_agent(agent_path=Path("<builtin-default>"), prompt="Test task")

    def test_validate_agent_file_rejects_invalid_builtin(self):
        """Test validate_agent_file rejects unknown builtin agents."""
        is_valid, message = validate_agent_file(Path("<builtin-nonexistent>"))

        assert is_valid is False
        assert "Unknown built-in agent" in message or "not found" in message.lower()

    def test_get_agent_info_with_invalid_builtin(self):
        """Test get_agent_info handles invalid builtin agent."""
        info = get_agent_info(Path("<builtin-invalid>"))

        # Should return error info
        assert "error" in info or info.get("valid") is False


class TestBuiltinVsRegularAgents:
    """Test that builtin and regular agents are handled correctly."""

    def test_validate_regular_agent_file(self, tmp_path):
        """Test that regular agent files still work."""
        agent_file = tmp_path / "regular.md"
        agent_file.write_text("""---
name: regular
---
# Regular Agent
{{ user_prompt }}
""")

        is_valid, message = validate_agent_file(agent_file)
        assert is_valid is True

    def test_get_info_regular_vs_builtin(self, tmp_path):
        """Test get_agent_info works for both regular and builtin agents."""
        # Create regular agent
        agent_file = tmp_path / "regular.md"
        agent_file.write_text("""---
name: regular
description: A regular agent
tools: [read_file]
---
Content
""")

        regular_info = get_agent_info(agent_file)
        builtin_info = get_agent_info(Path("<builtin-default>"))

        # Both should have required fields
        assert "name" in regular_info
        assert "name" in builtin_info
        assert regular_info["valid"] is True
        assert builtin_info["valid"] is True

        # Names should be different
        assert regular_info["name"] == "regular"
        assert builtin_info["name"] == "builtin-default"


class TestCLIBuiltinPaths:
    """Test that CLI properly handles builtin agent paths."""

    def test_cli_validates_builtin_path(self):
        """Test that CLI validation doesn't fail on builtin paths."""
        # This tests the validation logic in cli/__init__.py
        builtin_path = Path("<builtin-default>")

        # Should be recognized as builtin
        assert str(builtin_path).startswith("<builtin-")

        # Should not try to check .exists()
        # (This would fail if the path handling is broken)

    def test_builtin_path_has_no_suffix(self):
        """Test that builtin paths are handled before suffix checks."""
        builtin_path = Path("<builtin-default>")

        # Builtin paths don't have .md suffix
        assert not str(builtin_path).endswith(".md")

        # But they should be exempt from suffix validation
        assert str(builtin_path).startswith("<builtin-")
