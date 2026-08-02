"""Tests for builtin agent path handling in agent_runner functions."""

from tsugite.agent_inheritance import get_builtin_agents_path
from tsugite.agent_runner import (
    get_agent_info,
    run_agent,
    validate_agent_file,
)
from tsugite.md_agents import validate_agent_execution
from tsugite.options import ExecutionOptions


def _exec_options() -> ExecutionOptions:
    """Builtins declare no model, so a run without one falls back to the machine's
    configured default and fails wherever none is set."""
    return ExecutionOptions(model_override="openai:gpt-4o-mini")


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
        assert info["prefetch_count"] == 2  # get_skills_for_template + get_failed_skills_for_template

    def test_get_agent_info_returns_model_raw(self, tmp_path):
        """get_agent_info includes model_raw with the unformatted model string."""
        agent_path = tmp_path / "test.md"
        agent_path.write_text("---\nname: test\nmodel: openai:gpt-4\ntools: []\n---\nHello\n{{ user_prompt }}")
        info = get_agent_info(agent_path)
        assert info["model_raw"] == "openai:gpt-4"

    def test_get_agent_info_model_raw_none_when_unset(self, tmp_path):
        """model_raw is None when agent has no model set."""
        agent_path = tmp_path / "test.md"
        agent_path.write_text("---\nname: test\ntools: []\n---\nHello\n{{ user_prompt }}")
        info = get_agent_info(agent_path)
        assert info["model_raw"] is None

    def test_run_agent_with_builtin(self, monkeypatch):
        """run_agent resolves a builtin agent file and runs it, prompt included."""
        builtin_path = get_builtin_agents_path() / "default.md"
        seen = {}

        async def fake_agent_run(self, task, return_full_result=False, stream=False):
            seen["task"] = task
            return "Test result"

        monkeypatch.setattr("tsugite.core.agent.TsugiteAgent.run", fake_agent_run)

        result = run_agent(agent_path=builtin_path, prompt="Test task", exec_options=_exec_options())

        assert result == "Test result"
        assert "Test task" in seen["task"]

    def test_run_agent_with_builtin_no_steps(self, monkeypatch):
        """A stepless builtin runs as a single prompt rather than being rejected."""
        builtin_path = get_builtin_agents_path() / "default.md"

        async def fake_agent_run(self, task, return_full_result=False, stream=False):
            return "done"

        monkeypatch.setattr("tsugite.core.agent.TsugiteAgent.run", fake_agent_run)

        assert run_agent(agent_path=builtin_path, prompt="Test task", exec_options=_exec_options()) == "done"

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
