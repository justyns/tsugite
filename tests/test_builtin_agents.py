"""Tests for built-in agents."""

from pathlib import Path

from tsugite.agent_inheritance import find_agent_file
from tsugite.agent_utils import list_local_agents
from tsugite.builtin_agents import get_builtin_default_agent, is_builtin_agent


class TestBuiltinAgents:
    def test_get_builtin_default_agent(self):
        """Test getting the built-in default agent."""
        agent = get_builtin_default_agent()

        assert agent.config.name == "builtin-default"
        assert agent.config.description
        assert "helpful" in agent.config.instructions.lower()
        assert agent.file_path == Path("<builtin-default>")

    def test_is_builtin_agent(self):
        """Test checking if agent name is built-in."""
        assert is_builtin_agent("builtin-default") is True
        assert is_builtin_agent("default") is False
        assert is_builtin_agent("custom") is False
        assert is_builtin_agent("") is False

    def test_find_builtin_agent(self, tmp_path):
        """Test that find_agent_file returns special path for built-in."""
        found = find_agent_file("builtin-default", tmp_path)

        assert found is not None
        assert str(found) == "<builtin-default>"

    def test_list_includes_builtin(self, tmp_path):
        """Test that list_local_agents includes built-in agents."""
        agents = list_local_agents(tmp_path)

        assert "Built-in" in agents
        assert len(agents["Built-in"]) == 1
        assert agents["Built-in"][0] == Path("<builtin-default>")


class TestBuiltinInheritance:
    def test_inheritance_uses_builtin_when_no_default_md(self, tmp_path):
        """Test that agents inherit from built-in when no default.md exists."""
        from tsugite.md_agents import parse_agent_file

        # Create an agent without explicit extends
        agent_file = tmp_path / "test.md"
        agent_file.write_text(
            """---
name: test_agent
tools: [read_file]
---

# Test Agent
Task: {{ user_prompt }}
"""
        )

        agent = parse_agent_file(agent_file)

        # Agent should have inherited instructions from built-in
        assert agent.config.instructions
        assert "helpful" in agent.config.instructions.lower()

    def test_user_default_overrides_builtin(self, tmp_path):
        """Test that user's default.md overrides built-in."""
        from tsugite.md_agents import parse_agent_file

        # Create user's default.md
        tsugite_dir = tmp_path / ".tsugite"
        tsugite_dir.mkdir()
        default_file = tsugite_dir / "default.md"
        default_file.write_text(
            """---
name: default
instructions: Custom user instructions
---

# User Default
"""
        )

        # Create agent
        agent_file = tmp_path / "test.md"
        agent_file.write_text(
            """---
name: test_agent
---

# Test Agent
"""
        )

        agent = parse_agent_file(agent_file)

        # Should have user's custom instructions, not built-in
        assert "Custom user instructions" in agent.config.instructions
        assert "helpful" not in agent.config.instructions.lower()

    def test_extends_none_opts_out_of_builtin(self, tmp_path):
        """Test that extends: none opts out of built-in inheritance."""
        from tsugite.md_agents import parse_agent_file

        agent_file = tmp_path / "standalone.md"
        agent_file.write_text(
            """---
name: standalone
extends: none
---

# Standalone Agent
Task: {{ user_prompt }}
"""
        )

        agent = parse_agent_file(agent_file)

        # Should not have built-in instructions
        assert not agent.config.instructions or "helpful" not in agent.config.instructions.lower()

    def test_builtin_agent_parseable(self):
        """Test that built-in agent can be parsed successfully."""
        agent = get_builtin_default_agent()

        assert agent.config.name == "builtin-default"
        assert agent.content
        assert "{{ user_prompt }}" in agent.content


class TestBuiltinDefaultAutoDiscovery:
    """Test builtin-default agent's auto-discovery features."""

    def test_builtin_default_has_spawn_agent_tool(self):
        """Test that builtin-default includes spawn_agent tool."""
        agent = get_builtin_default_agent()

        assert "spawn_agent" in agent.config.tools

    def test_builtin_default_has_prefetch(self):
        """Test that builtin-default has prefetch configured."""
        agent = get_builtin_default_agent()

        assert agent.config.prefetch is not None
        assert len(agent.config.prefetch) > 0

        # Check that list_agents is in prefetch
        prefetch_tools = [p.get("tool") for p in agent.config.prefetch]
        assert "list_agents" in prefetch_tools

    def test_builtin_default_prefetch_assigns_variable(self):
        """Test that prefetch assigns to available_agents variable."""
        agent = get_builtin_default_agent()

        list_agents_prefetch = next((p for p in agent.config.prefetch if p.get("tool") == "list_agents"), None)

        assert list_agents_prefetch is not None
        assert list_agents_prefetch.get("assign") == "available_agents"

    def test_builtin_default_content_structure(self):
        """Test that builtin-default has delegation instructions."""
        agent = get_builtin_default_agent()

        # Should have conditional block for available agents
        assert "{% if available_agents %}" in agent.content

        # Should mention delegation
        assert "delegate" in agent.content.lower()
        assert "spawn_agent" in agent.content

        # Should have example usage
        assert "agents/" in agent.content or "agent_path" in agent.content

    def test_builtin_default_instructions_mention_delegation(self):
        """Test that instructions guide on delegation."""
        agent = get_builtin_default_agent()

        # Instructions should be set
        assert agent.config.instructions

        # Content should explain when/how to delegate
        content_lower = agent.content.lower()
        assert "specialized" in content_lower or "delegate" in content_lower
