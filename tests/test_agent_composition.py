"""Tests for agent composition utilities."""

import pytest

from tsugite.agent_composition import (
    create_delegation_tool,
    create_delegation_tools,
    parse_agent_references,
    resolve_agent_reference,
)


class TestResolveAgentReference:
    def test_resolve_shorthand_from_tsugite(self, tmp_path):
        """Test resolving +name shorthand from .tsugite directory."""
        tsugite_dir = tmp_path / ".tsugite"
        tsugite_dir.mkdir()
        agent_file = tsugite_dir / "test_agent.md"
        agent_file.write_text("---\nname: test\n---\n")

        result = resolve_agent_reference("+test_agent", tmp_path)
        assert result == agent_file.resolve()

    def test_resolve_shorthand_from_agents(self, tmp_path):
        """Test resolving +name shorthand from agents directory."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        agent_file = agents_dir / "helper.md"
        agent_file.write_text("---\nname: helper\n---\n")

        result = resolve_agent_reference("+helper", tmp_path)
        assert result == agent_file.resolve()

    def test_resolve_regular_path(self, tmp_path):
        """Test resolving regular file path."""
        agent_file = tmp_path / "my_agent.md"
        agent_file.write_text("---\nname: my_agent\n---\n")

        result = resolve_agent_reference("my_agent.md", tmp_path)
        assert result == agent_file.resolve()

    def test_resolve_shorthand_not_found(self, tmp_path):
        """Test error when shorthand agent not found."""
        with pytest.raises(ValueError, match="Agent not found"):
            resolve_agent_reference("+nonexistent", tmp_path)

    def test_resolve_path_not_found(self, tmp_path):
        """Test error when file path not found."""
        with pytest.raises(ValueError, match="Agent file not found"):
            resolve_agent_reference("nonexistent.md", tmp_path)


class TestDelegationTools:
    def test_create_delegation_tool(self, tmp_path):
        """Test creating a delegation tool for an agent."""
        agent_file = tmp_path / "helper.md"
        agent_file.write_text(
            """---
name: helper
model: ollama:qwen2.5-coder:7b
tools: []
---

# Helper Agent
Task: {{ user_prompt }}
"""
        )

        tool = create_delegation_tool("helper", agent_file)

        assert tool.name == "spawn_helper"
        assert "helper" in tool.description.lower()
        assert "prompt" in tool.parameters["properties"]

    def test_create_multiple_delegation_tools(self, tmp_path):
        """Test creating delegation tools for multiple agents."""
        # Create test agents
        agent1 = tmp_path / "agent1.md"
        agent1.write_text("---\nname: agent1\n---\nContent")

        agent2 = tmp_path / "agent2.md"
        agent2.write_text("---\nname: agent2\n---\nContent")

        delegation_agents = [("agent1", agent1), ("agent2", agent2)]

        tools = create_delegation_tools(delegation_agents)

        assert len(tools) == 2
        assert tools[0].name == "spawn_agent1"
        assert tools[1].name == "spawn_agent2"


class TestParseAgentReferences:
    def test_parse_single_agent(self, tmp_path):
        """Test parsing single agent reference."""
        agent_file = tmp_path / "test.md"
        agent_file.write_text("---\nname: test\n---\n")

        primary, delegation = parse_agent_references(["test.md"], None, tmp_path)

        assert primary == agent_file.resolve()
        assert delegation == []

    def test_parse_multiple_agents(self, tmp_path):
        """Test parsing multiple agent references."""
        # Create test agents
        tsugite_dir = tmp_path / ".tsugite"
        tsugite_dir.mkdir()

        primary_agent = tmp_path / "primary.md"
        primary_agent.write_text("---\nname: primary\n---\n")

        helper1 = tsugite_dir / "helper1.md"
        helper1.write_text("---\nname: helper1\n---\n")

        helper2 = tsugite_dir / "helper2.md"
        helper2.write_text("---\nname: helper2\n---\n")

        primary, delegation = parse_agent_references(["primary.md", "+helper1", "+helper2"], None, tmp_path)

        assert primary == primary_agent.resolve()
        assert len(delegation) == 2
        assert delegation[0] == ("helper1", helper1.resolve())
        assert delegation[1] == ("helper2", helper2.resolve())

    def test_parse_with_agents_option(self, tmp_path):
        """Test parsing with --with-agents option."""
        tsugite_dir = tmp_path / ".tsugite"
        tsugite_dir.mkdir()

        primary_agent = tmp_path / "primary.md"
        primary_agent.write_text("---\nname: primary\n---\n")

        helper1 = tsugite_dir / "helper1.md"
        helper1.write_text("---\nname: helper1\n---\n")

        helper2 = tsugite_dir / "helper2.md"
        helper2.write_text("---\nname: helper2\n---\n")

        primary, delegation = parse_agent_references(["primary.md"], "helper1,+helper2", tmp_path)

        assert primary == primary_agent.resolve()
        assert len(delegation) == 2
        # Names should have + stripped
        assert delegation[0][0] == "helper1"
        assert delegation[1][0] == "helper2"

    def test_parse_mixed_syntax(self, tmp_path):
        """Test parsing with both positional and --with-agents."""
        tsugite_dir = tmp_path / ".tsugite"
        tsugite_dir.mkdir()

        primary_agent = tsugite_dir / "primary.md"
        primary_agent.write_text("---\nname: primary\n---\n")

        helper1 = tsugite_dir / "helper1.md"
        helper1.write_text("---\nname: helper1\n---\n")

        helper2 = tsugite_dir / "helper2.md"
        helper2.write_text("---\nname: helper2\n---\n")

        helper3 = tsugite_dir / "helper3.md"
        helper3.write_text("---\nname: helper3\n---\n")

        primary, delegation = parse_agent_references(["+primary", "+helper1"], "helper2,+helper3", tmp_path)

        assert primary == primary_agent.resolve()
        assert len(delegation) == 3

    def test_parse_no_agents_error(self, tmp_path):
        """Test error when no agents provided."""
        with pytest.raises(ValueError, match="No agent specified"):
            parse_agent_references([], None, tmp_path)

    def test_parse_name_extraction_from_path(self, tmp_path):
        """Test that agent names are correctly extracted from paths."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        primary_agent = tmp_path / "primary.md"
        primary_agent.write_text("---\nname: primary\n---\n")

        helper = agents_dir / "my_helper.md"
        helper.write_text("---\nname: helper\n---\n")

        primary, delegation = parse_agent_references(["primary.md", "agents/my_helper.md"], None, tmp_path)

        # Name should be extracted from filename without path
        assert delegation[0][0] == "my_helper"
