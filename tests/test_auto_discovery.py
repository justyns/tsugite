"""Integration tests for agent auto-discovery feature."""

from unittest.mock import patch

import pytest

from tsugite.agent_runner import execute_prefetch
from tsugite.tools import tool
from tsugite.tools.agents import discover_agents, format_agents_markdown, list_available_agents


class TestPrefetchListAgents:
    """Test prefetch execution with list_available_agents tool."""

    def test_prefetch_list_available_agents_execution(self, tmp_path, monkeypatch):
        """Test that prefetch can execute list_available_agents and assign result."""
        monkeypatch.chdir(tmp_path)

        # Register list_available_agents tool
        tool(list_available_agents)

        # Create an agent
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        agent_file = agents_dir / "test.md"
        agent_file.write_text("""---
name: test
description: Test agent
---
""")

        # Configure prefetch like default does
        prefetch_config = [{"tool": "list_available_agents", "args": {}, "assign": "available_agents"}]

        # Execute prefetch with mocked global paths
        with patch("tsugite.agent_inheritance.get_global_agents_paths", return_value=[]):
            context = execute_prefetch(prefetch_config)

            # Should have assigned the result
            assert "available_agents" in context
            assert isinstance(context["available_agents"], list)
            assert "test" in [a["name"] for a in context["available_agents"]]

    def test_prefetch_with_package_agents(self, tmp_path, monkeypatch):
        """Test prefetch with only package-provided agents available."""
        monkeypatch.chdir(tmp_path)

        # Register list_available_agents tool
        tool(list_available_agents)

        prefetch_config = [{"tool": "list_available_agents", "args": {}, "assign": "available_agents"}]

        # Execute with mocked global paths
        with patch("tsugite.agent_inheritance.get_global_agents_paths", return_value=[]):
            context = execute_prefetch(prefetch_config)

            # Should include package-provided agents
            assert "available_agents" in context
            names = [a["name"] for a in context["available_agents"]]
            assert "default" in names
            assert "file_searcher" in names


class TestAutoDiscoveryWorkflow:
    """Test the complete auto-discovery workflow."""

    @pytest.fixture
    def agent_setup(self, tmp_path, monkeypatch):
        """Set up test agents."""
        monkeypatch.chdir(tmp_path)

        # Create a code review agent
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        code_review = agents_dir / "code_review.md"
        code_review.write_text("""---
name: code_review
description: Reviews code for security and best practices
tools: [read_file]
---
# Code Review Agent
Task: {{ user_prompt }}
Review the code.
""")

        # Create a research agent
        research = agents_dir / "research.md"
        research.write_text("""---
name: research
description: Researches topics and gathers information
tools: [web_search]
---
# Research Agent
Task: {{ user_prompt }}
Research the topic.
""")

        return {"code_review": code_review, "research": research}

    def test_discover_agents_finds_multiple(self, agent_setup):
        """Test that discover_agents finds all available agents."""
        result = format_agents_markdown(discover_agents())

        assert "code_review" in result
        assert "research" in result
        assert "Reviews code for security" in result
        assert "Researches topics" in result

    def test_conditional_rendering_with_package_agents(self, tmp_path, monkeypatch):
        """Test that package-provided agents are always available."""
        from jinja2 import Template

        monkeypatch.chdir(tmp_path)

        template_text = """
{% if available_agents %}
Agents available: {{ available_agents }}
{% else %}
No agents found.
{% endif %}
"""

        template = Template(template_text)

        # Get agents list with mocked global paths - should still have package-provided agents
        with patch("tsugite.agent_inheritance.get_global_agents_paths", return_value=[]):
            agents_list = format_agents_markdown(discover_agents())

        result = template.render(available_agents=agents_list)

        # Package-provided agents are always available
        assert "Agents available:" in result
        assert "default" in result
        assert "file_searcher" in result


class TestBuiltinDefaultIntegration:
    """Test default agent with real prefetch."""

    def test_builtin_default_prefetch_integration(self, tmp_path, monkeypatch):
        """Default's prefetch covers skills; agents come on-demand via tool."""
        from tsugite.agent_inheritance import get_builtin_agents_path
        from tsugite.md_agents import parse_agent_file

        monkeypatch.chdir(tmp_path)

        default_path = get_builtin_agents_path() / "default.md"
        agent = parse_agent_file(default_path)

        context = execute_prefetch(agent.config.prefetch)

        # Skill discovery still runs through prefetch.
        assert "available_skills" in context
        # Agent listing is no longer pre-injected; agents call list_available_agents() instead.
        assert "available_agents" not in context

    @patch("tsugite.agent_runner.runner.TsugiteAgent")
    @patch("tsugite.core.tools.create_tool_from_tsugite")
    def test_builtin_default_has_delegation_tools(self, mock_create_tool, mock_agent):
        """Default exposes both spawn_agent and the on-demand list_available_agents tool."""
        from tsugite.agent_inheritance import get_builtin_agents_path
        from tsugite.md_agents import parse_agent_file

        default_path = get_builtin_agents_path() / "default.md"
        agent = parse_agent_file(default_path)

        assert "spawn_agent" in agent.config.tools
        assert "list_available_agents" in agent.config.tools
