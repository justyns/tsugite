"""Integration tests for agent auto-discovery feature."""

from unittest.mock import patch

import pytest

from tsugite.agent_runner import execute_prefetch
from tsugite.tools import tool
from tsugite.tools.agents import list_agents, spawn_agent


class TestPrefetchListAgents:
    """Test prefetch execution with list_agents tool."""

    def test_prefetch_list_agents_execution(self, tmp_path, monkeypatch):
        """Test that prefetch can execute list_agents and assign result."""
        monkeypatch.chdir(tmp_path)

        # Register list_agents tool
        tool(list_agents)

        # Create an agent
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        agent_file = agents_dir / "test.md"
        agent_file.write_text("""---
name: test
description: Test agent
---
""")

        # Configure prefetch like builtin-default does
        prefetch_config = [{"tool": "list_agents", "args": {}, "assign": "available_agents"}]

        # Execute prefetch with mocked global paths
        with patch("tsugite.agent_inheritance.get_global_agents_paths", return_value=[]):
            context = execute_prefetch(prefetch_config)

            # Should have assigned the result
            assert "available_agents" in context
            assert isinstance(context["available_agents"], str)
            assert "test" in context["available_agents"]

    def test_prefetch_empty_agents_list(self, tmp_path, monkeypatch):
        """Test prefetch with no agents available."""
        monkeypatch.chdir(tmp_path)

        # Register list_agents tool
        tool(list_agents)

        prefetch_config = [{"tool": "list_agents", "args": {}, "assign": "available_agents"}]

        # Execute with mocked global paths
        with patch("tsugite.agent_inheritance.get_global_agents_paths", return_value=[]):
            context = execute_prefetch(prefetch_config)

            # Should assign empty string
            assert "available_agents" in context
            assert context["available_agents"] == ""


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

    def test_list_agents_discovers_multiple(self, agent_setup):
        """Test that list_agents discovers all available agents."""
        result = list_agents()

        assert "code_review" in result
        assert "research" in result
        assert "Reviews code for security" in result
        assert "Researches topics" in result

    def test_spawn_agent_can_execute_discovered_agent(self, agent_setup):
        """Test that spawn_agent can execute a discovered agent."""
        # Get the list first (simulating what LLM would do)
        agents_list = list_agents()
        assert "code_review" in agents_list

        # Now spawn it
        with patch("tsugite.agent_runner.run_agent") as mock_run:
            mock_run.return_value = "Code review complete"

            result = spawn_agent(agent_path="agents/code_review.md", prompt="Review authentication.py")

            assert result == "Code review complete"
            mock_run.assert_called_once()

    def test_conditional_rendering_with_agents(self, agent_setup):
        """Test that {% if available_agents %} works correctly."""
        from jinja2 import Template

        # Simulate builtin-default's conditional block
        template_text = """
{% if available_agents %}
## Available Agents
{{ available_agents }}
You can delegate using spawn_agent()
{% endif %}

## Task
{{ user_prompt }}
"""

        template = Template(template_text)

        # With agents
        agents_list = list_agents()
        result_with = template.render(available_agents=agents_list, user_prompt="Do work")

        assert "Available Agents" in result_with
        assert "code_review" in result_with
        assert "spawn_agent()" in result_with

        # Without agents
        result_without = template.render(available_agents="", user_prompt="Do work")

        assert "Available Agents" not in result_without
        assert "spawn_agent()" not in result_without

    def test_conditional_rendering_no_agents(self, tmp_path, monkeypatch):
        """Test conditional rendering when no agents are available."""
        from jinja2 import Template

        monkeypatch.chdir(tmp_path)

        template_text = """
{% if available_agents %}
Agents available: {{ available_agents }}
{% else %}
No specialized agents found.
{% endif %}
"""

        template = Template(template_text)

        # Get agents list with mocked global paths
        with patch("tsugite.agent_inheritance.get_global_agents_paths", return_value=[]):
            agents_list = list_agents()  # Empty

        result = template.render(available_agents=agents_list)

        # Empty string is falsy in Jinja2
        assert "No specialized agents found" in result


class TestBuiltinDefaultIntegration:
    """Test builtin-default agent with real prefetch."""

    def test_builtin_default_prefetch_integration(self, tmp_path, monkeypatch):
        """Test that builtin-default's prefetch config works."""
        from tsugite.builtin_agents import get_builtin_default_agent

        monkeypatch.chdir(tmp_path)

        # Create a test agent
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        test_agent = agents_dir / "helper.md"
        test_agent.write_text("""---
name: helper
description: Helps with tasks
---
""")

        # Get builtin-default agent
        agent = get_builtin_default_agent()

        # Execute its prefetch
        context = execute_prefetch(agent.config.prefetch)

        # Should have available_agents
        assert "available_agents" in context

        # Should include our test agent
        if context["available_agents"]:  # Only if agents found
            assert "helper" in context["available_agents"]

    @patch("tsugite.agent_runner.runner.TsugiteAgent")
    @patch("tsugite.core.tools.create_tool_from_tsugite")
    def test_builtin_default_has_delegation_tools(self, mock_create_tool, mock_agent):
        """Test that builtin-default provides delegation tools."""
        from tsugite.builtin_agents import get_builtin_default_agent

        agent = get_builtin_default_agent()

        # Should have spawn_agent in tools
        assert "spawn_agent" in agent.config.tools

        # Prefetch should be configured
        assert agent.config.prefetch
        assert any(p.get("tool") == "list_agents" for p in agent.config.prefetch)
