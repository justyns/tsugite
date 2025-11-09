"""Tests for built-in agents."""

from pathlib import Path

from tsugite.agent_inheritance import find_agent_file, get_builtin_agents_path
from tsugite.agent_utils import list_local_agents
from tsugite.md_agents import parse_agent_file
from tsugite.renderer import AgentRenderer


class TestBuiltinAgents:
    def test_get_builtin_default_agent(self):
        """Test getting the package-provided default agent."""
        builtin_path = get_builtin_agents_path() / "default.md"
        assert builtin_path.exists(), f"Package agent file not found: {builtin_path}"

        agent = parse_agent_file(builtin_path)

        assert agent.config.name == "default"
        assert agent.config.description
        assert "helpful" in agent.config.instructions.lower()
        # Verify the agent path is within the builtin_agents directory
        assert get_builtin_agents_path() in agent.file_path.parents

    def test_package_agents_in_correct_directory(self):
        """Test that package agents are in the builtin_agents directory."""
        builtin_path = get_builtin_agents_path() / "default.md"
        builtin_dir = get_builtin_agents_path()

        # Package agent should be in the builtin_agents directory
        assert builtin_dir in builtin_path.parents or builtin_path.parent == builtin_dir

        # Non-package path should not be in builtin_agents directory
        custom_path = Path("/tmp/custom.md")
        assert builtin_dir not in custom_path.parents and custom_path.parent != builtin_dir

    def test_find_builtin_agent(self, tmp_path):
        """Test that find_agent_file finds package-provided agents."""
        found = find_agent_file("default", tmp_path)

        assert found is not None
        assert found.exists()
        # Verify it's from the package directory
        builtin_dir = get_builtin_agents_path()
        assert builtin_dir in found.parents or found.parent == builtin_dir

    def test_list_includes_builtin(self, tmp_path):
        """Test that list_local_agents includes package-provided agents."""
        agents = list_local_agents(tmp_path)

        # Package agents should be in the "Built-in" category or mixed with others
        # Check that at least one package agent is present
        all_agents = []
        for category_agents in agents.values():
            all_agents.extend(category_agents)

        builtin_dir = get_builtin_agents_path()
        builtin_agents = [a for a in all_agents if builtin_dir in a.parents or a.parent == builtin_dir]
        assert len(builtin_agents) > 0, "No package agents found in list"


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
        builtin_path = get_builtin_agents_path() / "default.md"
        agent = parse_agent_file(builtin_path)

        assert agent.config.name == "default"
        assert agent.content
        assert "{{ user_prompt }}" in agent.content


class TestBuiltinDefaultAutoDiscovery:
    """Test default agent's auto-discovery features."""

    def test_builtin_default_has_spawn_agent_tool(self):
        """Test that default includes spawn_agent tool."""
        builtin_path = get_builtin_agents_path() / "default.md"
        agent = parse_agent_file(builtin_path)

        assert "spawn_agent" in agent.config.tools

    def test_builtin_default_has_prefetch(self):
        """Test that default has prefetch configured."""
        builtin_path = get_builtin_agents_path() / "default.md"
        agent = parse_agent_file(builtin_path)

        assert agent.config.prefetch is not None
        assert len(agent.config.prefetch) > 0

        # Check that list_agents is in prefetch
        prefetch_tools = [p.get("tool") for p in agent.config.prefetch]
        assert "list_agents" in prefetch_tools

    def test_builtin_default_prefetch_assigns_variable(self):
        """Test that prefetch assigns to available_agents variable."""
        builtin_path = get_builtin_agents_path() / "default.md"
        agent = parse_agent_file(builtin_path)

        list_agents_prefetch = next((p for p in agent.config.prefetch if p.get("tool") == "list_agents"), None)

        assert list_agents_prefetch is not None
        assert list_agents_prefetch.get("assign") == "available_agents"

    def test_builtin_default_content_structure(self):
        """Test that default has delegation instructions."""
        builtin_path = get_builtin_agents_path() / "default.md"
        agent = parse_agent_file(builtin_path)

        # Should have conditional block for available agents
        assert "{% if available_agents %}" in agent.content

        # Should mention delegation
        assert "delegate" in agent.content.lower()
        assert "spawn_agent" in agent.content

        # Should have example usage
        assert "agents/" in agent.content or "agent_path" in agent.content

    def test_builtin_default_instructions_mention_delegation(self):
        """Test that instructions guide on delegation."""
        builtin_path = get_builtin_agents_path() / "default.md"
        agent = parse_agent_file(builtin_path)

        # Instructions should be set
        assert agent.config.instructions

        # Content should explain when/how to delegate
        content_lower = agent.content.lower()
        assert "specialized" in content_lower or "delegate" in content_lower

    def test_builtin_default_web_search_guidelines_conditional(self):
        """Test that web search guidelines only appear when web_search tool is available."""
        builtin_path = get_builtin_agents_path() / "default.md"
        agent = parse_agent_file(builtin_path)
        renderer = AgentRenderer()

        # Base context needed for default template
        base_context = {
            "user_prompt": "test",
            "task_summary": "",
            "text_mode": False,
            "is_interactive": False,
            "available_agents": "",  # From prefetch
            "available_skills": "",  # From prefetch
        }

        # Test WITH web_search tool
        context_with_web_search = {
            **base_context,
            "tools": ["read_file", "write_file", "web_search"],
        }
        rendered_with = renderer.render(agent.content, context_with_web_search)
        assert "Web Search Guidelines" in rendered_with
        assert "web_search(query=" in rendered_with

        # Test WITHOUT web_search tool
        context_without_web_search = {
            **base_context,
            "tools": ["read_file", "write_file"],
        }
        rendered_without = renderer.render(agent.content, context_without_web_search)
        assert "Web Search Guidelines" not in rendered_without
        assert "web_search(query=" not in rendered_without


class TestBuiltinChatAssistant:
    """Test chat-assistant agent configuration."""

    def test_chat_assistant_has_web_search_tool(self):
        """Test that chat assistant includes web_search tool."""
        builtin_path = get_builtin_agents_path() / "chat-assistant.md"
        agent = parse_agent_file(builtin_path)

        assert "web_search" in agent.config.tools

    def test_chat_assistant_has_fetch_text_tool(self):
        """Test that chat assistant includes fetch_text tool."""
        builtin_path = get_builtin_agents_path() / "chat-assistant.md"
        agent = parse_agent_file(builtin_path)

        assert "fetch_text" in agent.config.tools

    def test_chat_assistant_documents_web_search_format(self):
        """Test that chat assistant explains web_search return format."""
        builtin_path = get_builtin_agents_path() / "chat-assistant.md"
        agent = parse_agent_file(builtin_path)

        content = agent.content.lower()

        # Should document the return format with all three fields
        assert "title" in content
        assert "url" in content
        assert "snippet" in content

        # Should show example structure (the [{"... format)
        assert "[{" in content or "returns:" in content

    def test_chat_assistant_warns_against_raw_json(self):
        """Test that chat assistant warns against returning raw JSON."""
        builtin_path = get_builtin_agents_path() / "chat-assistant.md"
        agent = parse_agent_file(builtin_path)

        content = agent.content.lower()

        # Should instruct to format results as readable text and warn against raw dicts/lists
        assert "format" in content, "Chat assistant should mention formatting output"
        assert "readable text" in content or "raw python" in content, (
            "Chat assistant should warn about raw Python objects"
        )

    def test_chat_assistant_mentions_fetch_text_usage(self):
        """Test that chat assistant explains when to use fetch_text."""
        builtin_path = get_builtin_agents_path() / "chat-assistant.md"
        agent = parse_agent_file(builtin_path)

        content = agent.content.lower()

        # Should mention using fetch_text for full page content
        assert "fetch_text" in content
        assert "full" in content or "page" in content or "content" in content
