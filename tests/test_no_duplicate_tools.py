"""Tests for ensuring no duplicate tools in agent prompts."""


class TestNoDuplicateToolsInPrompt:
    """Test that agent preparation doesn't create duplicate tools."""

    def test_prepare_agent_no_duplicate_task_tools(self, tmp_path):
        """Test that @tasks category + hardcoded task tools doesn't create duplicates."""
        from tsugite.agent_preparation import AgentPreparer
        from tsugite.md_agents import parse_agent_file

        agent_file = tmp_path / "agent.md"
        agent_file.write_text(
            """---
name: test_agent
extends: none
tools:
  - "@tasks"
---

{{ user_prompt }}
"""
        )

        agent = parse_agent_file(agent_file)
        preparer = AgentPreparer()
        prepared = preparer.prepare(agent, "test prompt")

        tool_names = [t.name for t in prepared.tools]

        # Each task tool should appear exactly once
        assert tool_names.count("task_add") == 1
        assert tool_names.count("task_update") == 1
        assert tool_names.count("task_complete") == 1
        assert tool_names.count("task_list") == 1
        assert tool_names.count("task_get") == 1
        assert tool_names.count("spawn_agent") == 1

    def test_prepare_agent_no_duplicates_with_explicit_tools(self, tmp_path):
        """Test that explicit task tools + hardcoded ones doesn't create duplicates."""
        from tsugite.agent_preparation import AgentPreparer
        from tsugite.md_agents import parse_agent_file

        agent_file = tmp_path / "agent.md"
        agent_file.write_text(
            """---
name: test_agent
extends: none
tools:
  - task_add
  - task_update
  - spawn_agent
---

{{ user_prompt }}
"""
        )

        agent = parse_agent_file(agent_file)
        preparer = AgentPreparer()
        prepared = preparer.prepare(agent, "test prompt")

        tool_names = [t.name for t in prepared.tools]

        # Explicit tools should appear exactly once
        assert tool_names.count("task_add") == 1
        assert tool_names.count("task_update") == 1
        assert tool_names.count("spawn_agent") == 1

        # Other task tools added by hardcoded list
        assert tool_names.count("task_complete") == 1
        assert tool_names.count("task_list") == 1
        assert tool_names.count("task_get") == 1

    def test_prepare_agent_no_duplicates_empty_tools(self, tmp_path):
        """Test that agents with no tools still get task tools exactly once."""
        from tsugite.agent_preparation import AgentPreparer
        from tsugite.md_agents import parse_agent_file

        agent_file = tmp_path / "agent.md"
        agent_file.write_text(
            """---
name: test_agent
extends: none
---

{{ user_prompt }}
"""
        )

        agent = parse_agent_file(agent_file)
        preparer = AgentPreparer()
        prepared = preparer.prepare(agent, "test prompt")

        tool_names = [t.name for t in prepared.tools]

        # Task tools should be present exactly once (added by hardcoded list)
        assert tool_names.count("task_add") == 1
        assert tool_names.count("task_update") == 1
        assert tool_names.count("task_complete") == 1
        assert tool_names.count("task_list") == 1
        assert tool_names.count("task_get") == 1
        assert tool_names.count("spawn_agent") == 1
