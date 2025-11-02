"""Tests for multi-agent mode and visibility system."""

import pytest

from tsugite.agent_runner.helpers import (
    clear_allowed_agents,
    get_allowed_agents,
    set_allowed_agents,
)
from tsugite.cli.helpers import parse_cli_arguments
from tsugite.md_agents import parse_agent_file
from tsugite.tools.agents import spawn_agent


class TestAllowedAgentsStorage:
    """Test module-level allowed agents storage."""

    def test_set_get_clear_allowed_agents(self):
        """Should store, retrieve, and clear allowed agents."""
        clear_allowed_agents()
        assert get_allowed_agents() is None

        set_allowed_agents(["agent1", "agent2"])
        assert get_allowed_agents() == ["agent1", "agent2"]

        clear_allowed_agents()
        assert get_allowed_agents() is None


class TestCLIArgumentParsing:
    """Test CLI argument parsing for multi-agent mode."""

    def test_parse_single_agent(self):
        """Single agent should be captured."""
        agents, prompt, _ = parse_cli_arguments(["+agent", "do something"])
        assert agents == ["+agent"]
        assert prompt == "do something"

    def test_parse_multiple_agents(self):
        """Multiple agents should all be captured."""
        agents, prompt, _ = parse_cli_arguments(["+primary", "+helper1", "+helper2", "task"])
        assert agents == ["+primary", "+helper1", "+helper2"]
        assert prompt == "task"

    def test_parse_no_agent_defaults_to_default(self):
        """No agent should default to +default."""
        agents, prompt, _ = parse_cli_arguments(["just a task"])
        assert agents == ["+default"]
        assert prompt == "just a task"


class TestVisibility:
    """Test visibility and spawnable validation."""

    def test_default_visibility_and_spawnable(self, temp_dir):
        """Agents default to public and spawnable."""
        agent_file = temp_dir / "agent.md"
        agent_file.write_text("""---
name: test_agent
---
Content""")

        agent = parse_agent_file(agent_file)
        assert agent.config.visibility == "public"
        assert agent.config.spawnable is True

    def test_accepts_valid_visibility_values(self, temp_dir):
        """Should accept public, private, internal."""
        for visibility in ["public", "private", "internal"]:
            agent_file = temp_dir / f"{visibility}.md"
            agent_file.write_text(f"""---
name: test_agent
visibility: {visibility}
extends: none
---
Content""")
            agent = parse_agent_file(agent_file)
            assert agent.config.visibility == visibility

    def test_rejects_invalid_visibility(self, temp_dir):
        """Invalid visibility should raise error."""
        agent_file = temp_dir / "invalid.md"
        agent_file.write_text("""---
name: test_agent
visibility: invalid_value
---
Content""")

        with pytest.raises(Exception) as exc:
            parse_agent_file(agent_file)
        assert "visibility must be one of" in str(exc.value)

    def test_spawnable_false(self, temp_dir):
        """spawnable: false should be accepted."""
        agent_file = temp_dir / "agent.md"
        agent_file.write_text("""---
name: test_agent
spawnable: false
extends: none
---
Content""")

        agent = parse_agent_file(agent_file)
        assert agent.config.spawnable is False


class TestSpawnAgentVisibility:
    """Test spawn_agent respects visibility rules."""

    def test_private_agent_blocked_without_permission(self, temp_dir):
        """Private agents should be blocked without permission."""
        agent_file = temp_dir / "private.md"
        agent_file.write_text("""---
name: private_agent
visibility: private
---
Content""")

        from tsugite.agent_runner.helpers import set_current_agent

        set_current_agent("coordinator")
        clear_allowed_agents()

        with pytest.raises(ValueError, match="visibility 'private'"):
            spawn_agent(str(agent_file), "task")

    def test_spawnable_false_blocks_spawning(self, temp_dir):
        """spawnable: false blocks even with permission."""
        agent_file = temp_dir / "blocked.md"
        agent_file.write_text("""---
name: blocked_agent
spawnable: false
---
Content""")

        from tsugite.agent_runner.helpers import set_current_agent

        set_current_agent("coordinator")
        set_allowed_agents(["blocked_agent"])

        with pytest.raises(ValueError, match="non-spawnable"):
            spawn_agent(str(agent_file), "task")

        clear_allowed_agents()

    def test_allowed_list_enforced(self, temp_dir):
        """Only agents in allowed list can spawn."""
        agent_file = temp_dir / "agent.md"
        agent_file.write_text("""---
name: some_agent
---
Content""")

        from tsugite.agent_runner.helpers import set_current_agent

        set_current_agent("coordinator")
        set_allowed_agents(["other_agent"])

        with pytest.raises(ValueError, match="not in the allowed agents list"):
            spawn_agent(str(agent_file), "task")

        clear_allowed_agents()
