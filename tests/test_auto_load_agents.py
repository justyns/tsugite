"""Tests for opt-in agent listing via frontmatter (auto_load_agent_list / auto_load_agents)
and the structured list_available_agents() tool.

The default agent no longer carries the full agent roster on every turn; agents call
list_available_agents() on demand, or opt in via frontmatter when they really need
the always-on listing.
"""

from unittest.mock import patch

from tsugite.agent_preparation import AgentPreparer
from tsugite.md_agents import parse_agent_file


class TestListAvailableAgentsTool:
    """The new structured tool returns list[dict] (name/path/description)."""

    def test_returns_structured_list(self, tmp_path, monkeypatch):
        from tsugite.tools.agents import list_available_agents

        monkeypatch.chdir(tmp_path)
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "alpha.md").write_text("---\nname: alpha\ndescription: Alpha agent\n---\n")
        (agents_dir / "beta.md").write_text("---\nname: beta\ndescription: Beta agent\n---\n")

        with patch("tsugite.agent_inheritance.get_global_agents_paths", return_value=[]):
            result = list_available_agents()

        assert isinstance(result, list)
        names = {entry["name"] for entry in result}
        assert {"alpha", "beta"}.issubset(names)
        for entry in result:
            assert set(entry.keys()) >= {"name", "path", "description"}

    def test_returns_empty_list_when_no_agents(self, tmp_path, monkeypatch):
        from tsugite.tools.agents import list_available_agents

        monkeypatch.chdir(tmp_path)
        fake_builtin = tmp_path / "no_such_dir"
        with (
            patch("tsugite.agent_inheritance.get_global_agents_paths", return_value=[]),
            patch("tsugite.agent_inheritance.get_builtin_agents_path", return_value=fake_builtin),
        ):
            result = list_available_agents()

        assert result == []


class TestDefaultAgentNoLongerPrefetchesAgentList:
    """default.md should no longer auto-inject available_agents on every turn."""

    def test_default_md_does_not_prefetch_list_agents(self):
        from tsugite.agent_inheritance import get_builtin_agents_path

        builtin_path = get_builtin_agents_path() / "default.md"
        agent = parse_agent_file(builtin_path)

        prefetch_tools = [p.get("tool") for p in (agent.config.prefetch or [])]
        assert "list_agents" not in prefetch_tools, (
            "default.md still has list_agents in prefetch; the always-on agent listing was removed"
        )


class TestAutoLoadAgentListFlag:
    """`auto_load_agent_list: true` opts the agent back into seeing every agent."""

    def test_flag_true_populates_available_agents(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "helper.md").write_text("---\nname: helper\ndescription: Helps with stuff\n---\n")

        agent_file = tmp_path / "orchestrator.md"
        agent_file.write_text(
            "---\nname: orchestrator\nextends: none\nauto_load_agent_list: true\ntools: []\n---\n\n{{ user_prompt }}\n"
        )

        agent = parse_agent_file(agent_file)
        prepared = AgentPreparer().prepare(agent=agent, prompt="go")

        available = prepared.context.get("available_agents")
        assert available, "expected available_agents to be populated when auto_load_agent_list=true"
        assert "helper" in available

    def test_flag_absent_leaves_available_agents_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "helper.md").write_text("---\nname: helper\ndescription: Helps with stuff\n---\n")

        agent_file = tmp_path / "frugal.md"
        agent_file.write_text("---\nname: frugal\nextends: none\ntools: []\n---\n\n{{ user_prompt }}\n")

        agent = parse_agent_file(agent_file)
        prepared = AgentPreparer().prepare(agent=agent, prompt="go")

        # No prefetch, no flag → either missing or falsy. Either is fine; the {% if %} block suppresses.
        assert not prepared.context.get("available_agents")


class TestAutoLoadAgentsSubset:
    """`auto_load_agents: [name, ...]` opts in but only lists the named agents."""

    def test_filter_includes_only_named_agents(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "alpha.md").write_text("---\nname: alpha\ndescription: A\n---\n")
        (agents_dir / "beta.md").write_text("---\nname: beta\ndescription: B\n---\n")
        (agents_dir / "gamma.md").write_text("---\nname: gamma\ndescription: G\n---\n")

        agent_file = tmp_path / "picky.md"
        agent_file.write_text(
            "---\n"
            "name: picky\n"
            "extends: none\n"
            "auto_load_agents:\n"
            "  - alpha\n"
            "  - gamma\n"
            "tools: []\n"
            "---\n\n"
            "{{ user_prompt }}\n"
        )

        agent = parse_agent_file(agent_file)
        prepared = AgentPreparer().prepare(agent=agent, prompt="go")

        available = prepared.context.get("available_agents") or ""
        assert "alpha" in available
        assert "gamma" in available
        assert "beta" not in available
