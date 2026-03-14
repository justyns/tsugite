"""Tests for run_if guard in agent preparation."""

from pathlib import Path

from tsugite.agent_preparation import AgentPreparer
from tsugite.md_agents import Agent, AgentConfig


def _make_agent(run_if: str, prefetch=None) -> Agent:
    config = AgentConfig(name="test", run_if=run_if, prefetch=prefetch or [])
    return Agent(config=config, content="Do something.", file_path=Path("test.md"))


class TestRunIfGuard:
    def test_truthy_expression_not_skipped(self):
        agent = _make_agent(run_if="True")
        preparer = AgentPreparer()
        result = preparer.prepare(agent, prompt="go")
        assert not result.skipped

    def test_falsy_expression_skipped(self):
        agent = _make_agent(run_if="False")
        preparer = AgentPreparer()
        result = preparer.prepare(agent, prompt="go")
        assert result.skipped
        assert "false" in result.skip_reason.lower()

    def test_numeric_zero_skipped(self):
        agent = _make_agent(run_if="0")
        preparer = AgentPreparer()
        result = preparer.prepare(agent, prompt="go")
        assert result.skipped

    def test_none_expression_skipped(self):
        agent = _make_agent(run_if="None")
        preparer = AgentPreparer()
        result = preparer.prepare(agent, prompt="go")
        assert result.skipped

    def test_truthy_string_not_skipped(self):
        agent = _make_agent(run_if="'hello'")
        preparer = AgentPreparer()
        result = preparer.prepare(agent, prompt="go")
        assert not result.skipped

    def test_bad_expression_skipped_with_error(self):
        agent = _make_agent(run_if="undefined_var > 0")
        preparer = AgentPreparer()
        result = preparer.prepare(agent, prompt="go")
        assert result.skipped
        assert "error" in result.skip_reason.lower()

    def test_file_exists_helper(self, tmp_path):
        trigger = tmp_path / "trigger.flag"
        # File doesn't exist → skipped
        agent = _make_agent(run_if=f"file_exists('{trigger}')")
        preparer = AgentPreparer()
        result = preparer.prepare(agent, prompt="go")
        assert result.skipped

        # Create the file → not skipped
        trigger.touch()
        result = preparer.prepare(agent, prompt="go")
        assert not result.skipped

    def test_no_run_if_not_skipped(self):
        config = AgentConfig(name="test")
        agent = Agent(config=config, content="Do something.", file_path=Path("test.md"))
        preparer = AgentPreparer()
        result = preparer.prepare(agent, prompt="go")
        assert not result.skipped

    def test_skipped_agent_has_empty_tools(self):
        agent = _make_agent(run_if="False")
        preparer = AgentPreparer()
        result = preparer.prepare(agent, prompt="go")
        assert result.skipped
        assert result.tools == []
        assert result.system_message == ""

    def test_prefetch_variable_in_run_if(self):
        agent = _make_agent(run_if="count | int > 0")
        preparer = AgentPreparer()
        # Inject prefetch context manually via context param
        result = preparer.prepare(agent, prompt="go", context={"count": "5"})
        assert not result.skipped

    def test_prefetch_variable_zero_skips(self):
        agent = _make_agent(run_if="count | int > 0")
        preparer = AgentPreparer()
        result = preparer.prepare(agent, prompt="go", context={"count": "0"})
        assert result.skipped
