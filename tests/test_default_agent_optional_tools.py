"""The default agent runs on a base install (no [web] extra) and only mentions web_search
when it's actually installed."""

from pathlib import Path

import tsugite
from tsugite.md_agents import AgentConfig, parse_agent


def _default_md() -> str:
    return (Path(tsugite.__file__).parent / "builtin_agents" / "default.md").read_text()


def test_default_agent_still_declares_web_search():
    # Kept in the tools list so it's available when [web] is installed; lenient expansion
    # skips it on a base install rather than failing the agent.
    assert "web_search" in parse_agent(_default_md()).config.tools


def test_strict_tools_frontmatter_defaults_false_and_toggles():
    assert AgentConfig(name="x").strict_tools is False
    assert AgentConfig(name="x", strict_tools=True).strict_tools is True


def _prepare_default(model_override=None):
    from tsugite.agent_preparation import AgentPreparer
    from tsugite.md_agents import parse_agent_file
    from tsugite.tools import tool
    from tsugite.tools.secrets import get_secret, list_secrets

    tool(parent_only=True)(get_secret)
    tool(parent_only=True)(list_secrets)

    agent = parse_agent_file(Path(tsugite.__file__).parent / "builtin_agents" / "default.md")
    return AgentPreparer().prepare(agent=agent, prompt="hi", context={}, model_override=model_override)


def test_default_system_prompt_does_not_teach_final_answer():
    system = _prepare_default().system_message
    assert "final_answer" not in system
    assert "Variables persist between turns" not in system
    assert "Always respond with Python code blocks" not in system


def test_default_agent_uses_real_skill_tag_and_run():
    text = _default_md()
    assert "<loaded_skill>" not in text
    assert "shell.run(" not in text


def test_default_system_prompt_renders_secrets_section():
    assert "## Secrets" in _prepare_default().rendered_prompt


def test_response_patterns_skill_uses_existing_tools():
    text = (Path(tsugite.__file__).parent / "builtin_skills" / "response-patterns" / "SKILL.md").read_text()
    assert "final_answer" not in text
    assert "search_code" not in text


def test_default_prompt_states_knowledge_cutoff_only_when_known():
    known = _prepare_default("anthropic:claude-haiku-4-5")
    assert "training data runs to about 2025-02-28" in known.rendered_prompt
    assert "Recency" not in _prepare_default("anthropic:no-such-model").rendered_prompt


def test_prepare_step_forwards_model_override_to_knowledge_cutoff():
    from tsugite.agent_runner.runner import _prepare_step, _RunSetup
    from tsugite.options import ExecutionOptions

    agent = parse_agent("---\nname: demo\nextends: none\nmodel: anthropic:no-such-model\n---\nbody")
    setup = _RunSetup(
        exec_options=ExecutionOptions(model_override="anthropic:claude-haiku-4-5"),
        hooks_dir=Path("."),
        hook_message="",
        hook_vars={},
        agent_stem="demo",
    )
    prepared = _prepare_step(agent, "hi", {}, setup)
    assert prepared.context["knowledge_cutoff"] == "2025-02-28"
