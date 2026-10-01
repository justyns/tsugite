"""The default agent runs on a base install (no [web] extra) and only mentions web_search
when it's actually installed."""

from pathlib import Path

import tsugite
from tsugite.md_agents import AgentConfig


def _default_md() -> str:
    return (Path(tsugite.__file__).parent / "builtin_agents" / "default.md").read_text()


def test_default_agent_still_declares_web_search():
    # Kept in the tools list so it's available when [web] is installed; lenient expansion
    # skips it on a base install rather than failing the agent.
    assert "web_search" in _default_md()


def test_strict_tools_frontmatter_defaults_false_and_toggles():
    assert AgentConfig(name="x").strict_tools is False
    assert AgentConfig(name="x", strict_tools=True).strict_tools is True
