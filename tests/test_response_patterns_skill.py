"""Tests for response_patterns skill."""

from pathlib import Path

from tsugite.agent_inheritance import get_builtin_agents_path
from tsugite.md_agents import parse_agent_file


def test_response_patterns_skill_exists():
    """response_patterns skill should exist."""
    skill_path = Path(__file__).parent.parent / "tsugite/builtin_skills/response_patterns.md"
    assert skill_path.exists()


def test_response_patterns_skill_has_required_sections():
    """response_patterns skill should contain key sections."""
    skill_path = Path(__file__).parent.parent / "tsugite/builtin_skills/response_patterns.md"
    content = skill_path.read_text()

    assert "Output Channels" in content
    assert "Simple Responses" in content
    assert "Progress Updates" in content
    assert "Anti-patterns" in content
    assert "send_message" in content
    assert "final_answer" in content
    assert "print" in content


def test_default_agent_loads_response_patterns():
    """default agent should auto-load response_patterns skill."""
    agent_path = get_builtin_agents_path() / "default.md"
    agent = parse_agent_file(agent_path)

    assert "response_patterns" in (agent.config.auto_load_skills or [])


def test_default_agent_has_send_message_tool():
    """default agent should have send_message in tools list."""
    agent_path = get_builtin_agents_path() / "default.md"
    agent = parse_agent_file(agent_path)

    assert "send_message" in (agent.config.tools or [])
