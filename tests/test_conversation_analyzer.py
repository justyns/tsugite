"""Integration tests for conversation analyzer agent."""

from pathlib import Path

import pytest

from tsugite.agent_preparation import AgentPreparer
from tsugite.md_agents import parse_agent_file


@pytest.fixture
def history_tools(reset_tool_registry):
    """Register history tools for testing."""
    from tsugite.tools import tool
    from tsugite.tools.history import list_conversations, read_conversation

    tool(read_conversation)
    tool(list_conversations)


@pytest.fixture
def conversation_analyzer_path():
    """Get path to conversation analyzer agent."""
    return Path(__file__).parent.parent / "tsugite" / "builtin_agents" / "conversation_analyzer.md"


def test_conversation_analyzer_exists(conversation_analyzer_path):
    """Test that conversation analyzer agent file exists."""
    assert conversation_analyzer_path.exists(), "conversation_analyzer.md should exist in builtin_agents"


def test_conversation_analyzer_parses(conversation_analyzer_path):
    """Test that conversation analyzer agent parses correctly."""
    agent = parse_agent_file(conversation_analyzer_path)

    # Verify basic structure
    assert agent.config.name == "conversation_analyzer"
    assert agent.config.extends == "none"
    assert "anthropic" in agent.config.model.lower() or "claude" in agent.config.model.lower()
    assert agent.config.max_turns >= 5

    # Verify tools
    assert "read_conversation" in agent.config.tools
    assert "list_conversations" in agent.config.tools

    # Verify content is not empty
    assert len(agent.content.strip()) > 0
    assert "conversation" in agent.content.lower()
    assert "analysis" in agent.content.lower()


def test_conversation_analyzer_prepares(conversation_analyzer_path, history_tools):
    """Test that conversation analyzer agent can be prepared for execution."""
    agent = parse_agent_file(conversation_analyzer_path)

    # Prepare agent with a sample conversation ID
    preparer = AgentPreparer()
    prepared = preparer.prepare(
        agent=agent,
        prompt="20251110_120000_chat_abc123",
        context={},
    )

    # Verify prepared agent
    assert prepared is not None
    assert prepared.agent == agent
    assert prepared.agent_config == agent.config
    assert prepared.user_message.strip() != ""

    # Verify tools are expanded (tools are Tool objects with .name attribute)
    tool_names = [tool.name for tool in prepared.tools]
    assert "read_conversation" in tool_names
    assert "list_conversations" in tool_names

    # Verify system message includes analysis guidance
    system_msg = prepared.system_message.lower()
    assert "conversation" in system_msg or "analysis" in system_msg


def test_conversation_analyzer_has_analysis_sections(conversation_analyzer_path):
    """Test that conversation analyzer includes key analysis sections."""
    agent = parse_agent_file(conversation_analyzer_path)
    content = agent.content

    # Check for key analysis categories mentioned in requirements
    required_sections = [
        "efficiency",
        "correctness",
        "proactivity",
        "tool usage",
        "skill",
        "subagent",
    ]

    content_lower = content.lower()
    for section in required_sections:
        assert section in content_lower, f"Agent should mention '{section}' in analysis framework"


def test_conversation_analyzer_has_severity_levels(conversation_analyzer_path):
    """Test that conversation analyzer includes severity indicators."""
    agent = parse_agent_file(conversation_analyzer_path)
    content = agent.content

    # Check for severity indicators
    assert "🔴" in content or "critical" in content.lower()
    assert "🟡" in content or "medium" in content.lower()
    assert "🟢" in content or "minor" in content.lower()


def test_conversation_analyzer_has_output_format(conversation_analyzer_path):
    """Test that conversation analyzer specifies structured output format."""
    agent = parse_agent_file(conversation_analyzer_path)
    content = agent.content

    # Check for output structure guidance
    content_lower = content.lower()
    assert "report" in content_lower
    assert "recommendations" in content_lower
    assert "overview" in content_lower or "summary" in content_lower


def test_conversation_analyzer_description(conversation_analyzer_path):
    """Test that conversation analyzer has appropriate description."""
    agent = parse_agent_file(conversation_analyzer_path)

    description = agent.config.description.lower()
    assert "conversation" in description or "analyze" in description
    assert len(agent.config.description) > 20, "Description should be meaningful"
