"""Tests for agent-defined attachments."""

from tsugite.attachments import add_attachment
from tsugite.md_agents import parse_agent


class TestAgentAttachments:
    """Test attachment support in agent definitions."""

    def test_agent_with_attachments_field(self):
        """Test parsing agent with attachments field."""
        agent_text = """---
name: test_agent
model: openai:gpt-4o-mini
tools: []
attachments:
  - coding-standards
  - api-docs
---

Task: {{ user_prompt }}
"""
        agent = parse_agent(agent_text)

        assert agent.config.attachments == ["coding-standards", "api-docs"]

    def test_agent_without_attachments_field(self):
        """Test parsing agent without attachments field."""
        agent_text = """---
name: test_agent
model: openai:gpt-4o-mini
tools: []
---

Task: {{ user_prompt }}
"""
        agent = parse_agent(agent_text)

        assert agent.config.attachments == []

    def test_agent_with_empty_attachments(self):
        """Test parsing agent with empty attachments list."""
        agent_text = """---
name: test_agent
model: openai:gpt-4o-mini
tools: []
attachments: []
---

Task: {{ user_prompt }}
"""
        agent = parse_agent(agent_text)

        assert agent.config.attachments == []

    def test_agent_with_single_attachment(self):
        """Test parsing agent with single attachment."""
        agent_text = """---
name: test_agent
model: openai:gpt-4o-mini
tools: []
attachments:
  - style-guide
---

Task: {{ user_prompt }}
"""
        agent = parse_agent(agent_text)

        assert agent.config.attachments == ["style-guide"]

    def test_agent_attachments_in_agent_info(self, tmp_path, monkeypatch):
        """Test that attachments appear in get_agent_info."""
        from tsugite.agent_runner import get_agent_info

        monkeypatch.setenv("HOME", str(tmp_path))

        agent_text = """---
name: test_agent
model: openai:gpt-4o-mini
tools: []
attachments:
  - coding-standards
  - security-guide
---

Task: {{ user_prompt }}
"""
        agent_file = tmp_path / "test_agent.md"
        agent_file.write_text(agent_text)

        agent_info = get_agent_info(agent_file)

        assert "attachments" in agent_info
        assert agent_info["attachments"] == ["coding-standards", "security-guide"]

    def test_agent_attachments_integration(self, tmp_path, monkeypatch):
        """Test full integration with attachment resolution."""
        from tsugite.utils import resolve_attachments

        monkeypatch.setenv("HOME", str(tmp_path))

        # Create an inline attachment
        add_attachment("style-guide", source="inline", content="Use tabs for indentation")

        # Create an agent with that attachment
        agent_text = """---
name: code_reviewer
model: openai:gpt-4o-mini
tools: []
attachments:
  - style-guide
---

You are a code reviewer.

Task: {{ user_prompt }}
"""
        agent = parse_agent(agent_text)

        # Resolve agent's attachments
        resolved = resolve_attachments(agent.config.attachments)

        assert len(resolved) == 1
        assert resolved[0].name == "style-guide"
        assert resolved[0].content == "Use tabs for indentation"
