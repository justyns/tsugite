"""Tests for agent skills integration."""

from pathlib import Path

import pytest

from tsugite.agent_preparation import AgentPreparer
from tsugite.md_agents import parse_agent_file


class TestAgentConfigSkillsField:
    """Test auto_load_skills field in AgentConfig."""

    def test_agent_with_auto_load_skills(self, tmp_path):
        """Test parsing agent with auto_load_skills field."""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text(
            """---
name: test_agent
extends: none
auto_load_skills:
  - skill1
  - skill2
  - skill3
---

# Test Agent
{{ user_prompt }}
"""
        )

        agent = parse_agent_file(agent_file)

        assert hasattr(agent.config, "auto_load_skills")
        assert agent.config.auto_load_skills == ["skill1", "skill2", "skill3"]

    def test_agent_without_auto_load_skills(self, tmp_path):
        """Test parsing agent without auto_load_skills field."""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text(
            """---
name: test_agent
extends: none
---

# Test Agent
{{ user_prompt }}
"""
        )

        agent = parse_agent_file(agent_file)

        assert hasattr(agent.config, "auto_load_skills")
        assert agent.config.auto_load_skills == []  # Default to empty list

    def test_agent_with_empty_auto_load_skills(self, tmp_path):
        """Test parsing agent with empty auto_load_skills."""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text(
            """---
name: test_agent
extends: none
auto_load_skills: []
---

# Test Agent
{{ user_prompt }}
"""
        )

        agent = parse_agent_file(agent_file)

        assert agent.config.auto_load_skills == []


class TestAgentPreparationWithSkills:
    """Test agent preparation with skill loading."""

    @pytest.fixture
    def skill_files(self, tmp_path):
        """Create test skill files."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        skill1 = skills_dir / "skill1.md"
        skill1.write_text(
            """---
name: skill1
description: First skill
---

# Skill 1
This is the first skill content.
"""
        )

        skill2 = skills_dir / "skill2.md"
        skill2.write_text(
            """---
name: skill2
description: Second skill
---

# Skill 2
This is the second skill content.
"""
        )

        return {"skill1": skill1, "skill2": skill2, "dir": skills_dir}

    @pytest.fixture
    def agent_with_skills(self, tmp_path):
        """Create agent file with auto_load_skills."""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text(
            """---
name: test_agent
extends: none
auto_load_skills:
  - skill1
  - skill2
tools: []
---

# Test Agent
{{ user_prompt }}
"""
        )
        return agent_file

    def test_prepare_agent_loads_skills(
        self, agent_with_skills, skill_files, monkeypatch
    ):
        """Test that AgentPreparer loads skills from auto_load_skills."""
        monkeypatch.chdir(agent_with_skills.parent)

        agent = parse_agent_file(agent_with_skills)
        preparer = AgentPreparer()

        prepared = preparer.prepare(
            agent=agent,
            prompt="Test task",
            context={},
        )

        # Check that skills were loaded
        assert hasattr(prepared, "skills")
        assert isinstance(prepared.skills, list)

        # Should have loaded skill1 and skill2
        skill_names = [name for name, _ in prepared.skills]
        assert "skill1" in skill_names
        assert "skill2" in skill_names

    def test_prepare_agent_renders_skill_content(
        self, agent_with_skills, skill_files, monkeypatch
    ):
        """Test that skill content is rendered with Jinja2."""
        monkeypatch.chdir(agent_with_skills.parent)

        # Create skill with template
        template_skill = skill_files["dir"] / "template_skill.md"
        template_skill.write_text(
            """---
name: template_skill
description: Template skill
---

# Template Skill
Current date: {{ today() }}
"""
        )

        # Update agent to load template skill
        agent_file = agent_with_skills.parent / "template_agent.md"
        agent_file.write_text(
            """---
name: template_agent
extends: none
auto_load_skills:
  - template_skill
tools: []
---

# Template Agent
{{ user_prompt }}
"""
        )

        agent = parse_agent_file(agent_file)
        preparer = AgentPreparer()

        prepared = preparer.prepare(
            agent=agent,
            prompt="Test task",
            context={},
        )

        # Find template_skill in loaded skills
        template_content = None
        for name, content in prepared.skills:
            if name == "template_skill":
                template_content = content
                break

        assert template_content is not None
        assert "Current date:" in template_content
        # Template should be rendered (no {{ }})
        assert "{{ today() }}" not in template_content

    def test_prepare_agent_without_skills(self, tmp_path):
        """Test preparing agent without any skills."""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text(
            """---
name: simple_agent
extends: none
tools: []
---

# Simple Agent
{{ user_prompt }}
"""
        )

        agent = parse_agent_file(agent_file)
        preparer = AgentPreparer()

        prepared = preparer.prepare(
            agent=agent,
            prompt="Test task",
            context={},
        )

        # Skills should be empty
        assert hasattr(prepared, "skills")
        assert prepared.skills == []

    def test_prepare_agent_skill_not_found(self, tmp_path, monkeypatch):
        """Test preparing agent when requested skill doesn't exist."""
        monkeypatch.chdir(tmp_path)

        agent_file = tmp_path / "agent.md"
        agent_file.write_text(
            """---
name: test_agent
extends: none
auto_load_skills:
  - nonexistent_skill
tools: []
---

# Test Agent
{{ user_prompt }}
"""
        )

        agent = parse_agent_file(agent_file)
        preparer = AgentPreparer()

        # Should not raise, but skill won't be loaded
        prepared = preparer.prepare(
            agent=agent,
            prompt="Test task",
            context={},
        )

        # Skills list should be empty (skill not found)
        skill_names = [name for name, _ in prepared.skills]
        assert "nonexistent_skill" not in skill_names


class TestSystemPromptWithSkills:
    """Test that skills are added to system prompt correctly."""

    @pytest.fixture
    def mock_agent_with_skills(self, tmp_path):
        """Create a mock agent with prepared skills."""
        from dataclasses import dataclass
        from typing import List, Tuple

        @dataclass
        class MockPreparedAgent:
            skills: List[Tuple[str, str]]

        return MockPreparedAgent(
            skills=[
                ("skill1", "# Skill 1\nContent for skill 1"),
                ("skill2", "# Skill 2\nContent for skill 2"),
            ]
        )

    def test_build_messages_includes_skills(self, mock_agent_with_skills):
        """Test that _build_messages includes skills with cache control."""
        from tsugite.core.agent import TsugiteAgent

        # Create agent instance with correct parameters
        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[],
            instructions="Test instructions",
            max_turns=3,
            attachments=[],
            skills=mock_agent_with_skills.skills,
        )

        # Set task in memory
        agent.memory.task = "Test task"

        messages = agent._build_messages()

        # System message should be a list of blocks
        assert len(messages) > 0
        assert messages[0]["role"] == "system"
        assert isinstance(messages[0]["content"], list)

        # Should have base system prompt + 2 skills
        content_blocks = messages[0]["content"]
        assert len(content_blocks) >= 3  # base + 2 skills

        # Find skill blocks
        skill_blocks = [
            b
            for b in content_blocks
            if isinstance(b, dict) and "<Skill:" in b.get("text", "")
        ]

        assert len(skill_blocks) == 2

        # Check skill formatting
        skill_texts = [b["text"] for b in skill_blocks]
        assert any("<Skill: skill1>" in t for t in skill_texts)
        assert any("<Skill: skill2>" in t for t in skill_texts)

        # Check cache control markers
        for block in skill_blocks:
            assert "cache_control" in block
            assert block["cache_control"]["type"] == "ephemeral"

    def test_skills_formatted_like_attachments(self):
        """Test that skills use same format as attachments."""
        from tsugite.core.agent import TsugiteAgent

        # Create agent instance with correct parameters
        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[],
            instructions="Test instructions",
            max_turns=3,
            attachments=[("attach1", "Attachment content")],
            skills=[("skill1", "Skill content")],
        )

        # Set task in memory
        agent.memory.task = "Test task"

        messages = agent._build_messages()
        content_blocks = messages[0]["content"]

        # Find attachment and skill blocks
        attachment_blocks = [
            b
            for b in content_blocks
            if isinstance(b, dict) and "<Attachment:" in b.get("text", "")
        ]
        skill_blocks = [
            b
            for b in content_blocks
            if isinstance(b, dict) and "<Skill:" in b.get("text", "")
        ]

        # Both should exist
        assert len(attachment_blocks) == 1
        assert len(skill_blocks) == 1

        # Both should have same cache control structure
        assert attachment_blocks[0].get("cache_control") == skill_blocks[0].get(
            "cache_control"
        )
