"""Tests for agent skills integration."""

import pytest

from tsugite.agent_preparation import AgentPreparer
from tsugite.attachments.base import Attachment, AttachmentContentType
from tsugite.events import EventBus, SkillLoadFailedEvent
from tsugite.md_agents import parse_agent_file
from tsugite.skill_discovery import Skill


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

    def test_prepare_agent_loads_skills(self, agent_with_skills, skill_files, monkeypatch):
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
        skill_names = [s.name for s in prepared.skills]
        assert "skill1" in skill_names
        assert "skill2" in skill_names

    def test_prepare_agent_renders_skill_content(self, agent_with_skills, skill_files, monkeypatch):
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
        for skill in prepared.skills:
            if skill.name == "template_skill":
                template_content = skill.content
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
        skill_names = [s.name for s in prepared.skills]
        assert "nonexistent_skill" not in skill_names


class TestSystemPromptWithSkills:
    """Test that skills are added to system prompt correctly."""

    @pytest.fixture
    def mock_agent_with_skills(self, tmp_path):
        """Create a mock agent with prepared skills."""
        from dataclasses import dataclass
        from typing import List

        @dataclass
        class MockPreparedAgent:
            skills: List[Skill]

        return MockPreparedAgent(
            skills=[
                Skill(name="skill1", content="# Skill 1\nContent for skill 1"),
                Skill(name="skill2", content="# Skill 2\nContent for skill 2"),
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
        skill_blocks = [b for b in content_blocks if isinstance(b, dict) and "<Skill:" in b.get("text", "")]

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
            attachments=[
                Attachment(
                    name="attach1",
                    content="Attachment content",
                    content_type=AttachmentContentType.TEXT,
                    mime_type="text/plain",
                )
            ],
            skills=[Skill(name="skill1", content="Skill content")],
        )

        # Set task in memory
        agent.memory.task = "Test task"

        messages = agent._build_messages()
        content_blocks = messages[0]["content"]

        # Find attachment and skill blocks
        attachment_blocks = [b for b in content_blocks if isinstance(b, dict) and "<Attachment:" in b.get("text", "")]
        skill_blocks = [b for b in content_blocks if isinstance(b, dict) and "<Skill:" in b.get("text", "")]

        # Both should exist
        assert len(attachment_blocks) == 1
        assert len(skill_blocks) == 1

        # Both should have same cache control structure
        assert attachment_blocks[0].get("cache_control") == skill_blocks[0].get("cache_control")


class TestSkillLoadErrorHandling:
    """Test skill load error handling and event emission."""

    def test_skill_not_found_emits_error_event(self, tmp_path, monkeypatch):
        """Test that SkillLoadFailedEvent is emitted when skill not found."""
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

        # Create event bus and capture events
        event_bus = EventBus()
        captured_events = []

        def capture_event(event):
            captured_events.append(event)

        event_bus.subscribe(capture_event)

        preparer = AgentPreparer()
        prepared = preparer.prepare(
            agent=agent,
            prompt="Test task",
            context={},
            event_bus=event_bus,
        )

        # Should have emitted SkillLoadFailedEvent
        error_events = [e for e in captured_events if isinstance(e, SkillLoadFailedEvent)]
        assert len(error_events) == 1
        assert error_events[0].skill_name == "nonexistent_skill"
        assert "not found" in error_events[0].error_message.lower()

        # Skill should not be loaded
        skill_names = [s.name for s in prepared.skills]
        assert "nonexistent_skill" not in skill_names

    def test_multiple_missing_skills_emit_multiple_events(self, tmp_path, monkeypatch):
        """Test that each missing skill emits its own error event."""
        monkeypatch.chdir(tmp_path)

        agent_file = tmp_path / "agent.md"
        agent_file.write_text(
            """---
name: test_agent
extends: none
auto_load_skills:
  - missing_skill1
  - missing_skill2
  - missing_skill3
tools: []
---

# Test Agent
{{ user_prompt }}
"""
        )

        agent = parse_agent_file(agent_file)

        event_bus = EventBus()
        captured_events = []

        def capture_event(event):
            captured_events.append(event)

        event_bus.subscribe(capture_event)

        preparer = AgentPreparer()
        preparer.prepare(
            agent=agent,
            prompt="Test task",
            context={},
            event_bus=event_bus,
        )

        # Should have emitted 3 error events
        error_events = [e for e in captured_events if isinstance(e, SkillLoadFailedEvent)]
        assert len(error_events) == 3

        # Check all skill names are present
        error_skill_names = [e.skill_name for e in error_events]
        assert "missing_skill1" in error_skill_names
        assert "missing_skill2" in error_skill_names
        assert "missing_skill3" in error_skill_names

    def test_partial_skill_load_emits_events_for_failures_only(self, tmp_path, monkeypatch):
        """Test that only failing skills emit error events."""
        monkeypatch.chdir(tmp_path)

        # Create one valid skill
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        valid_skill = skills_dir / "valid_skill.md"
        valid_skill.write_text(
            """---
name: valid_skill
description: Valid skill
---

# Valid Skill
This skill loads successfully.
"""
        )

        agent_file = tmp_path / "agent.md"
        agent_file.write_text(
            """---
name: test_agent
extends: none
auto_load_skills:
  - valid_skill
  - invalid_skill
tools: []
---

# Test Agent
{{ user_prompt }}
"""
        )

        agent = parse_agent_file(agent_file)

        event_bus = EventBus()
        captured_events = []

        def capture_event(event):
            captured_events.append(event)

        event_bus.subscribe(capture_event)

        preparer = AgentPreparer()
        prepared = preparer.prepare(
            agent=agent,
            prompt="Test task",
            context={},
            event_bus=event_bus,
        )

        # Should have emitted 1 error event (only for invalid_skill)
        error_events = [e for e in captured_events if isinstance(e, SkillLoadFailedEvent)]
        assert len(error_events) == 1
        assert error_events[0].skill_name == "invalid_skill"

        # valid_skill should be loaded
        skill_names = [s.name for s in prepared.skills]
        assert "valid_skill" in skill_names
        assert "invalid_skill" not in skill_names

    def test_skill_load_continues_on_error(self, tmp_path, monkeypatch):
        """Test that agent preparation continues even when skills fail to load."""
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
This agent should still work despite skill load failure.
{{ user_prompt }}
"""
        )

        agent = parse_agent_file(agent_file)

        event_bus = EventBus()
        preparer = AgentPreparer()

        # Should not raise exception
        prepared = preparer.prepare(
            agent=agent,
            prompt="Test task",
            context={},
            event_bus=event_bus,
        )

        # Agent should be prepared successfully
        assert prepared is not None
        assert prepared.agent_config.name == "test_agent"
        assert prepared.rendered_prompt is not None

    def test_no_event_bus_skill_load_failures_silent(self, tmp_path, monkeypatch):
        """Test that skill load failures work without event bus (backward compat)."""
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

        # Should not raise even without event_bus
        prepared = preparer.prepare(
            agent=agent,
            prompt="Test task",
            context={},
            event_bus=None,  # No event bus
        )

        assert prepared is not None
        assert len(prepared.skills) == 0
