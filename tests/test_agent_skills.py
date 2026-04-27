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
        agent_file.write_text("""---
name: test_agent
extends: none
auto_load_skills:
  - skill1
  - skill2
  - skill3
---

# Test Agent
{{ user_prompt }}
""")

        agent = parse_agent_file(agent_file)

        assert hasattr(agent.config, "auto_load_skills")
        assert agent.config.auto_load_skills == ["skill1", "skill2", "skill3"]

    def test_agent_without_auto_load_skills(self, tmp_path):
        """Test parsing agent without auto_load_skills field."""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text("""---
name: test_agent
extends: none
---

# Test Agent
{{ user_prompt }}
""")

        agent = parse_agent_file(agent_file)

        assert hasattr(agent.config, "auto_load_skills")
        assert agent.config.auto_load_skills == []  # Default to empty list

    def test_agent_with_empty_auto_load_skills(self, tmp_path):
        """Test parsing agent with empty auto_load_skills."""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text("""---
name: test_agent
extends: none
auto_load_skills: []
---

# Test Agent
{{ user_prompt }}
""")

        agent = parse_agent_file(agent_file)

        assert agent.config.auto_load_skills == []


class TestAgentConfigSkillPathsField:
    """Test skill_paths field in AgentConfig."""

    def test_agent_with_skill_paths(self, tmp_path):
        agent_file = tmp_path / "agent.md"
        agent_file.write_text("""---
name: test_agent
extends: none
skill_paths:
  - ~/my-skills
  - /opt/team-skills
---

# Test Agent
{{ user_prompt }}
""")
        agent = parse_agent_file(agent_file)
        assert agent.config.skill_paths == ["~/my-skills", "/opt/team-skills"]

    def test_agent_without_skill_paths(self, tmp_path):
        agent_file = tmp_path / "agent.md"
        agent_file.write_text("""---
name: test_agent
extends: none
---

# Test Agent
{{ user_prompt }}
""")
        agent = parse_agent_file(agent_file)
        assert agent.config.skill_paths == []


class TestAgentPreparationWithSkills:
    """Test agent preparation with skill loading."""

    @pytest.fixture
    def skill_files(self, tmp_path):
        """Create directory-based test skills."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        skill1_dir = skills_dir / "skill1"
        skill1_dir.mkdir()
        (skill1_dir / "SKILL.md").write_text("""---
name: skill1
description: First skill
---

# Skill 1
This is the first skill content.
""")

        skill2_dir = skills_dir / "skill2"
        skill2_dir.mkdir()
        (skill2_dir / "SKILL.md").write_text("""---
name: skill2
description: Second skill
---

# Skill 2
This is the second skill content.
""")

        return {"skill1": skill1_dir, "skill2": skill2_dir, "dir": skills_dir}

    @pytest.fixture
    def agent_with_skills(self, tmp_path):
        """Create agent file with auto_load_skills."""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text("""---
name: test_agent
extends: none
auto_load_skills:
  - skill1
  - skill2
tools: []
---

# Test Agent
{{ user_prompt }}
""")
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
        template_dir = skill_files["dir"] / "template_skill"
        template_dir.mkdir()
        (template_dir / "SKILL.md").write_text("""---
name: template_skill
description: Template skill
---

# Template Skill
Current date: {{ today() }}
""")

        # Update agent to load template skill
        agent_file = agent_with_skills.parent / "template_agent.md"
        agent_file.write_text("""---
name: template_agent
extends: none
auto_load_skills:
  - template_skill
tools: []
---

# Template Agent
{{ user_prompt }}
""")

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
        agent_file.write_text("""---
name: simple_agent
extends: none
tools: []
---

# Simple Agent
{{ user_prompt }}
""")

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
        agent_file.write_text("""---
name: test_agent
extends: none
auto_load_skills:
  - nonexistent_skill
tools: []
---

# Test Agent
{{ user_prompt }}
""")

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

    def test_suppressed_skill_is_not_loaded(self, agent_with_skills, skill_files, monkeypatch):
        """Skills present in context['suppressed_skills'] are skipped even if
        listed in auto_load_skills."""
        monkeypatch.chdir(agent_with_skills.parent)

        agent = parse_agent_file(agent_with_skills)
        preparer = AgentPreparer()

        prepared = preparer.prepare(
            agent=agent,
            prompt="Test task",
            context={"suppressed_skills": ["skill1"]},
        )

        skill_names = [s.name for s in prepared.skills]
        assert "skill1" not in skill_names
        assert "skill2" in skill_names

    def test_multiple_suppressed_skills(self, agent_with_skills, skill_files, monkeypatch):
        """Suppressing every auto_load skill leaves prepared.skills empty."""
        monkeypatch.chdir(agent_with_skills.parent)

        agent = parse_agent_file(agent_with_skills)
        preparer = AgentPreparer()

        prepared = preparer.prepare(
            agent=agent,
            prompt="Test task",
            context={"suppressed_skills": ["skill1", "skill2"]},
        )

        assert [s.name for s in prepared.skills] == []

    def test_suppressed_skills_accepts_any_iterable(self, agent_with_skills, skill_files, monkeypatch):
        """Context may arrive with set/tuple/list; all should filter correctly."""
        monkeypatch.chdir(agent_with_skills.parent)

        agent = parse_agent_file(agent_with_skills)
        preparer = AgentPreparer()

        for shape in ({"skill1"}, ("skill1",), ["skill1"]):
            prepared = preparer.prepare(
                agent=agent,
                prompt="Test task",
                context={"suppressed_skills": shape},
            )
            names = [s.name for s in prepared.skills]
            assert "skill1" not in names, f"failed for shape {type(shape).__name__}"
            assert "skill2" in names

    def test_missing_suppressed_skills_key(self, agent_with_skills, skill_files, monkeypatch):
        """Context without the key behaves as if nothing is suppressed."""
        monkeypatch.chdir(agent_with_skills.parent)

        agent = parse_agent_file(agent_with_skills)
        preparer = AgentPreparer()

        prepared = preparer.prepare(agent=agent, prompt="Test task", context={})
        names = {s.name for s in prepared.skills}
        assert names == {"skill1", "skill2"}

    def test_none_suppressed_skills(self, agent_with_skills, skill_files, monkeypatch):
        """Explicit None behaves the same as an absent key (no suppression)."""
        monkeypatch.chdir(agent_with_skills.parent)

        agent = parse_agent_file(agent_with_skills)
        preparer = AgentPreparer()

        prepared = preparer.prepare(
            agent=agent,
            prompt="Test task",
            context={"suppressed_skills": None},
        )
        names = {s.name for s in prepared.skills}
        assert names == {"skill1", "skill2"}

    def test_empty_suppressed_skills(self, agent_with_skills, skill_files, monkeypatch):
        """An empty iterable suppresses nothing."""
        monkeypatch.chdir(agent_with_skills.parent)

        agent = parse_agent_file(agent_with_skills)
        preparer = AgentPreparer()

        prepared = preparer.prepare(
            agent=agent,
            prompt="Test task",
            context={"suppressed_skills": []},
        )
        names = {s.name for s in prepared.skills}
        assert names == {"skill1", "skill2"}

    def test_sticky_skill_reloads_across_turns(self, tmp_path, monkeypatch):
        """Sticky skills carry over into the next preparer.prepare() call."""
        monkeypatch.chdir(tmp_path)

        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my_skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my_skill\ndescription: Sticky test\nttl: 5\n---\nBody.\n")

        agent_file = tmp_path / "agent.md"
        agent_file.write_text("---\nname: test_agent\nextends: none\ntools: []\n---\n\n{{ user_prompt }}\n")
        agent = parse_agent_file(agent_file)
        preparer = AgentPreparer()

        # Counter 2 out of ttl 5 -> loads, not expiring yet.
        prepared = preparer.prepare(
            agent=agent,
            prompt="unrelated",
            context={"sticky_skills": {"my_skill": 2}, "skill_ttl_default": 10},
        )
        assert "my_skill" in [s.name for s in prepared.skills]
        assert prepared.expiring_skills == {}

    def test_sticky_skill_expires_when_counter_exceeds_ttl(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my_skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my_skill\ndescription: Test\nttl: 3\n---\nBody.\n")

        agent_file = tmp_path / "agent.md"
        agent_file.write_text("---\nname: test_agent\nextends: none\ntools: []\n---\n\n{{ user_prompt }}\n")
        agent = parse_agent_file(agent_file)
        preparer = AgentPreparer()

        prepared = preparer.prepare(
            agent=agent,
            prompt="unrelated",
            context={"sticky_skills": {"my_skill": 4}, "skill_ttl_default": 10},
        )
        assert "my_skill" not in [s.name for s in prepared.skills]
        assert "my_skill" in prepared.context.get("_expired_sticky_skills", [])

    def test_expiring_skills_populated_on_last_turn(self, tmp_path, monkeypatch):
        """counter == ttl -> 0 turns remaining; still loads, but flagged expiring."""
        monkeypatch.chdir(tmp_path)

        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my_skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my_skill\ndescription: Test\nttl: 3\n---\nBody.\n")

        agent_file = tmp_path / "agent.md"
        agent_file.write_text("---\nname: test_agent\nextends: none\ntools: []\n---\n\n{{ user_prompt }}\n")
        agent = parse_agent_file(agent_file)
        preparer = AgentPreparer()

        prepared = preparer.prepare(
            agent=agent,
            prompt="unrelated",
            context={"sticky_skills": {"my_skill": 3}, "skill_ttl_default": 10},
        )
        assert "my_skill" in [s.name for s in prepared.skills]
        assert prepared.expiring_skills == {"my_skill": 0}

    def test_config_default_ttl_used_when_frontmatter_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my_skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my_skill\ndescription: Test\n---\nBody.\n")

        agent_file = tmp_path / "agent.md"
        agent_file.write_text("---\nname: test_agent\nextends: none\ntools: []\n---\n\n{{ user_prompt }}\n")
        agent = parse_agent_file(agent_file)
        preparer = AgentPreparer()

        # Default ttl 2, counter 3 -> expired
        prepared = preparer.prepare(
            agent=agent,
            prompt="unrelated",
            context={"sticky_skills": {"my_skill": 3}, "skill_ttl_default": 2},
        )
        assert "my_skill" not in [s.name for s in prepared.skills]
        assert "my_skill" in prepared.context.get("_expired_sticky_skills", [])

    def test_ttl_zero_means_never_expire(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my_skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my_skill\ndescription: Test\nttl: 0\n---\nBody.\n")

        agent_file = tmp_path / "agent.md"
        agent_file.write_text("---\nname: test_agent\nextends: none\ntools: []\n---\n\n{{ user_prompt }}\n")
        agent = parse_agent_file(agent_file)
        preparer = AgentPreparer()

        # Counter is huge, but ttl == 0 -> never drops.
        prepared = preparer.prepare(
            agent=agent,
            prompt="unrelated",
            context={"sticky_skills": {"my_skill": 999}, "skill_ttl_default": 10},
        )
        assert "my_skill" in [s.name for s in prepared.skills]
        assert prepared.expiring_skills == {}

    def test_triggered_skill_names_surfaced_for_daemon(self, agent_with_skills, skill_files, monkeypatch):
        """Daemon needs to know which skills matched triggers so it can mark them sticky."""
        monkeypatch.chdir(agent_with_skills.parent)

        skill_dir = skill_files["dir"] / "weather"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: weather\ndescription: Weather\ntriggers:\n  - forecast\n---\nBody.\n"
        )
        agent = parse_agent_file(agent_with_skills)
        preparer = AgentPreparer()

        prepared = preparer.prepare(
            agent=agent,
            prompt="what is the forecast",
            context={},
        )
        assert "weather" in prepared.context.get("_triggered_skill_names", [])

    def test_auto_loaded_skill_names_surfaced_for_daemon(self, agent_with_skills, skill_files, monkeypatch):
        """Daemon uses this list to exempt auto-loaded skills from sticky tracking."""
        monkeypatch.chdir(agent_with_skills.parent)

        agent = parse_agent_file(agent_with_skills)
        preparer = AgentPreparer()
        prepared = preparer.prepare(agent=agent, prompt="Test", context={})
        assert set(prepared.context.get("_auto_loaded_skill_names", [])) == {"skill1", "skill2"}

    def test_suppressing_unknown_skill_does_not_affect_others(self, agent_with_skills, skill_files, monkeypatch):
        """Suppressing a name that isn't in auto_load_skills is harmless."""
        monkeypatch.chdir(agent_with_skills.parent)

        agent = parse_agent_file(agent_with_skills)
        preparer = AgentPreparer()

        prepared = preparer.prepare(
            agent=agent,
            prompt="Test task",
            context={"suppressed_skills": ["does-not-exist"]},
        )
        names = {s.name for s in prepared.skills}
        assert names == {"skill1", "skill2"}

    def test_suppression_does_not_emit_skill_loaded_event(self, agent_with_skills, skill_files, monkeypatch):
        """When a skill is suppressed it should not fire SkillLoadedEvent.

        SkillLoadedEvent is emitted via the ui_context event bus (not the
        preparer's event_bus arg), so we set it there.
        """
        from tsugite import ui_context
        from tsugite.events import EventBus, SkillLoadedEvent

        monkeypatch.chdir(agent_with_skills.parent)

        agent = parse_agent_file(agent_with_skills)
        preparer = AgentPreparer()
        bus = EventBus()
        events = []
        bus.subscribe(events.append)

        token = ui_context._event_bus_var.set(bus)
        try:
            preparer.prepare(
                agent=agent,
                prompt="Test task",
                context={"suppressed_skills": ["skill1"]},
            )
        finally:
            ui_context._event_bus_var.reset(token)

        loaded_names = {e.skill_name for e in events if isinstance(e, SkillLoadedEvent)}
        assert "skill1" not in loaded_names
        assert "skill2" in loaded_names

    def test_suppressed_triggered_skill_is_not_loaded(self, skill_files, tmp_path, monkeypatch):
        """Suppression also filters out trigger-matched skills, not only
        auto_load_skills."""
        monkeypatch.chdir(tmp_path)

        triggered_dir = skill_files["dir"] / "triggered_skill"
        triggered_dir.mkdir()
        (triggered_dir / "SKILL.md").write_text("""---
name: triggered_skill
description: A skill with trigger words
triggers:
  - magicword
---

Triggered body.
""")

        agent_file = tmp_path / "agent.md"
        agent_file.write_text("""---
name: test_agent
extends: none
tools: []
---

{{ user_prompt }}
""")

        agent = parse_agent_file(agent_file)
        preparer = AgentPreparer()

        baseline = preparer.prepare(agent=agent, prompt="please use magicword here", context={})
        assert "triggered_skill" in [s.name for s in baseline.skills]

        prepared = preparer.prepare(
            agent=agent,
            prompt="please use magicword here",
            context={"suppressed_skills": ["triggered_skill"]},
        )
        assert "triggered_skill" not in [s.name for s in prepared.skills]


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
        """Test that _build_messages includes skills in context turn."""
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

        # New architecture: system message is stable string, skills in context turn
        assert len(messages) >= 4  # system, context user, context assistant, task
        assert messages[0]["role"] == "system"
        assert isinstance(messages[0]["content"], str)  # Stable system message

        # Context turn should exist (user message with skills)
        assert messages[1]["role"] == "user"
        assert "cache_control" in messages[1]

        # Context should contain skills in XML format
        context_content = messages[1]["content"]
        assert isinstance(context_content, list)
        context_text = context_content[0]["text"]
        assert "<context>" in context_text
        assert '<skill_content name="skill1">' in context_text
        assert '<skill_content name="skill2">' in context_text
        assert "# Skill 1" in context_text
        assert "# Skill 2" in context_text

        # Assistant acknowledgement
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Context loaded."

    def test_dynamically_loaded_skill_appears_in_context_turn(self):
        """Skills added to self.skills after a dynamic load_skill() call must
        show up in the cached context turn on subsequent _build_messages calls."""
        from tsugite.core.agent import TsugiteAgent

        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[],
            instructions="Test instructions",
            max_turns=3,
            attachments=[],
            skills=[Skill(name="auto-loaded", content="# Auto\nalways on")],
        )
        agent.memory.task = "Test task"

        agent.skills.append(Skill(name="dyn-loaded", content="# Dynamic\nloaded mid-run"))

        messages = agent._build_messages()
        context_text = messages[1]["content"][0]["text"]
        assert '<skill_content name="auto-loaded">' in context_text
        assert '<skill_content name="dyn-loaded">' in context_text
        assert "loaded mid-run" in context_text

    def test_expiring_skill_emits_skill_expiring_tag(self):
        """Skills listed in expiring_skills get a sibling XML block in context."""
        from tsugite.core.agent import TsugiteAgent

        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[],
            instructions="Test",
            max_turns=3,
            skills=[Skill(name="weather", content="# Weather\nbody")],
            expiring_skills={"weather": 1},
        )
        agent.memory.task = "hi"
        messages = agent._build_messages()
        context_text = messages[1]["content"][0]["text"]
        assert '<skill_content name="weather">' in context_text
        assert '<skill_expiring name="weather" turns_remaining="1">' in context_text
        assert "load_skill" in context_text
        assert "unload_skill" in context_text

    def test_non_expiring_skill_has_no_expiring_tag(self):
        from tsugite.core.agent import TsugiteAgent

        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[],
            instructions="Test",
            max_turns=3,
            skills=[Skill(name="weather", content="# Weather\nbody")],
            expiring_skills={},
        )
        agent.memory.task = "hi"
        messages = agent._build_messages()
        context_text = messages[1]["content"][0]["text"]
        assert "<skill_expiring" not in context_text

    def test_build_observation_does_not_embed_skill_content(self):
        """Dynamic skill content must not live in the observation replay:
        it flows via self.skills + context turn instead, so compaction and
        per-turn token cost stay clean."""
        from tsugite.core.agent import TsugiteAgent
        from tsugite.core.memory import StepResult

        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[],
            instructions="Test instructions",
            max_turns=3,
        )

        step = StepResult(
            step_number=1,
            thought="thinking",
            code="load_skill('foo')",
            output="ok",
            error=None,
            tools_called=["load_skill"],
            loaded_skills={"foo": "# Foo\nsecret skill body"},
            xml_observation="<observation>ok</observation>",
            content_blocks=[],
        )

        rendered = agent._build_observation(step)
        assert "secret skill body" not in rendered
        assert "<loaded_skill" not in rendered
        assert rendered == "<observation>ok</observation>"

    def test_skills_formatted_like_attachments(self):
        """Test that skills and attachments are in the same context turn."""
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

        # New architecture: both in context turn (user message)
        assert messages[1]["role"] == "user"
        context_content = messages[1]["content"]
        assert isinstance(context_content, list)
        context_text = context_content[0]["text"]

        # Both attachment and skill should be in context turn
        assert "<context>" in context_text
        assert '<attachment name="attach1">' in context_text
        assert "Attachment content" in context_text
        assert '<skill_content name="skill1">' in context_text
        assert "Skill content" in context_text

        # Context turn should have cache control
        assert "cache_control" in messages[1]
        assert messages[1]["cache_control"]["type"] == "ephemeral"


class TestSkillLoadErrorHandling:
    """Test skill load error handling and event emission."""

    def test_skill_not_found_emits_error_event(self, tmp_path, monkeypatch):
        """Test that SkillLoadFailedEvent is emitted when skill not found."""
        monkeypatch.chdir(tmp_path)

        agent_file = tmp_path / "agent.md"
        agent_file.write_text("""---
name: test_agent
extends: none
auto_load_skills:
  - nonexistent_skill
tools: []
---

# Test Agent
{{ user_prompt }}
""")

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
        agent_file.write_text("""---
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
""")

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

        valid_dir = skills_dir / "valid_skill"
        valid_dir.mkdir()
        (valid_dir / "SKILL.md").write_text("""---
name: valid_skill
description: Valid skill
---

# Valid Skill
This skill loads successfully.
""")

        agent_file = tmp_path / "agent.md"
        agent_file.write_text("""---
name: test_agent
extends: none
auto_load_skills:
  - valid_skill
  - invalid_skill
tools: []
---

# Test Agent
{{ user_prompt }}
""")

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
        agent_file.write_text("""---
name: test_agent
extends: none
auto_load_skills:
  - nonexistent_skill
tools: []
---

# Test Agent
This agent should still work despite skill load failure.
{{ user_prompt }}
""")

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
        agent_file.write_text("""---
name: test_agent
extends: none
auto_load_skills:
  - nonexistent_skill
tools: []
---

# Test Agent
{{ user_prompt }}
""")

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
