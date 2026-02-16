"""Tests for skill loading tools."""

import pytest

from tsugite.tools import call_tool
from tsugite.tools.skills import SkillManager, set_skill_manager


class TestLoadSkillTool:
    """Test load_skill tool functionality."""

    @pytest.fixture
    def skill_file(self, tmp_path):
        """Create a test skill file."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        skill = skills_dir / "test_skill.md"
        skill.write_text("""---
name: test_skill
description: A test skill
---

# Test Skill

This skill provides test capabilities.

## Usage
Use this skill for testing.
""")
        return skill

    @pytest.fixture
    def skill_registry(self, skill_file):
        """Create a skill registry with test skill."""
        from tsugite.skill_discovery import SkillMeta

        return {
            "test_skill": SkillMeta(
                name="test_skill",
                description="A test skill",
                path=skill_file,
            )
        }

    def test_load_skill_success(self, skill_file, skill_registry, monkeypatch):
        """Test successfully loading a skill."""
        monkeypatch.chdir(skill_file.parent.parent)

        # Create a skill manager with the test registry
        manager = SkillManager()
        manager._skill_registry = skill_registry
        manager._registry_initialized = True
        set_skill_manager(manager)

        # Register the load_skill tool
        from tsugite.tools import tool
        from tsugite.tools.skills import load_skill

        tool(load_skill)

        result = call_tool("load_skill", skill_name="test_skill")

        assert "test_skill" in result
        assert "success" in result.lower() or "loaded" in result.lower()

    def test_load_skill_not_found(self, skill_registry):
        """Test loading a skill that doesn't exist."""
        # Create a skill manager with the test registry
        manager = SkillManager()
        manager._skill_registry = skill_registry
        manager._registry_initialized = True
        set_skill_manager(manager)

        from tsugite.tools import tool
        from tsugite.tools.skills import load_skill

        tool(load_skill)

        result = call_tool("load_skill", skill_name="nonexistent_skill")

        assert "not found" in result.lower() or "error" in result.lower()
        assert "nonexistent_skill" in result

    def test_load_skill_already_loaded(self, skill_file, skill_registry, monkeypatch):
        """Test loading a skill that's already loaded."""
        monkeypatch.chdir(skill_file.parent.parent)

        # Create a skill manager with the test registry
        manager = SkillManager()
        manager._skill_registry = skill_registry
        manager._registry_initialized = True
        manager._loaded_skills = {"test_skill": "already loaded content"}
        set_skill_manager(manager)

        from tsugite.tools import tool
        from tsugite.tools.skills import load_skill

        tool(load_skill)

        result = call_tool("load_skill", skill_name="test_skill")

        assert "already" in result.lower()
        assert "test_skill" in result

    def test_load_skill_renders_template(self, skill_file, skill_registry, monkeypatch):
        """Test that skill content is rendered with Jinja2."""
        # Create skill with template variables
        skill_with_template = skill_file.parent / "template_skill.md"
        skill_with_template.write_text("""---
name: template_skill
description: Skill with template
---

# Template Skill

Today's date: {{ today() }}
User prompt: {{ user_prompt }}
""")

        from tsugite.skill_discovery import SkillMeta

        skill_registry["template_skill"] = SkillMeta(
            name="template_skill",
            description="Skill with template",
            path=skill_with_template,
        )

        monkeypatch.chdir(skill_file.parent.parent)

        # Create a skill manager with the test registry
        manager = SkillManager()
        manager._skill_registry = skill_registry
        manager._registry_initialized = True
        set_skill_manager(manager)

        from tsugite.tools import tool
        from tsugite.tools.skills import load_skill

        tool(load_skill)

        result = call_tool("load_skill", skill_name="template_skill")

        # Should succeed
        assert "success" in result.lower() or "loaded" in result.lower()

        # Content should be rendered (check in loaded_skills)
        loaded_content = manager._loaded_skills.get("template_skill", "")
        assert "Today's date:" in loaded_content
        # Template variables should be replaced (not {{ ... }})
        assert "{{" not in loaded_content or "}}" not in loaded_content


class TestListAvailableSkillsTool:
    """Test list_available_skills tool."""

    @pytest.fixture
    def skill_registry(self, tmp_path):
        """Create a skill registry with multiple skills."""
        from tsugite.skill_discovery import SkillMeta

        return {
            "skill1": SkillMeta(
                name="skill1",
                description="First skill",
                path=tmp_path / "skill1.md",
            ),
            "skill2": SkillMeta(
                name="skill2",
                description="Second skill",
                path=tmp_path / "skill2.md",
            ),
            "skill3": SkillMeta(
                name="skill3",
                description="Third skill",
                path=tmp_path / "skill3.md",
            ),
        }

    def test_list_available_skills(self, skill_registry):
        """Test listing all available skills."""
        # Create a skill manager with the test registry
        manager = SkillManager()
        manager._skill_registry = skill_registry
        manager._registry_initialized = True
        set_skill_manager(manager)

        from tsugite.tools import tool
        from tsugite.tools.skills import list_available_skills

        tool(list_available_skills)

        result = call_tool("list_available_skills")

        # All skills should be listed
        assert "skill1" in result
        assert "skill2" in result
        assert "skill3" in result

        # Descriptions should be present
        assert "First skill" in result
        assert "Second skill" in result
        assert "Third skill" in result

    def test_list_available_skills_empty(self):
        """Test listing when no skills are available."""
        # Create a skill manager with empty registry
        manager = SkillManager()
        manager._skill_registry = {}
        manager._registry_initialized = True
        set_skill_manager(manager)

        from tsugite.tools import tool
        from tsugite.tools.skills import list_available_skills

        tool(list_available_skills)

        result = call_tool("list_available_skills")

        assert "no skill" in result.lower() or "empty" in result.lower() or result == ""


class TestSkillManagerEvents:
    """Test skill manager event emission."""

    def test_load_skill_emits_event(self, tmp_path):
        """Test that loading a skill emits a SkillLoadedEvent."""
        from tsugite import ui_context
        from tsugite.events import EventBus, SkillLoadedEvent
        from tsugite.skill_discovery import SkillMeta

        # Create a test skill file
        skill_file = tmp_path / "test_skill.md"
        skill_file.write_text("""---
name: test_skill
description: A test skill
---
Content
""")

        # Create event bus and track emitted events
        event_bus = EventBus()
        emitted_events = []

        def capture_event(event):
            emitted_events.append(event)

        event_bus.subscribe(capture_event)

        # Set event bus in ui_context so helpers can find it
        ui_context._event_bus_var.set(event_bus)

        try:
            # Create skill manager (no event_bus parameter needed)
            manager = SkillManager()
            manager._skill_registry = {
                "test_skill": SkillMeta(
                    name="test_skill",
                    description="A test skill",
                    path=skill_file,
                )
            }
            manager._registry_initialized = True

            # Load skill
            manager.load_skill("test_skill")

            # Should emit SkillLoadedEvent
            assert len(emitted_events) == 1
            assert isinstance(emitted_events[0], SkillLoadedEvent)
            assert emitted_events[0].skill_name == "test_skill"
            assert emitted_events[0].description == "A test skill"
        finally:
            # Clean up
            ui_context._event_bus_var.set(None)

    def test_unload_skill_emits_event(self):
        """Test that unloading a skill emits a SkillUnloadedEvent."""
        from tsugite import ui_context
        from tsugite.events import EventBus, SkillUnloadedEvent

        # Create event bus and track emitted events
        event_bus = EventBus()
        emitted_events = []

        def capture_event(event):
            emitted_events.append(event)

        event_bus.subscribe(capture_event)

        # Set event bus in ui_context so helpers can find it
        ui_context._event_bus_var.set(event_bus)

        try:
            # Create skill manager (no event_bus parameter needed)
            manager = SkillManager()
            manager._loaded_skills = {"test_skill": "content"}

            # Unload skill
            manager.unload_skill("test_skill")

            # Should emit SkillUnloadedEvent
            assert len(emitted_events) == 1
            assert isinstance(emitted_events[0], SkillUnloadedEvent)
            assert emitted_events[0].skill_name == "test_skill"
        finally:
            # Clean up
            ui_context._event_bus_var.set(None)

    def test_load_skill_no_event_without_bus(self, tmp_path):
        """Test that loading a skill without event bus doesn't crash."""
        from tsugite import ui_context
        from tsugite.skill_discovery import SkillMeta

        # Create a test skill file
        skill_file = tmp_path / "test_skill.md"
        skill_file.write_text("""---
name: test_skill
description: A test skill
---
Content
""")

        # Ensure no event bus is set in ui_context
        ui_context._event_bus_var.set(None)

        try:
            # Create skill manager (helper will handle missing event bus gracefully)
            manager = SkillManager()
            manager._skill_registry = {
                "test_skill": SkillMeta(
                    name="test_skill",
                    description="A test skill",
                    path=skill_file,
                )
            }
            manager._registry_initialized = True

            # Load skill - should work without crashing
            result = manager.load_skill("test_skill")
            assert "success" in result.lower() or "loaded" in result.lower()
        finally:
            # Clean up (though already None)
            ui_context._event_bus_var.set(None)
