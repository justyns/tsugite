"""Tests for skill loading tools."""

from pathlib import Path

import pytest

from tsugite.skill_discovery import SkillMeta
from tsugite.tools import call_tool
from tsugite.tools.skills import SkillManager, set_skill_manager


def _make_skill_dir(
    root: Path,
    name: str,
    description: str = "A test skill",
    body: str = "# Test Skill\n\nBody.\n",
    frontmatter_extra: str = "",
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(f"---\nname: {name}\ndescription: {description}\n{frontmatter_extra}---\n\n{body}")
    return skill_dir


def _meta(skill_dir: Path) -> SkillMeta:
    """Build a SkillMeta for the given directory by reading its frontmatter."""
    from tsugite.utils import parse_yaml_frontmatter

    skill_md = skill_dir / "SKILL.md"
    fm, _ = parse_yaml_frontmatter(skill_md.read_text())
    return SkillMeta(
        name=fm["name"],
        description=fm.get("description", ""),
        directory=skill_dir,
        skill_md_path=skill_md,
        triggers=fm.get("triggers", []) or [],
    )


@pytest.fixture(autouse=True)
def _register_skill_tools():
    """Ensure load_skill and list_available_skills tools are registered."""
    from tsugite.tools import tool
    from tsugite.tools.skills import list_available_skills, load_skill

    tool(load_skill)
    tool(list_available_skills)


class TestLoadSkillTool:
    @pytest.fixture
    def skill_dir(self, tmp_path):
        return _make_skill_dir(tmp_path / "skills", "test-skill", body="# Test\nUse this.\n")

    @pytest.fixture
    def registry(self, skill_dir):
        return {"test-skill": _meta(skill_dir)}

    def test_load_skill_success(self, skill_dir, registry, monkeypatch):
        monkeypatch.chdir(skill_dir.parent.parent)
        manager = SkillManager()
        manager._skill_registry = registry
        manager._registry_initialized = True
        set_skill_manager(manager)

        result = call_tool("load_skill", skill_name="test-skill")

        assert "test-skill" in result
        assert "success" in result.lower() or "loaded" in result.lower()

    def test_load_skill_not_found(self, registry):
        manager = SkillManager()
        manager._skill_registry = registry
        manager._registry_initialized = True
        set_skill_manager(manager)

        result = call_tool("load_skill", skill_name="nonexistent-skill")
        assert "not found" in result.lower()
        assert "nonexistent-skill" in result

    def test_load_skill_already_loaded(self, skill_dir, registry, monkeypatch):
        monkeypatch.chdir(skill_dir.parent.parent)
        manager = SkillManager()
        manager._skill_registry = registry
        manager._registry_initialized = True
        manager._loaded_skills = {"test-skill": "already loaded content"}
        set_skill_manager(manager)

        result = call_tool("load_skill", skill_name="test-skill")
        assert "already" in result.lower()

    def test_load_skill_renders_template(self, tmp_path, monkeypatch):
        skill_dir = _make_skill_dir(
            tmp_path / "skills",
            "template-skill",
            body="Today's date: {{ today() }}\nPrompt: {{ user_prompt }}\n",
        )
        registry = {"template-skill": _meta(skill_dir)}
        monkeypatch.chdir(tmp_path)

        manager = SkillManager()
        manager._skill_registry = registry
        manager._registry_initialized = True
        set_skill_manager(manager)

        call_tool("load_skill", skill_name="template-skill")
        rendered = manager._loaded_skills["template-skill"]
        assert "Today's date:" in rendered
        assert "{{" not in rendered

    def test_load_skill_appends_bundled_resources(self, tmp_path, monkeypatch):
        skill_dir = _make_skill_dir(tmp_path / "skills", "resourceful", body="Main body.\n")
        (skill_dir / "scripts").mkdir()
        (skill_dir / "scripts" / "build.sh").write_text("#!/bin/bash\necho hi\n")
        (skill_dir / "references").mkdir()
        (skill_dir / "references" / "api.md").write_text("# API\n")
        (skill_dir / "assets").mkdir()
        (skill_dir / "assets" / "template.txt").write_text("template")

        registry = {"resourceful": _meta(skill_dir)}
        monkeypatch.chdir(tmp_path)
        manager = SkillManager()
        manager._skill_registry = registry
        manager._registry_initialized = True
        set_skill_manager(manager)

        call_tool("load_skill", skill_name="resourceful")
        rendered = manager._loaded_skills["resourceful"]
        assert "**Skill directory:**" in rendered
        assert str(skill_dir) in rendered
        assert "**Bundled resources:**" in rendered
        assert "scripts/build.sh" in rendered
        assert "references/api.md" in rendered
        assert "assets/template.txt" in rendered

    def test_load_skill_no_bundled_subdirs_omits_resource_block(self, tmp_path, monkeypatch):
        skill_dir = _make_skill_dir(tmp_path / "skills", "plain", body="Body.\n")
        registry = {"plain": _meta(skill_dir)}
        monkeypatch.chdir(tmp_path)

        manager = SkillManager()
        manager._skill_registry = registry
        manager._registry_initialized = True
        set_skill_manager(manager)

        call_tool("load_skill", skill_name="plain")
        rendered = manager._loaded_skills["plain"]
        # Minimal skills stay uncluttered: no divider, no skill-directory line,
        # no bundled-resources block when there's nothing to bundle.
        assert "**Skill directory:**" not in rendered
        assert "**Bundled resources:**" not in rendered

    def test_load_skill_caps_large_resource_listings(self, tmp_path, monkeypatch):
        from tsugite.tools.skills import _MAX_RESOURCES_LISTED

        skill_dir = _make_skill_dir(tmp_path / "skills", "big", body="Body.\n")
        assets = skill_dir / "assets"
        assets.mkdir()
        extra = 5
        total = _MAX_RESOURCES_LISTED + extra
        for i in range(total):
            (assets / f"file-{i:02d}.txt").write_text("x")

        registry = {"big": _meta(skill_dir)}
        monkeypatch.chdir(tmp_path)
        manager = SkillManager()
        manager._skill_registry = registry
        manager._registry_initialized = True
        set_skill_manager(manager)

        call_tool("load_skill", skill_name="big")
        rendered = manager._loaded_skills["big"]
        assert f"+{extra} more" in rendered


class TestListAvailableSkillsTool:
    @pytest.fixture
    def registry(self, tmp_path):
        skills_root = tmp_path / "skills"
        return {
            "skill-one": _meta(_make_skill_dir(skills_root, "skill-one", description="First skill")),
            "skill-two": _meta(_make_skill_dir(skills_root, "skill-two", description="Second skill")),
        }

    def test_list_available_skills(self, registry):
        manager = SkillManager()
        manager._skill_registry = registry
        manager._registry_initialized = True
        set_skill_manager(manager)

        result = call_tool("list_available_skills")
        assert "skill-one" in result
        assert "skill-two" in result
        assert "First skill" in result
        assert "Second skill" in result

    def test_list_available_skills_empty(self):
        manager = SkillManager()
        manager._skill_registry = {}
        manager._registry_initialized = True
        set_skill_manager(manager)

        result = call_tool("list_available_skills")
        assert "no skill" in result.lower() or result == ""


class TestSkillManagerEvents:
    def test_load_skill_emits_event(self, tmp_path):
        from tsugite import ui_context
        from tsugite.events import EventBus, SkillLoadedEvent

        skill_dir = _make_skill_dir(tmp_path / "skills", "evt-skill", body="Content\n")
        event_bus = EventBus()
        events = []
        event_bus.subscribe(events.append)
        ui_context._event_bus_var.set(event_bus)
        try:
            manager = SkillManager()
            manager._skill_registry = {"evt-skill": _meta(skill_dir)}
            manager._registry_initialized = True
            manager.load_skill("evt-skill")

            loaded_events = [e for e in events if isinstance(e, SkillLoadedEvent)]
            assert len(loaded_events) == 1
            assert loaded_events[0].skill_name == "evt-skill"
            assert loaded_events[0].description == "A test skill"
        finally:
            ui_context._event_bus_var.set(None)

    def test_unload_skill_emits_event(self):
        from tsugite import ui_context
        from tsugite.events import EventBus, SkillUnloadedEvent

        event_bus = EventBus()
        events = []
        event_bus.subscribe(events.append)
        ui_context._event_bus_var.set(event_bus)
        try:
            manager = SkillManager()
            manager._loaded_skills = {"test-skill": "content"}
            manager.unload_skill("test-skill")
            unloaded = [e for e in events if isinstance(e, SkillUnloadedEvent)]
            assert len(unloaded) == 1
            assert unloaded[0].skill_name == "test-skill"
        finally:
            ui_context._event_bus_var.set(None)

    def test_load_skill_no_event_without_bus(self, tmp_path):
        from tsugite import ui_context

        ui_context._event_bus_var.set(None)
        skill_dir = _make_skill_dir(tmp_path / "skills", "test-skill")
        manager = SkillManager()
        manager._skill_registry = {"test-skill": _meta(skill_dir)}
        manager._registry_initialized = True

        result = manager.load_skill("test-skill")
        assert "loaded" in result.lower() or "success" in result.lower()
