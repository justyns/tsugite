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
    """Ensure load_skill, unload_skill and list_available_skills tools are registered."""
    from tsugite.tools import tool
    from tsugite.tools.skills import list_available_skills, load_skill, unload_skill

    tool(load_skill)
    tool(unload_skill)
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
        assert f'<skill_resources dir="{skill_dir}">' in rendered
        assert "</skill_resources>" in rendered
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
        # Minimal skills stay uncluttered: no resource wrapper when there's nothing to bundle.
        assert "<skill_resources" not in rendered
        assert "</skill_resources>" not in rendered

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


class TestUnloadSkillTool:
    """unload_skill tool drops a skill from the manager + registers with executor."""

    def test_unload_removes_from_manager(self, tmp_path, monkeypatch):
        skill_dir = _make_skill_dir(tmp_path / "skills", "drop-me")
        monkeypatch.chdir(tmp_path)

        manager = SkillManager()
        manager._skill_registry = {"drop-me": _meta(skill_dir)}
        manager._registry_initialized = True
        manager._loaded_skills = {"drop-me": "content"}
        set_skill_manager(manager)

        result = call_tool("unload_skill", skill_name="drop-me")
        assert "unload" in result.lower() or "success" in result.lower()
        assert "drop-me" not in manager._loaded_skills

    def test_unload_not_currently_loaded(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        manager = SkillManager()
        manager._skill_registry = {}
        manager._registry_initialized = True
        set_skill_manager(manager)

        result = call_tool("unload_skill", skill_name="missing")
        assert "not currently loaded" in result.lower()

    def test_unload_registers_with_executor(self, tmp_path, monkeypatch):
        """Mid-turn unload_skill surfaces in ExecutionResult.unloaded_skills so the
        daemon can drop the skill from session-level sticky state."""
        skill_dir = _make_skill_dir(tmp_path / "skills", "drop-me")
        monkeypatch.chdir(tmp_path)

        class _FakeExecutor:
            def __init__(self):
                self.unloaded: list[str] = []

            def register_loaded_skill(self, name, content):
                pass

            def register_unloaded_skill(self, name):
                self.unloaded.append(name)

        executor = _FakeExecutor()
        manager = SkillManager()
        manager._skill_registry = {"drop-me": _meta(skill_dir)}
        manager._registry_initialized = True
        manager._loaded_skills = {"drop-me": "content"}
        manager.set_executor(executor)
        set_skill_manager(manager)

        call_tool("unload_skill", skill_name="drop-me")
        assert executor.unloaded == ["drop-me"]


class TestLoadSkillRenewal:
    """load_skill on an already-loaded skill registers with the executor so the
    daemon can reset the TTL counter."""

    def test_reload_registers_again(self, tmp_path, monkeypatch):
        skill_dir = _make_skill_dir(tmp_path / "skills", "already-here")
        monkeypatch.chdir(tmp_path)

        class _FakeExecutor:
            def __init__(self):
                self.loaded: list[str] = []

            def register_loaded_skill(self, name, content):
                self.loaded.append(name)

        executor = _FakeExecutor()
        manager = SkillManager()
        manager._skill_registry = {"already-here": _meta(skill_dir)}
        manager._registry_initialized = True
        manager._loaded_skills = {"already-here": "cached content"}
        manager.set_executor(executor)
        set_skill_manager(manager)

        result = call_tool("load_skill", skill_name="already-here")
        assert "renew" in result.lower() or "already" in result.lower()
        assert executor.loaded == ["already-here"]


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


class TestSkillFailureTracking:
    """SkillManager surfaces scan + load failures for user-facing surfaces."""

    def test_scan_issues_captured_during_init(self, tmp_path, monkeypatch):
        skills_root = tmp_path / ".tsugite" / "skills"
        skills_root.mkdir(parents=True)
        bad = skills_root / "bad-trig"
        bad.mkdir()
        (bad / "SKILL.md").write_text(
            "---\nname: bad-trig\ndescription: x\ntriggers:\n  - 403\n---\nBody.\n"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))

        manager = SkillManager()
        manager._ensure_registry_initialized()

        assert "bad-trig" in manager._skill_registry
        names_with_issues = [i.name for i in manager._scan_issues]
        assert "bad-trig" in names_with_issues

    def test_load_failure_recorded_on_render_error(self, tmp_path, monkeypatch):
        skill_dir = _make_skill_dir(
            tmp_path / "skills", "broken", body="Hello {{ undefined_var }}\n"
        )
        registry = {"broken": _meta(skill_dir)}
        monkeypatch.chdir(tmp_path)

        manager = SkillManager()
        manager._skill_registry = registry
        manager._registry_initialized = True
        set_skill_manager(manager)

        result = manager.load_skill("broken")
        assert "Failed" in result
        assert "broken" in manager._load_failures
        assert manager._load_failures["broken"]

    def test_load_failure_cleared_on_successful_reload(self, tmp_path, monkeypatch):
        skill_dir = _make_skill_dir(tmp_path / "skills", "fixme", body="Body.\n")
        registry = {"fixme": _meta(skill_dir)}
        monkeypatch.chdir(tmp_path)

        manager = SkillManager()
        manager._skill_registry = registry
        manager._registry_initialized = True
        manager._load_failures["fixme"] = "previous error"

        manager.load_skill("fixme")
        assert "fixme" not in manager._load_failures

    def test_skill_not_found_does_not_track_as_failure(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        manager = SkillManager()
        manager._skill_registry = {}
        manager._registry_initialized = True

        manager.load_skill("typo-skill")
        assert "typo-skill" not in manager._load_failures

    def test_get_failed_skills_list_merges_scan_and_load_failures(self, tmp_path, monkeypatch):
        skills_root = tmp_path / ".tsugite" / "skills"
        skills_root.mkdir(parents=True)
        # Scan issue: bad ttl
        bad = skills_root / "bad-ttl"
        bad.mkdir()
        (bad / "SKILL.md").write_text(
            '---\nname: bad-ttl\ndescription: x\nttl: "oops"\n---\nBody.\n'
        )
        # Load issue: undefined Jinja var
        broken = skills_root / "broken"
        broken.mkdir()
        (broken / "SKILL.md").write_text(
            "---\nname: broken\ndescription: x\n---\n{{ undefined_var }}\n"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))

        manager = SkillManager()
        manager._ensure_registry_initialized()
        set_skill_manager(manager)
        manager.load_skill("broken")

        failed = manager.get_failed_skills_list()
        sources_by_name = {item["name"]: item["source"] for item in failed}
        assert sources_by_name.get("bad-ttl") == "scan"
        assert sources_by_name.get("broken") == "load"
        for item in failed:
            assert set(item.keys()) >= {"name", "source", "path", "severity", "message"}

    def test_get_failed_skills_for_template_tool(self, tmp_path, monkeypatch):
        skills_root = tmp_path / ".tsugite" / "skills"
        skills_root.mkdir(parents=True)
        bad = skills_root / "bad-trig"
        bad.mkdir()
        (bad / "SKILL.md").write_text(
            "---\nname: bad-trig\ndescription: x\ntriggers:\n  - 403\n---\nBody.\n"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))

        manager = SkillManager()
        manager._ensure_registry_initialized()
        set_skill_manager(manager)

        from tsugite.tools.skills import get_failed_skills_for_template

        result = get_failed_skills_for_template()
        assert isinstance(result, list)
        names = [item["name"] for item in result]
        assert "bad-trig" in names
