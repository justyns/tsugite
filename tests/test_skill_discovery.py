"""Tests for skill discovery system."""

import logging
from pathlib import Path

import pytest

from tsugite.skill_discovery import (
    SkillMeta,
    _validate_skill_name,
    build_skill_index,
    get_builtin_skills_path,
    match_triggered_skills,
    scan_skills,
)


def _write_skill(
    root: Path,
    name: str,
    description: str = "A skill",
    dir_name: str | None = None,
    triggers: list[str] | None = None,
    extra_frontmatter: str = "",
) -> Path:
    """Create a directory-based skill at <root>/<dir_name>/SKILL.md."""
    skill_dir = root / (dir_name or name)
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    triggers_block = ""
    if triggers is not None:
        trigger_lines = "\n".join(f"  - {t}" for t in triggers)
        triggers_block = f"triggers:\n{trigger_lines}\n"
    frontmatter = (
        f"---\nname: {name}\ndescription: {description}\n{triggers_block}{extra_frontmatter}---\nBody for {name}.\n"
    )
    skill_md.write_text(frontmatter)
    return skill_dir


class TestSkillMeta:
    def test_skill_meta_creation(self):
        skill = SkillMeta(
            name="test-skill",
            description="A test skill",
            directory=Path("/path/to/test-skill"),
            skill_md_path=Path("/path/to/test-skill/SKILL.md"),
        )
        assert skill.name == "test-skill"
        assert skill.description == "A test skill"
        assert skill.directory == Path("/path/to/test-skill")
        assert skill.skill_md_path == Path("/path/to/test-skill/SKILL.md")
        assert skill.triggers == []


class TestGetBuiltinSkillsPath:
    def test_get_builtin_skills_path(self):
        path = get_builtin_skills_path()
        assert path.is_absolute()
        assert path.name == "builtin_skills"
        assert "tsugite" in str(path)


class TestValidateSkillName:
    def test_valid_name(self):
        assert _validate_skill_name("python-math", "python-math") == []

    def test_mismatched_directory(self):
        warnings = _validate_skill_name("python-math", "python_math")
        assert any("does not match directory" in w for w in warnings)

    def test_uppercase_is_invalid(self):
        warnings = _validate_skill_name("PythonMath", "PythonMath")
        assert any("not spec-compliant" in w for w in warnings)

    def test_underscore_is_invalid(self):
        warnings = _validate_skill_name("python_math", "python_math")
        assert any("not spec-compliant" in w for w in warnings)

    def test_leading_hyphen_is_invalid(self):
        warnings = _validate_skill_name("-foo", "-foo")
        assert any("not spec-compliant" in w for w in warnings)

    def test_consecutive_hyphens_invalid(self):
        warnings = _validate_skill_name("foo--bar", "foo--bar")
        assert any("not spec-compliant" in w for w in warnings)

    def test_too_long(self):
        long_name = "a" + "-a" * 40
        warnings = _validate_skill_name(long_name, long_name)
        assert any("exceeds" in w for w in warnings)


class TestScanSkills:
    @pytest.fixture
    def skill_dirs(self, tmp_path):
        project_local = tmp_path / ".tsugite" / "skills"
        project_conv = tmp_path / "skills"
        project_agents = tmp_path / ".agents" / "skills"
        user_agents = tmp_path / ".agents" / "skills"
        user_global = tmp_path / ".config" / "tsugite" / "skills"
        for p in (project_local, project_conv, project_agents, user_global):
            p.mkdir(parents=True, exist_ok=True)
        return {
            "project_local": project_local,
            "project_conv": project_conv,
            "project_agents": project_agents,
            "user_agents": user_agents,
            "user_global": user_global,
            "base": tmp_path,
        }

    def test_scan_empty_directories(self, skill_dirs, monkeypatch):
        monkeypatch.chdir(skill_dirs["base"])
        monkeypatch.setenv("HOME", str(skill_dirs["base"]))
        skills = scan_skills()
        names = {s.name for s in skills}
        for d in skill_dirs.values():
            assert isinstance(d, Path) or d is skill_dirs["base"]
        # No user-defined skills; may still contain built-ins shipped with the package
        assert "fake-skill" not in names

    def test_scan_single_skill(self, skill_dirs, monkeypatch):
        monkeypatch.chdir(skill_dirs["base"])
        monkeypatch.setenv("HOME", str(skill_dirs["base"]))
        skill_dir = _write_skill(skill_dirs["project_local"], "my-skill", "Test skill")

        skills = scan_skills()
        mine = [s for s in skills if s.name == "my-skill"]
        assert len(mine) == 1
        assert mine[0].directory == skill_dir
        assert mine[0].skill_md_path == skill_dir / "SKILL.md"
        assert mine[0].description == "Test skill"

    def test_scan_multiple_skills(self, skill_dirs, monkeypatch):
        monkeypatch.chdir(skill_dirs["base"])
        monkeypatch.setenv("HOME", str(skill_dirs["base"]))
        _write_skill(skill_dirs["project_local"], "skill-one", "First skill")
        _write_skill(skill_dirs["project_conv"], "skill-two", "Second skill")
        skills = {s.name for s in scan_skills()}
        assert "skill-one" in skills
        assert "skill-two" in skills

    def test_scan_skill_priority(self, skill_dirs, monkeypatch):
        """project_local beats project_conv for same-named skills."""
        monkeypatch.chdir(skill_dirs["base"])
        monkeypatch.setenv("HOME", str(skill_dirs["base"]))
        high = _write_skill(skill_dirs["project_local"], "duplicate", "High priority")
        _write_skill(skill_dirs["project_conv"], "duplicate", "Low priority")

        hits = [s for s in scan_skills() if s.name == "duplicate"]
        assert len(hits) == 1
        assert hits[0].description == "High priority"
        assert hits[0].directory == high

    def test_directory_without_skill_md_is_ignored(self, skill_dirs, monkeypatch):
        monkeypatch.chdir(skill_dirs["base"])
        monkeypatch.setenv("HOME", str(skill_dirs["base"]))
        (skill_dirs["project_local"] / "empty").mkdir()
        (skill_dirs["project_local"] / "empty" / "README.md").write_text("not a skill")
        _write_skill(skill_dirs["project_local"], "real-skill", "Real")
        names = {s.name for s in scan_skills()}
        assert "real-skill" in names
        assert "empty" not in names

    def test_nested_subdirs_not_discovered(self, skill_dirs, monkeypatch):
        monkeypatch.chdir(skill_dirs["base"])
        monkeypatch.setenv("HOME", str(skill_dirs["base"]))
        outer = _write_skill(skill_dirs["project_local"], "outer", "Outer")
        nested = outer / "inner"
        nested.mkdir()
        (nested / "SKILL.md").write_text("---\nname: inner\ndescription: Inner\n---\nBody\n")
        names = {s.name for s in scan_skills()}
        assert "outer" in names
        assert "inner" not in names

    def test_agents_skills_discovered(self, skill_dirs, monkeypatch):
        monkeypatch.chdir(skill_dirs["base"])
        monkeypatch.setenv("HOME", str(skill_dirs["base"]))
        _write_skill(skill_dirs["project_agents"], "agents-skill", "From .agents/skills")
        names = {s.name for s in scan_skills()}
        assert "agents-skill" in names

    def test_user_level_agents_skills_discovered(self, skill_dirs, monkeypatch):
        monkeypatch.chdir(skill_dirs["base"])
        monkeypatch.setenv("HOME", str(skill_dirs["base"]))
        user_agents = skill_dirs["base"] / ".agents" / "skills"
        user_agents.mkdir(parents=True, exist_ok=True)
        _write_skill(user_agents, "user-skill", "User-level")
        names = {s.name for s in scan_skills()}
        assert "user-skill" in names

    def test_invalid_yaml_frontmatter(self, skill_dirs, monkeypatch):
        monkeypatch.chdir(skill_dirs["base"])
        monkeypatch.setenv("HOME", str(skill_dirs["base"]))
        bad = skill_dirs["project_local"] / "bad"
        bad.mkdir()
        (bad / "SKILL.md").write_text("---\nname: bad\ninvalid: yaml: structure: bad\n---\nBody\n")
        names = {s.name for s in scan_skills()}
        assert "bad" not in names

    def test_missing_name_is_skipped(self, skill_dirs, monkeypatch):
        monkeypatch.chdir(skill_dirs["base"])
        monkeypatch.setenv("HOME", str(skill_dirs["base"]))
        target = skill_dirs["project_local"] / "noname"
        target.mkdir()
        (target / "SKILL.md").write_text("---\ndescription: no name\n---\nBody\n")
        for s in scan_skills():
            assert s.directory != target

    def test_name_mismatch_logs_warning_but_loads(self, skill_dirs, monkeypatch, caplog):
        monkeypatch.chdir(skill_dirs["base"])
        monkeypatch.setenv("HOME", str(skill_dirs["base"]))
        _write_skill(
            skill_dirs["project_local"],
            name="declared-name",
            dir_name="other-dir",
            description="Mismatch",
        )
        with caplog.at_level(logging.WARNING):
            names = {s.name for s in scan_skills()}
        assert "declared-name" in names
        assert any("does not match directory" in rec.getMessage() for rec in caplog.records)

    def test_invalid_name_logs_warning_but_loads(self, skill_dirs, monkeypatch, caplog):
        monkeypatch.chdir(skill_dirs["base"])
        monkeypatch.setenv("HOME", str(skill_dirs["base"]))
        _write_skill(
            skill_dirs["project_local"],
            name="Invalid_Name",
            dir_name="Invalid_Name",
            description="Bad name",
        )
        with caplog.at_level(logging.WARNING):
            names = {s.name for s in scan_skills()}
        assert "Invalid_Name" in names
        assert any("not spec-compliant" in rec.getMessage() for rec in caplog.records)


class TestBuildSkillIndex:
    def _meta(self, name, desc):
        return SkillMeta(
            name=name,
            description=desc,
            directory=Path(f"/skills/{name}"),
            skill_md_path=Path(f"/skills/{name}/SKILL.md"),
        )

    def test_build_empty_index(self):
        assert build_skill_index([]) == ""

    def test_build_single_skill_index(self):
        index = build_skill_index([self._meta("one", "First")])
        assert "- one: First" in index

    def test_build_multiple_skills_index(self):
        skills = [self._meta("a", "A desc"), self._meta("b", "B desc")]
        index = build_skill_index(skills)
        assert "- a: A desc" in index
        assert "- b: B desc" in index


class TestScanSkillsExtraPaths:
    @pytest.fixture
    def dirs(self, tmp_path):
        project_local = tmp_path / ".tsugite" / "skills"
        extra = tmp_path / "extra_skills"
        project_local.mkdir(parents=True)
        extra.mkdir(parents=True)
        return {"project_local": project_local, "extra": extra, "base": tmp_path}

    def test_extra_paths_scanned(self, dirs, monkeypatch):
        monkeypatch.chdir(dirs["base"])
        monkeypatch.setenv("HOME", str(dirs["base"]))
        _write_skill(dirs["extra"], "custom-skill", "Custom")
        skills = scan_skills(extra_paths=[str(dirs["extra"])])
        assert "custom-skill" in {s.name for s in skills}

    def test_extra_paths_priority_over_project(self, dirs, monkeypatch):
        monkeypatch.chdir(dirs["base"])
        monkeypatch.setenv("HOME", str(dirs["base"]))
        _write_skill(dirs["extra"], "dup", "Extra version")
        _write_skill(dirs["project_local"], "dup", "Project version")
        skills = scan_skills(extra_paths=[str(dirs["extra"])])
        dup = [s for s in skills if s.name == "dup"]
        assert len(dup) == 1
        assert dup[0].description == "Extra version"

    def test_extra_paths_tilde_expansion(self, dirs, monkeypatch):
        monkeypatch.chdir(dirs["base"])
        monkeypatch.setenv("HOME", str(dirs["base"]))
        custom = dirs["base"] / "custom_skills"
        custom.mkdir()
        _write_skill(custom, "tilde-skill", "Tilde test")
        skills = scan_skills(extra_paths=["~/custom_skills"])
        assert "tilde-skill" in {s.name for s in skills}

    def test_extra_paths_nonexistent_ignored(self, dirs, monkeypatch):
        monkeypatch.chdir(dirs["base"])
        monkeypatch.setenv("HOME", str(dirs["base"]))
        _write_skill(dirs["project_local"], "real-skill", "Real")
        skills = scan_skills(extra_paths=["/nonexistent/path"])
        assert "real-skill" in {s.name for s in skills}

    def test_extra_paths_empty_list(self, dirs, monkeypatch):
        monkeypatch.chdir(dirs["base"])
        monkeypatch.setenv("HOME", str(dirs["base"]))
        _write_skill(dirs["project_local"], "t", "t")
        names_none = {s.name for s in scan_skills(extra_paths=None)}
        names_empty = {s.name for s in scan_skills(extra_paths=[])}
        assert names_none == names_empty


class TestMatchTriggeredSkills:
    def _make_skill(self, name, triggers=None):
        return SkillMeta(
            name=name,
            description=f"{name} skill",
            directory=Path(f"/skills/{name}"),
            skill_md_path=Path(f"/skills/{name}/SKILL.md"),
            triggers=triggers or [],
        )

    def test_basic_trigger_match(self):
        skills = [self._make_skill("weather", triggers=["weather", "forecast"])]
        result = match_triggered_skills("What's the weather today?", skills)
        assert len(result) == 1
        assert result[0].name == "weather"

    def test_case_insensitive(self):
        skills = [self._make_skill("weather", triggers=["weather"])]
        assert len(match_triggered_skills("WEATHER report please", skills)) == 1

    def test_word_boundary_no_substring(self):
        skills = [self._make_skill("weather", triggers=["weather"])]
        assert len(match_triggered_skills("The weathering of rocks", skills)) == 0

    def test_word_boundary_matches_punctuation(self):
        skills = [self._make_skill("weather", triggers=["weather"])]
        assert len(match_triggered_skills("How's the weather?", skills)) == 1

    def test_no_triggers_skipped(self):
        skills = [self._make_skill("basic", triggers=[])]
        assert len(match_triggered_skills("anything at all", skills)) == 0

    def test_no_match(self):
        skills = [self._make_skill("weather", triggers=["weather", "forecast"])]
        assert len(match_triggered_skills("Tell me about Python", skills)) == 0

    def test_multiple_skills_matched(self):
        skills = [
            self._make_skill("weather", triggers=["weather"]),
            self._make_skill("travel", triggers=["trip", "travel"]),
        ]
        assert len(match_triggered_skills("What's the weather for my trip?", skills)) == 2

    def test_max_skills_cap(self):
        skills = [self._make_skill(f"skill{i}", triggers=[f"word{i}"]) for i in range(5)]
        assert len(match_triggered_skills("word0 word1 word2 word3 word4", skills, max_skills=3)) == 3

    def test_already_loaded_skipped(self):
        skills = [
            self._make_skill("weather", triggers=["weather"]),
            self._make_skill("news", triggers=["news"]),
        ]
        result = match_triggered_skills("weather and news", skills, already_loaded={"weather"})
        assert len(result) == 1
        assert result[0].name == "news"

    def test_ranked_by_match_count(self):
        skills = [
            self._make_skill("general", triggers=["info"]),
            self._make_skill("weather", triggers=["weather", "forecast", "rain"]),
        ]
        result = match_triggered_skills("What's the weather forecast for rain today? I need info.", skills)
        assert len(result) == 2
        assert result[0].name == "weather"

    def test_empty_message(self):
        skills = [self._make_skill("weather", triggers=["weather"])]
        assert len(match_triggered_skills("", skills)) == 0

    def test_empty_skills_list(self):
        assert len(match_triggered_skills("weather forecast", [])) == 0

    def test_scan_skills_parses_triggers(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        _write_skill(
            tmp_path / ".tsugite" / "skills",
            name="weather",
            description="Weather skill",
            triggers=["weather", "forecast", "temperature"],
        )
        skills = scan_skills()
        weather = [s for s in skills if s.name == "weather"]
        assert len(weather) == 1
        assert weather[0].triggers == ["weather", "forecast", "temperature"]

    def test_scan_skills_no_triggers_defaults_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        _write_skill(tmp_path / ".tsugite" / "skills", "basic", "Basic skill")
        skills = scan_skills()
        basic = [s for s in skills if s.name == "basic"]
        assert len(basic) == 1
        assert basic[0].triggers == []
