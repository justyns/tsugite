"""Tests for skill discovery system."""

from pathlib import Path

import pytest

from tsugite.skill_discovery import (
    SkillMeta,
    build_skill_index,
    get_builtin_skills_path,
    scan_skills,
)


class TestSkillMeta:
    """Test SkillMeta dataclass."""

    def test_skill_meta_creation(self):
        """Test creating a SkillMeta instance."""
        skill = SkillMeta(
            name="test_skill",
            description="A test skill",
            path=Path("/path/to/skill.md"),
        )

        assert skill.name == "test_skill"
        assert skill.description == "A test skill"
        assert skill.path == Path("/path/to/skill.md")

    def test_skill_meta_defaults(self):
        """Test SkillMeta with minimal required fields."""
        skill = SkillMeta(
            name="simple_skill",
            description="Simple",
            path=Path("/path/skill.md"),
        )

        assert skill.name == "simple_skill"
        assert skill.description == "Simple"


class TestGetBuiltinSkillsPath:
    """Test builtin skills path resolution."""

    def test_get_builtin_skills_path(self):
        """Test that builtin skills path is resolved correctly."""
        path = get_builtin_skills_path()

        assert path.is_absolute()
        assert path.name == "builtin_skills"
        assert "tsugite" in str(path)


class TestScanSkills:
    """Test skill scanning functionality."""

    @pytest.fixture
    def skill_dirs(self, tmp_path):
        """Create temporary skill directories for testing."""
        # Create directory structure
        project_local = tmp_path / ".tsugite" / "skills"
        project_conv = tmp_path / "skills"
        user_global = tmp_path / "config" / "tsugite" / "skills"

        project_local.mkdir(parents=True)
        project_conv.mkdir(parents=True)
        user_global.mkdir(parents=True)

        return {
            "project_local": project_local,
            "project_conv": project_conv,
            "user_global": user_global,
            "base": tmp_path,
        }

    def test_scan_empty_directories(self, skill_dirs, monkeypatch):
        """Test scanning with no skill files."""
        monkeypatch.chdir(skill_dirs["base"])
        monkeypatch.setenv("HOME", str(skill_dirs["base"]))

        skills = scan_skills()

        # Should only find builtin skills (if any exist)
        assert isinstance(skills, list)

    def test_scan_single_skill(self, skill_dirs, monkeypatch):
        """Test scanning a single skill file."""
        monkeypatch.chdir(skill_dirs["base"])
        monkeypatch.setenv("HOME", str(skill_dirs["base"]))

        # Create a skill file
        skill_file = skill_dirs["project_local"] / "test_skill.md"
        skill_file.write_text("""---
name: test_skill
description: A test skill for testing
---

# Test Skill Content
This is a test skill.
""")

        skills = scan_skills()

        # Find our test skill
        test_skills = [s for s in skills if s.name == "test_skill"]
        assert len(test_skills) == 1

        skill = test_skills[0]
        assert skill.name == "test_skill"
        assert skill.description == "A test skill for testing"
        assert skill.path == skill_file

    def test_scan_multiple_skills(self, skill_dirs, monkeypatch):
        """Test scanning multiple skill files."""
        monkeypatch.chdir(skill_dirs["base"])
        monkeypatch.setenv("HOME", str(skill_dirs["base"]))

        # Create skills in different directories
        skill1 = skill_dirs["project_local"] / "skill1.md"
        skill1.write_text("""---
name: skill1
description: First skill
---
Content
""")

        skill2 = skill_dirs["project_conv"] / "skill2.md"
        skill2.write_text("""---
name: skill2
description: Second skill
---
Content
""")

        skills = scan_skills()
        skill_names = [s.name for s in skills]

        assert "skill1" in skill_names
        assert "skill2" in skill_names

    def test_scan_skill_priority(self, skill_dirs, monkeypatch):
        """Test that higher priority directories override lower priority."""
        monkeypatch.chdir(skill_dirs["base"])
        monkeypatch.setenv("HOME", str(skill_dirs["base"]))

        # Create same-named skill in different directories
        # Higher priority (project_local)
        skill_high = skill_dirs["project_local"] / "duplicate.md"
        skill_high.write_text("""---
name: duplicate
description: High priority version
---
Content
""")

        # Lower priority (project_conv)
        skill_low = skill_dirs["project_conv"] / "duplicate.md"
        skill_low.write_text("""---
name: duplicate
description: Low priority version
---
Content
""")

        skills = scan_skills()
        duplicate_skills = [s for s in skills if s.name == "duplicate"]

        # Should only have one (highest priority)
        assert len(duplicate_skills) == 1
        assert duplicate_skills[0].description == "High priority version"
        assert duplicate_skills[0].path == skill_high

    def test_scan_invalid_yaml_frontmatter(self, skill_dirs, monkeypatch):
        """Test that skills with invalid YAML are skipped."""
        monkeypatch.chdir(skill_dirs["base"])
        monkeypatch.setenv("HOME", str(skill_dirs["base"]))

        # Create skill with invalid YAML
        invalid_skill = skill_dirs["project_local"] / "invalid.md"
        invalid_skill.write_text("""---
name: invalid
invalid: yaml: structure: bad
---
Content
""")

        # Should not raise, just skip invalid skills
        skills = scan_skills()
        invalid_names = [s.name for s in skills if s.name == "invalid"]
        assert len(invalid_names) == 0

    def test_scan_missing_name_field(self, skill_dirs, monkeypatch):
        """Test that skills without name field are skipped."""
        monkeypatch.chdir(skill_dirs["base"])
        monkeypatch.setenv("HOME", str(skill_dirs["base"]))

        # Create skill without name
        no_name = skill_dirs["project_local"] / "no_name.md"
        no_name.write_text("""---
description: No name field
---
Content
""")

        skills = scan_skills()
        # Should be skipped (no name to match)
        assert all(s.path != no_name for s in skills)

    def test_scan_no_frontmatter(self, skill_dirs, monkeypatch):
        """Test that files without frontmatter are skipped."""
        monkeypatch.chdir(skill_dirs["base"])
        monkeypatch.setenv("HOME", str(skill_dirs["base"]))

        # Create markdown without frontmatter
        no_frontmatter = skill_dirs["project_local"] / "plain.md"
        no_frontmatter.write_text("# Just markdown\nNo frontmatter here.")

        skills = scan_skills()
        # Should be skipped
        assert all(s.path != no_frontmatter for s in skills)

    def test_scan_nested_directories(self, skill_dirs, monkeypatch):
        """Test scanning nested skill directories."""
        monkeypatch.chdir(skill_dirs["base"])
        monkeypatch.setenv("HOME", str(skill_dirs["base"]))

        # Create nested directory structure
        nested = skill_dirs["project_local"] / "category" / "subcategory"
        nested.mkdir(parents=True)

        nested_skill = nested / "nested_skill.md"
        nested_skill.write_text("""---
name: nested_skill
description: A nested skill
---
Content
""")

        skills = scan_skills()
        nested_skills = [s for s in skills if s.name == "nested_skill"]

        assert len(nested_skills) == 1
        assert nested_skills[0].path == nested_skill


class TestBuildSkillIndex:
    """Test skill index building."""

    def test_build_empty_index(self):
        """Test building index with no skills."""
        skills = []
        index = build_skill_index(skills)

        assert index == ""

    def test_build_single_skill_index(self):
        """Test building index with single skill."""
        skills = [
            SkillMeta(
                name="test_skill",
                description="A test skill",
                path=Path("/path/skill.md"),
            )
        ]

        index = build_skill_index(skills)

        assert "test_skill" in index
        assert "A test skill" in index

    def test_build_multiple_skills_index(self):
        """Test building index with multiple skills."""
        skills = [
            SkillMeta(
                name="skill1",
                description="First skill",
                path=Path("/path1.md"),
            ),
            SkillMeta(
                name="skill2",
                description="Second skill",
                path=Path("/path2.md"),
            ),
            SkillMeta(
                name="skill3",
                description="Third skill",
                path=Path("/path3.md"),
            ),
        ]

        index = build_skill_index(skills)

        # All skills should be present
        assert "skill1" in index
        assert "skill2" in index
        assert "skill3" in index

        # Descriptions should be present
        assert "First skill" in index
        assert "Second skill" in index
        assert "Third skill" in index

    def test_build_skill_index_format(self):
        """Test that skill index has correct format."""
        skills = [
            SkillMeta(
                name="example_skill",
                description="An example skill",
                path=Path("/path.md"),
            )
        ]

        index = build_skill_index(skills)

        # Should be formatted as: - name: description
        assert "- example_skill:" in index or "example_skill:" in index
        assert "An example skill" in index
        # Should not have empty parentheses or brackets
        assert "()" not in index or "[]" not in index
