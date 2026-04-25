"""Tests for `tsu skill` CLI subcommands."""

from pathlib import Path

from typer.testing import CliRunner

from tsugite.cli import app

runner = CliRunner()


def _write_skill_md(directory: Path, name: str, body: str = "Body.\n") -> Path:
    skill_dir = directory / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(body)
    return skill_md


class TestSkillCheck:
    def test_help(self):
        result = runner.invoke(app, ["skill", "--help"])
        assert result.exit_code == 0
        assert "skill" in result.stdout.lower()

    def test_check_help(self):
        result = runner.invoke(app, ["skill", "check", "--help"])
        assert result.exit_code == 0
        assert "check" in result.stdout.lower()

    def test_clean_workspace_exits_zero(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        skills_root = tmp_path / ".tsugite" / "skills"
        skills_root.mkdir(parents=True)
        _write_skill_md(
            skills_root,
            "clean-skill",
            "---\nname: clean-skill\ndescription: ok\n---\nBody.\n",
        )
        result = runner.invoke(app, ["skill", "check"])
        assert result.exit_code == 0
        assert "no issues" in result.stdout.lower()

    def test_warning_issue_exits_zero(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        skills_root = tmp_path / ".tsugite" / "skills"
        skills_root.mkdir(parents=True)
        _write_skill_md(
            skills_root,
            "bad-trig",
            "---\nname: bad-trig\ndescription: x\ntriggers:\n  - 403\n---\nBody.\n",
        )
        result = runner.invoke(app, ["skill", "check"])
        assert result.exit_code == 0
        assert "bad-trig" in result.stdout
        assert "warning" in result.stdout.lower()

    def test_error_issue_exits_nonzero(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        skills_root = tmp_path / ".tsugite" / "skills"
        skills_root.mkdir(parents=True)
        _write_skill_md(
            skills_root,
            "broken",
            "---\nname: broken\ninvalid: yaml: structure: bad\n---\nBody\n",
        )
        result = runner.invoke(app, ["skill", "check"])
        assert result.exit_code == 1
        assert "error" in result.stdout.lower()

    def test_path_option_scopes_scan(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        custom = tmp_path / "custom_skills"
        custom.mkdir()
        _write_skill_md(
            custom,
            "ext-skill",
            "---\nname: ext-skill\ndescription: from custom path\n---\nBody.\n",
        )
        result = runner.invoke(app, ["skill", "check", "--path", str(custom)])
        assert result.exit_code == 0
