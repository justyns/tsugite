"""Tests for convention-based workspace discovery."""

from pathlib import Path

import pytest

from tsugite.workspace import Workspace, WorkspaceNotFoundError


def test_workspace_load_valid(tmp_path):
    """Test loading a valid workspace directory."""
    workspace_path = tmp_path / "test-workspace"
    workspace_path.mkdir()

    workspace = Workspace.load(workspace_path)

    assert workspace.name == "test-workspace"
    assert workspace.path == workspace_path


def test_workspace_load_nonexistent():
    """Test loading nonexistent workspace raises error."""
    with pytest.raises(WorkspaceNotFoundError):
        Workspace.load(Path("/nonexistent/workspace"))


def test_workspace_load_file_not_directory(tmp_path):
    """Test loading a file instead of directory raises error."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("test")

    with pytest.raises(WorkspaceNotFoundError):
        Workspace.load(file_path)


def test_workspace_get_workspace_files_none_exist(tmp_path):
    """Test workspace with no identity files."""
    workspace_path = tmp_path / "empty-workspace"
    workspace_path.mkdir()

    workspace = Workspace.load(workspace_path)
    files = workspace.get_workspace_files()

    assert files == []


def test_workspace_get_workspace_files_some_exist(tmp_path):
    """Test workspace with some identity files."""
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()

    (workspace_path / "PERSONA.md").write_text("# Persona")
    (workspace_path / "USER.md").write_text("# User")

    workspace = Workspace.load(workspace_path)
    files = workspace.get_workspace_files()

    assert len(files) == 2
    assert any(f.name == "PERSONA.md" for f in files)
    assert any(f.name == "USER.md" for f in files)


def test_workspace_get_workspace_files_all_exist(tmp_path):
    """Test workspace with all identity files."""
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()

    for filename in ["PERSONA.md", "SOUL.md", "USER.md", "MEMORY.md", "IDENTITY.md", "AGENTS.md"]:
        (workspace_path / filename).write_text(f"# {filename}")

    workspace = Workspace.load(workspace_path)
    files = workspace.get_workspace_files()

    assert len(files) == 6


def test_workspace_get_memory_files_none_exist(tmp_path):
    """Test workspace with no memory files."""
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()

    workspace = Workspace.load(workspace_path)
    memory_files = workspace.get_memory_files()

    assert memory_files == []


def test_workspace_get_memory_files_with_date_files(tmp_path):
    """Test workspace with valid date-named memory files."""
    from datetime import datetime, timedelta

    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    memory_dir = workspace_path / "memory"
    memory_dir.mkdir()

    # Create files with valid date names
    today = datetime.now()
    for i in range(5):
        date = today - timedelta(days=i)
        filename = date.strftime("%Y-%m-%d.md")
        (memory_dir / filename).write_text(f"# Memory {filename}")

    workspace = Workspace.load(workspace_path)
    memory_files = workspace.get_memory_files(days=3)

    # Should get 3 days (0, 1, 2)
    assert len(memory_files) == 3


def test_workspace_get_memory_files_ignores_invalid_names(tmp_path):
    """Test workspace ignores non-date memory files."""
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    memory_dir = workspace_path / "memory"
    memory_dir.mkdir()

    # Create invalid filenames
    (memory_dir / "notes.md").write_text("# Notes")
    (memory_dir / "2024-13-01.md").write_text("# Invalid month")
    (memory_dir / "invalid-date.md").write_text("# Invalid")

    workspace = Workspace.load(workspace_path)
    memory_files = workspace.get_memory_files()

    assert memory_files == []


def test_workspace_create(tmp_path):
    """Test creating a new workspace."""
    workspace_path = tmp_path / "new-workspace"

    workspace = Workspace.create(workspace_path)

    assert workspace.path.exists()
    assert workspace.memory_dir.exists()
    assert workspace.sessions_dir.exists()
    assert workspace.skills_dir.exists()
    assert workspace.agents_dir.exists()


def test_workspace_create_with_template(tmp_path):
    """Test creating workspace with persona template."""
    workspace_path = tmp_path / "templated-workspace"

    workspace = Workspace.create(workspace_path, persona_template="minimal", user_name="Test User")

    persona_file = workspace.path / "PERSONA.md"
    user_file = workspace.path / "USER.md"

    assert persona_file.exists()
    assert user_file.exists()

    persona_content = persona_file.read_text()
    assert "Persona: Minimal" in persona_content

    user_content = user_file.read_text()
    assert "Test User" in user_content


def test_workspace_properties(tmp_path):
    """Test workspace path properties."""
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()

    workspace = Workspace.load(workspace_path)

    assert workspace.session_path == workspace_path / "session.jsonl"
    assert workspace.sessions_dir == workspace_path / "sessions"
    assert workspace.memory_dir == workspace_path / "memory"
    assert workspace.skills_dir == workspace_path / "skills"
    assert workspace.agents_dir == workspace_path / "agents"
