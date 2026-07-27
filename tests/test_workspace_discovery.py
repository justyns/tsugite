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


def test_workspace_create(tmp_path):
    """Test creating a new workspace."""
    workspace_path = tmp_path / "new-workspace"

    workspace = Workspace.create(workspace_path)

    assert workspace.path.exists()
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

    assert workspace.skills_dir == workspace_path / "skills"
    assert workspace.agents_dir == workspace_path / "agents"
