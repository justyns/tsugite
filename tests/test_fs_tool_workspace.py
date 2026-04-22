"""Fs tools must resolve relative paths against the workspace ContextVar."""

import pytest

from tsugite.cli.helpers import set_workspace_dir
from tsugite.tools.fs import (
    create_directory,
    edit_file,
    file_exists,
    get_file_info,
    list_files,
    read_file,
    write_file,
)


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "notes.txt").write_text("hello world\n")
    return ws


def test_read_file_relative_resolves_against_workspace(workspace, tmp_path, monkeypatch):
    """Relative path must resolve against the workspace CV, not the process cwd."""
    # chdir somewhere else to prove we aren't silently using cwd
    monkeypatch.chdir(tmp_path)
    set_workspace_dir(workspace)
    assert read_file("notes.txt").strip() == "hello world"


def test_write_file_relative_writes_into_workspace(workspace, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    set_workspace_dir(workspace)
    write_file("new_file.txt", "contents")
    assert (workspace / "new_file.txt").read_text() == "contents"
    assert not (tmp_path / "new_file.txt").exists()


def test_absolute_path_passes_through(workspace, tmp_path):
    """Absolute paths must ignore the workspace CV."""
    elsewhere = tmp_path / "elsewhere.txt"
    elsewhere.write_text("outside")
    set_workspace_dir(workspace)
    assert read_file(str(elsewhere)).strip() == "outside"


def test_list_files_relative_resolves_against_workspace(workspace, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    set_workspace_dir(workspace)
    assert "notes.txt" in list_files(".")


def test_file_exists_relative(workspace, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    set_workspace_dir(workspace)
    assert file_exists("notes.txt") is True
    assert file_exists("nope.txt") is False


def test_create_directory_relative(workspace, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    set_workspace_dir(workspace)
    create_directory("subdir")
    assert (workspace / "subdir").is_dir()


def test_get_file_info_relative(workspace, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    set_workspace_dir(workspace)
    info = get_file_info("notes.txt")
    assert info["exists"] is True
    assert info["line_count"] == 1


def test_edit_file_relative(workspace, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    set_workspace_dir(workspace)
    edit_file("notes.txt", old_string="hello", new_string="hi")
    assert (workspace / "notes.txt").read_text().strip() == "hi world"


def test_relative_paths_use_cwd_when_no_workspace(tmp_path, monkeypatch):
    """Fallback: no workspace CV → relative paths use process cwd."""
    (tmp_path / "local.txt").write_text("local")
    monkeypatch.chdir(tmp_path)
    # Intentionally do NOT call set_workspace_dir
    assert read_file("local.txt").strip() == "local"
