"""`execute_shell_command` and `tools.shell.run` must run in the workspace
bound to the current task via the workspace ContextVar."""

import pytest

from tsugite.cli.helpers import set_workspace_dir
from tsugite.tools.shell import run as shell_run
from tsugite.utils import execute_shell_command


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "marker.txt").write_text("hello")
    return ws


def test_execute_shell_command_honors_cwd_kwarg(workspace):
    output = execute_shell_command("pwd", cwd=workspace)
    assert str(workspace) in output


def test_execute_shell_command_uses_workspace_cv_by_default(workspace):
    set_workspace_dir(workspace)
    output = execute_shell_command("ls")
    assert "marker.txt" in output


def test_execute_shell_command_explicit_cwd_overrides_cv(workspace, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "only_in_other.txt").write_text("x")
    set_workspace_dir(workspace)
    output = execute_shell_command("ls", cwd=other)
    assert "only_in_other.txt" in output
    assert "marker.txt" not in output


def test_shell_tool_run_uses_workspace_cv(workspace):
    set_workspace_dir(workspace)
    output = shell_run("pwd")
    assert str(workspace) in output
