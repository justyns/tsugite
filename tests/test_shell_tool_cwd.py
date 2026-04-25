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


def test_run_passes_env_to_subprocess(workspace):
    set_workspace_dir(workspace)
    output = shell_run("echo got=$FOO", env={"FOO": "bar"})
    assert "got=bar" in output


def test_run_env_merges_with_os_environ(workspace, monkeypatch):
    monkeypatch.setenv("BASELINE", "yes")
    set_workspace_dir(workspace)
    output = shell_run("printf '%s/%s' \"$BASELINE\" \"$FOO\"", env={"FOO": "bar"})
    assert "yes/bar" in output


def test_run_env_does_not_pollute_parent_environ(workspace, monkeypatch):
    import os

    monkeypatch.delenv("ONLY_IN_CHILD", raising=False)
    set_workspace_dir(workspace)
    shell_run("true", env={"ONLY_IN_CHILD": "child"})
    assert "ONLY_IN_CHILD" not in os.environ


def test_run_env_none_preserves_old_behavior(workspace, monkeypatch):
    monkeypatch.setenv("PARENT_VAR", "yes")
    set_workspace_dir(workspace)
    output = shell_run("echo parent=$PARENT_VAR")
    assert "parent=yes" in output


def test_execute_shell_command_env_kwarg(workspace):
    output = execute_shell_command("echo $FOO", cwd=workspace, env={"FOO": "x"})
    assert "x" in output


def test_execute_shell_command_env_kwarg_shell_false(workspace):
    output = execute_shell_command("printenv FOO", cwd=workspace, shell=False, env={"FOO": "y"})
    assert "y" in output
