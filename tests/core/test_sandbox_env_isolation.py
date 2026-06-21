"""The bwrap sandbox must not leak the operator's environment (API keys, tokens) to agent code.

Secrets reach an agent only via the allowlisted, masked get_secret tool; raw os.environ access
inside the sandbox would bypass that entirely.
"""

import asyncio

import pytest

from tsugite.core.sandbox import SandboxConfig, sandbox_available
from tsugite.core.subprocess_executor import SubprocessExecutor

pytestmark = pytest.mark.skipif(not sandbox_available("bwrap"), reason="bwrap not installed")


def _run_sandboxed(tmp_path, code):
    ex = SubprocessExecutor(workspace_dir=tmp_path, sandbox_config=SandboxConfig(no_network=True))
    try:
        ex.set_tools([])
        return asyncio.run(ex.execute(code))
    finally:
        ex.cleanup()


def ex_run(ex, code):
    return asyncio.run(ex.execute(code))


def test_secret_env_var_not_visible_in_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_FAKE_API_KEY", "supersecret-do-not-leak")
    result = _run_sandboxed(tmp_path, 'import os; print("VAL=" + repr(os.environ.get("MY_FAKE_API_KEY")))')
    assert result.error is None, result.error
    assert "supersecret-do-not-leak" not in result.output
    assert "VAL=None" in result.output


def test_sandbox_still_has_basic_env(tmp_path):
    """clearenv must not break the child: PATH/HOME and IPC still work."""
    result = _run_sandboxed(tmp_path, 'import os; print("PATH_OK", bool(os.environ.get("PATH")))')
    assert result.error is None, result.error
    assert "PATH_OK True" in result.output


def test_pass_env_exposes_named_var(tmp_path, monkeypatch):
    """Operator-opted-in env vars (pass_env) are visible in the sandbox; others still aren't."""
    monkeypatch.setenv("MY_OPTED_IN_VAR", "visible-value")
    monkeypatch.setenv("MY_OTHER_SECRET", "still-hidden")
    ex = SubprocessExecutor(
        workspace_dir=tmp_path,
        sandbox_config=SandboxConfig(no_network=True, pass_env=["MY_OPTED_IN_VAR"]),
    )
    try:
        ex.set_tools([])
        result = ex_run(
            ex,
            'import os; print("IN=" + repr(os.environ.get("MY_OPTED_IN_VAR")), "OTHER=" + repr(os.environ.get("MY_OTHER_SECRET")))',
        )
        assert result.error is None, result.error
        assert "visible-value" in result.output
        assert "still-hidden" not in result.output
    finally:
        ex.cleanup()


def test_pass_env_xdg_override_wins(tmp_path, monkeypatch):
    """Passing an XDG var through keeps the operator's value instead of the tmpfs default."""
    monkeypatch.setenv("XDG_CACHE_HOME", "/home/op/.cache")
    ex = SubprocessExecutor(
        workspace_dir=tmp_path,
        sandbox_config=SandboxConfig(no_network=True, pass_env=["XDG_CACHE_HOME"]),
    )
    try:
        ex.set_tools([])
        result = ex_run(ex, 'import os; print("XDG=" + os.environ.get("XDG_CACHE_HOME", ""))')
        assert result.error is None, result.error
        assert "XDG=/home/op/.cache" in result.output
    finally:
        ex.cleanup()


def test_sandbox_xdg_dirs_isolated_and_writable(tmp_path):
    """XDG dirs point into the sandbox (not the operator's home) and are writable."""
    code = (
        "import os; p = os.environ.get('XDG_CACHE_HOME', ''); "
        "os.makedirs(os.path.join(p, 'sub'), exist_ok=True); "
        "print('XDG', p, os.path.isdir(os.path.join(p, 'sub')))"
    )
    result = _run_sandboxed(tmp_path, code)
    assert result.error is None, result.error
    assert "XDG /tmp/.cache True" in result.output
