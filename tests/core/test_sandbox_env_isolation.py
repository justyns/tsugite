"""The sandbox must not leak the operator's environment (API keys, tokens) to agent code.

Secrets reach an agent only via the allowlisted, masked get_secret tool; raw os.environ access
inside the sandbox would bypass that entirely.
"""

import asyncio
import os

import pytest

from tsugite.core.sandbox import SandboxConfig, sandbox_available
from tsugite.core.subprocess_executor import SubprocessExecutor

pytestmark = pytest.mark.skipif(not sandbox_available(), reason="sandbox backend not available")


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
    """XDG dirs are writable, and writes there stay off the host."""
    code = (
        "import os; p = os.environ.get('XDG_CACHE_HOME', ''); "
        "os.makedirs(os.path.join(p, 'sub'), exist_ok=True); "
        "print('XDG', p, os.path.isdir(os.path.join(p, 'sub')))"
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()
    result = _run_sandboxed(workspace, code)
    assert result.error is None, result.error
    _, cache, made = result.output.split()
    assert made == "True"
    assert cache.startswith(os.path.expanduser("~"))
    assert not os.path.exists(os.path.join(cache, "sub"))
