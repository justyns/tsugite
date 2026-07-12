"""Tests for the explicit sandbox_ctx seam on spawn_terminal / maybe_sandbox_argv.

The resolved (agent-inherited) path always forces no_network=True - a sandboxed
agent's PTY must never get network. A caller may instead pass an explicit
sandbox context that honors its own no_network (so a driver can run a tool
filesystem-isolated but with network). These tests pin both behaviors.
"""

from __future__ import annotations

import sys

import pytest

from tsugite.agent_runner.helpers import SandboxContext, clear_sandbox_context

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="PTY/sandbox is POSIX-only")


class RecordingSandbox:
    """Fake sandbox backend: records the SandboxConfig it was built with and
    wraps argv with a sentinel prefix so a test can tell it wrapped."""

    last_config = None
    last_workspace = None

    def __init__(self, config, proxy_socket=None, workspace_dir=None, state_dir=None):
        RecordingSandbox.last_config = config
        RecordingSandbox.last_workspace = workspace_dir

    def build_command(self, inner_cmd):
        return ["FAKE-BWRAP", *inner_cmd]


@pytest.fixture(autouse=True)
def _clear_ctx():
    clear_sandbox_context()
    yield
    clear_sandbox_context()


@pytest.fixture
def fake_backend(monkeypatch):
    from tsugite.core import sandbox as core_sandbox

    RecordingSandbox.last_config = None
    RecordingSandbox.last_workspace = None
    monkeypatch.setattr(core_sandbox, "get_sandbox_class", lambda *a, **k: RecordingSandbox)
    return RecordingSandbox


class TestMaybeSandboxArgvForceNoNetwork:
    def test_force_no_network_true_forces_isolation(self, tmp_path, fake_backend):
        from tsugite_pty.terminal_runtime import maybe_sandbox_argv

        maybe_sandbox_argv(
            ["/bin/sh", "-c", "ls"],
            None,
            SandboxContext(workspace_dir=tmp_path, no_network=False),
            force_no_network=True,
        )
        # The resolved path must ignore the ctx's own no_network and force True.
        assert fake_backend.last_config.no_network is True

    def test_explicit_ctx_keeps_network(self, tmp_path, fake_backend):
        from tsugite_pty.terminal_runtime import maybe_sandbox_argv

        wrapped = maybe_sandbox_argv(
            ["/bin/sh", "-c", "ls"],
            None,
            SandboxContext(workspace_dir=tmp_path, no_network=False),
            force_no_network=False,
        )
        assert fake_backend.last_config.no_network is False
        assert wrapped[0] == "FAKE-BWRAP"
        assert wrapped[-3:] == ["/bin/sh", "-c", "ls"]

    def test_explicit_ctx_honors_its_own_no_network(self, tmp_path, fake_backend):
        from tsugite_pty.terminal_runtime import maybe_sandbox_argv

        maybe_sandbox_argv(
            ["/bin/sh", "-c", "ls"],
            None,
            SandboxContext(workspace_dir=tmp_path, no_network=True),
            force_no_network=False,
        )
        assert fake_backend.last_config.no_network is True

    def test_default_force_no_network_is_true(self, tmp_path, fake_backend):
        """Callers that don't pass force_no_network get the safe default (isolate)."""
        from tsugite_pty.terminal_runtime import maybe_sandbox_argv

        maybe_sandbox_argv(["/bin/sh", "-c", "ls"], None, SandboxContext(workspace_dir=tmp_path))
        assert fake_backend.last_config.no_network is True


class TestBubblewrapNetworkEvidence:
    """Direct evidence against the real bwrap backend: no_network=False without a
    proxy socket yields NO --unshare-net (full network + filesystem isolation)."""

    def test_no_unshare_net_when_network_allowed(self, tmp_path):
        pytest.importorskip("tsugite_sandbox")
        from tsugite_sandbox import BubblewrapSandbox

        from tsugite.core.sandbox import SandboxConfig

        cmd = BubblewrapSandbox(
            config=SandboxConfig(no_network=False),
            workspace_dir=tmp_path,
        ).build_command(["/bin/sh", "-c", "ls"])
        assert "--unshare-net" not in cmd

    def test_unshare_net_when_no_network(self, tmp_path):
        pytest.importorskip("tsugite_sandbox")
        from tsugite_sandbox import BubblewrapSandbox

        from tsugite.core.sandbox import SandboxConfig

        cmd = BubblewrapSandbox(
            config=SandboxConfig(no_network=True),
            workspace_dir=tmp_path,
        ).build_command(["/bin/sh", "-c", "ls"])
        assert "--unshare-net" in cmd


class TestSpawnTerminalSandboxRouting:
    """spawn_terminal routes the _UNSET (resolved) path through the forced-isolation
    branch, and an explicit ctx (including None) through the honor-its-own branch."""

    @pytest.fixture
    def runtime(self, tmp_path):
        from tsugite_pty.pty_manager import PtyManager
        from tsugite_pty.terminal_store import TerminalSessionStore

        mgr = PtyManager()
        store = TerminalSessionStore(tmp_path / "sessions.json")
        try:
            yield mgr, store
        finally:
            mgr.shutdown()

    def test_unset_uses_resolved_path_forcing_no_network(self, runtime, monkeypatch):
        from tsugite_pty import terminal_runtime

        calls = []

        def recorder(argv, cwd, ctx, force_no_network=True):
            calls.append((ctx, force_no_network))
            return argv

        resolved = SandboxContext(workspace_dir=None)
        monkeypatch.setattr(terminal_runtime, "maybe_sandbox_argv", recorder)
        monkeypatch.setattr(terminal_runtime, "resolve_terminal_sandbox", lambda sid: resolved)

        mgr, store = runtime
        terminal_runtime.spawn_terminal(store=store, manager=mgr, cmd="echo hi")

        assert calls == [(resolved, True)]

    def test_explicit_ctx_passes_force_no_network_false(self, runtime, monkeypatch):
        from tsugite_pty import terminal_runtime

        calls = []

        def recorder(argv, cwd, ctx, force_no_network=True):
            calls.append((ctx, force_no_network))
            return argv

        monkeypatch.setattr(terminal_runtime, "maybe_sandbox_argv", recorder)
        monkeypatch.setattr(
            terminal_runtime,
            "resolve_terminal_sandbox",
            lambda sid: pytest.fail("resolve_terminal_sandbox must not run on the explicit path"),
        )

        explicit = SandboxContext(workspace_dir=None, no_network=False)
        mgr, store = runtime
        terminal_runtime.spawn_terminal(store=store, manager=mgr, cmd="echo hi", sandbox_ctx=explicit)

        assert calls == [(explicit, False)]

    def test_explicit_none_is_unsandboxed(self, runtime, monkeypatch):
        from tsugite_pty import terminal_runtime

        monkeypatch.setattr(
            terminal_runtime,
            "resolve_terminal_sandbox",
            lambda sid: pytest.fail("resolve_terminal_sandbox must not run on the explicit path"),
        )

        mgr, store = runtime
        # sandbox_ctx=None -> maybe_sandbox_argv returns argv unchanged; no backend needed.
        session = terminal_runtime.spawn_terminal(store=store, manager=mgr, cmd="echo hi", sandbox_ctx=None)
        assert session is not None
