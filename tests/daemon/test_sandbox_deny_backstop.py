"""Tests for the sandbox policy context + deny backstop (zero-escape invariant).

The deny backstop guarantees that while an agent runs sandboxed, no host-exec /
spawn tool reaches the host via an unsandboxed path: covered paths are allowed
(they wrap in bwrap or inherit the sandbox), everything else is refused.
"""

from pathlib import Path

import pytest

from tsugite.agent_runner.helpers import (
    SandboxContext,
    SandboxToolDeniedError,
    clear_sandbox_context,
    get_sandbox_context,
    set_sandbox_context,
)


@pytest.fixture(autouse=True)
def _clear_ctx():
    clear_sandbox_context()
    yield
    clear_sandbox_context()


class TestSandboxContext:
    def test_round_trip(self):
        assert get_sandbox_context() is None
        ctx = SandboxContext(allow_domains=["github.com"], no_network=True)
        set_sandbox_context(ctx)
        assert get_sandbox_context() is ctx
        clear_sandbox_context()
        assert get_sandbox_context() is None


class TestDenyWhenSandboxed:
    def _decorated(self):
        from tsugite.tools import deny_when_sandboxed

        @deny_when_sandboxed
        def some_tool():
            return "ran"

        return some_tool

    def test_allowed_when_not_sandboxed(self):
        assert self._decorated()() == "ran"

    def test_denied_when_sandboxed(self):
        set_sandbox_context(SandboxContext())
        with pytest.raises(SandboxToolDeniedError, match="some_tool"):
            self._decorated()()


class TestRequireDaemonRoutesToParent:
    """require_daemon tools need the daemon runtime, which only exists in the
    parent process - so under the SubprocessExecutor they must route to the
    parent (via IPC), not run in the sandboxed child where there is no runtime."""

    def test_require_daemon_tool_routes_to_parent(self):
        from tsugite.core.subprocess_executor import SubprocessExecutor
        from tsugite.core.tools import create_tool_from_tsugite
        from tsugite.tools import set_daemon_mode

        set_daemon_mode(True)
        try:
            ex = SubprocessExecutor()
            wrapper = create_tool_from_tsugite("pty_create")
            ex.set_tools([wrapper])
            assert "pty_create" in ex._parent_only_tools
            assert "pty_create" not in ex._local_tools
        finally:
            set_daemon_mode(False)


class TestSpawnAgentInheritsSandbox:
    def test_no_sandbox_no_flags(self):
        from tsugite.tools.agents import _build_subagent_cmd

        cmd = _build_subagent_cmd(Path("/x/agent.md"), None, None)
        assert "--sandbox" not in cmd

    def test_sandbox_and_domains_propagated(self):
        from tsugite.tools.agents import _build_subagent_cmd

        ctx = SandboxContext(allow_domains=["github.com", "pypi.org"], no_network=False)
        cmd = _build_subagent_cmd(Path("/x/agent.md"), None, ctx)
        assert "--sandbox" in cmd
        assert cmd.count("--allow-domain") == 2
        assert "github.com" in cmd and "pypi.org" in cmd
        assert "--no-network" not in cmd

    def test_no_network_propagated(self):
        from tsugite.tools.agents import _build_subagent_cmd

        ctx = SandboxContext(no_network=True)
        cmd = _build_subagent_cmd(Path("/x/agent.md"), None, ctx)
        assert "--sandbox" in cmd
        assert "--no-network" in cmd


class TestPtySandboxing:
    def test_no_context_returns_argv_unchanged(self):
        from tsugite.daemon.terminal_runtime import maybe_sandbox_argv

        argv = ["/bin/sh", "-c", "ls"]
        assert maybe_sandbox_argv(argv, None, None) == argv

    def test_sandboxed_wraps_in_bwrap(self, tmp_path):
        from tsugite.daemon.terminal_runtime import maybe_sandbox_argv

        wrapped = maybe_sandbox_argv(["/bin/sh", "-c", "ls"], None, SandboxContext(workspace_dir=tmp_path))
        assert wrapped[0] == "bwrap"
        assert str(tmp_path) in wrapped
        # Sandboxed PTYs get no network (no filtering proxy is wired for PTYs).
        assert "--unshare-net" in wrapped
        # The original command is still present at the tail.
        assert wrapped[-3:] == ["/bin/sh", "-c", "ls"]

    def test_sandboxed_without_workspace_fails_closed(self):
        from tsugite.daemon.terminal_runtime import maybe_sandbox_argv

        with pytest.raises(RuntimeError, match="workspace"):
            maybe_sandbox_argv(["/bin/sh", "-c", "ls"], None, SandboxContext(workspace_dir=None))


class TestResolveTerminalSandbox:
    """resolve_terminal_sandbox picks the running agent's thread-local policy, else
    the parent session's agent config (so /run and API terminals are sandboxed)."""

    def test_thread_local_context_wins(self):
        from tsugite.daemon import terminal_runtime

        ctx = SandboxContext(workspace_dir=None, allow_domains=["a.com"])
        set_sandbox_context(ctx)
        assert terminal_runtime.resolve_terminal_sandbox("any-session") is ctx

    def test_falls_back_to_session_resolver(self, monkeypatch):
        from tsugite.daemon import terminal_runtime

        resolved = SandboxContext(allow_domains=["b.com"])
        monkeypatch.setattr(
            terminal_runtime, "_session_sandbox_resolver", lambda sid: resolved if sid == "s1" else None
        )
        clear_sandbox_context()
        assert terminal_runtime.resolve_terminal_sandbox("s1") is resolved
        assert terminal_runtime.resolve_terminal_sandbox("other") is None

    def test_none_when_no_context_and_no_resolver(self, monkeypatch):
        from tsugite.daemon import terminal_runtime

        monkeypatch.setattr(terminal_runtime, "_session_sandbox_resolver", None)
        clear_sandbox_context()
        assert terminal_runtime.resolve_terminal_sandbox("s1") is None


class TestScheduleToolsDeniedUnderSandbox:
    """Scheduling/background-tasking arranges persistent or detached host
    execution; a sandboxed agent must not be able to use it to escape."""

    def test_schedule_create_denied(self):
        from tsugite.tools.schedule import schedule_create

        set_sandbox_context(SandboxContext())
        with pytest.raises(SandboxToolDeniedError, match="schedule_create"):
            schedule_create(id="s1", prompt="x", cron="0 9 * * *")

    def test_background_task_denied(self):
        from tsugite.tools.schedule import background_task

        set_sandbox_context(SandboxContext())
        with pytest.raises(SandboxToolDeniedError, match="background_task"):
            background_task(prompt="x")

    def test_schedule_run_denied(self):
        from tsugite.tools.schedule import schedule_run

        set_sandbox_context(SandboxContext())
        with pytest.raises(SandboxToolDeniedError, match="schedule_run"):
            schedule_run(id="sched-1")
