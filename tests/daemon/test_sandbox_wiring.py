"""Tests for wiring daemon sandbox settings into agent execution.

Covers the two resolution helpers that route SandboxSettings into the single
ExecutionOptions chokepoint and fix the daemon workspace-bind:
  - base.resolve_sandbox_exec_options (config -> ExecutionOptions kwargs, with
    metadata inheritance override)
  - runner._resolve_workspace_dir (path_context fallback when no workspace object)
"""

from pathlib import Path

import pytest

from tsugite.core.sandbox import BubblewrapSandbox
from tsugite.daemon.adapters.base import resolve_sandbox_exec_options
from tsugite.daemon.config import AgentConfig, DaemonConfig, SandboxSettings
from tsugite.options import ExecutionOptions


class TestResolveSandboxExecOptions:
    def test_disabled_when_no_sandbox(self):
        opts = resolve_sandbox_exec_options(None, None)
        assert opts["sandbox"] is False
        assert opts["allow_domains"] == []
        assert opts["no_network"] is False
        # The resulting kwargs must construct a valid ExecutionOptions.
        ExecutionOptions(**opts)

    def test_from_agent_config(self):
        sb = SandboxSettings(enabled=True, no_network=True, allow_domains=["github.com"])
        opts = resolve_sandbox_exec_options(None, sb)
        assert opts["sandbox"] is True
        assert opts["no_network"] is True
        assert opts["allow_domains"] == ["github.com"]

    def test_extra_binds_passed_through(self):
        sb = SandboxSettings(
            enabled=True,
            extra_ro_binds=[Path("/etc/creds")],
            extra_rw_binds=[Path("/srv/scratch")],
        )
        opts = resolve_sandbox_exec_options(None, sb)
        assert opts["extra_ro_binds"] == [Path("/etc/creds")]
        assert opts["extra_rw_binds"] == [Path("/srv/scratch")]

    def test_metadata_override_wins_over_agent_config(self):
        # A spawning sandboxed agent stamps sandbox_override (a JSON dict) into
        # the message metadata; it must win over the target agent's own config so
        # the child stays sandboxed even if that agent isn't configured for it.
        agent_sb = SandboxSettings(enabled=False)
        meta = {"sandbox_override": {"enabled": True, "allow_domains": ["x.com"], "no_network": False}}
        opts = resolve_sandbox_exec_options(meta, agent_sb)
        assert opts["sandbox"] is True
        assert opts["allow_domains"] == ["x.com"]

    def test_metadata_override_accepts_settings_object(self):
        meta = {"sandbox_override": SandboxSettings(enabled=True, allow_domains=["y.com"])}
        opts = resolve_sandbox_exec_options(meta, None)
        assert opts["sandbox"] is True
        assert opts["allow_domains"] == ["y.com"]


class TestResolveWorkspaceDir:
    def test_workspace_object_wins(self):
        from types import SimpleNamespace

        from tsugite.agent_runner.runner import _resolve_workspace_dir

        ws = SimpleNamespace(path=Path("/ws/from/object"))
        pc = SimpleNamespace(workspace_dir=Path("/ws/from/context"))
        assert _resolve_workspace_dir(ws, pc) == Path("/ws/from/object")

    def test_falls_back_to_path_context(self):
        from types import SimpleNamespace

        from tsugite.agent_runner.runner import _resolve_workspace_dir

        pc = SimpleNamespace(workspace_dir=Path("/ws/from/context"))
        # This is the daemon case: no workspace object, only a path_context.
        assert _resolve_workspace_dir(None, pc) == Path("/ws/from/context")

    def test_none_when_neither(self):
        from tsugite.agent_runner.runner import _resolve_workspace_dir

        assert _resolve_workspace_dir(None, None) is None

    def test_none_when_path_context_has_no_workspace(self):
        from types import SimpleNamespace

        from tsugite.agent_runner.runner import _resolve_workspace_dir

        pc = SimpleNamespace(workspace_dir=None)
        assert _resolve_workspace_dir(None, pc) is None


class TestSandboxStartupCheck:
    def _config(self, tmp_path, **agent_sandboxes):
        agents = {
            name: AgentConfig(workspace_dir=tmp_path, agent_file="default", sandbox=sb)
            for name, sb in agent_sandboxes.items()
        }
        return DaemonConfig(state_dir=tmp_path, agents=agents)

    def test_raises_when_sandbox_enabled_and_bwrap_missing(self, tmp_path, monkeypatch):
        from tsugite.daemon.gateway import check_sandbox_prerequisites

        monkeypatch.setattr(BubblewrapSandbox, "check_available", staticmethod(lambda: False))
        config = self._config(tmp_path, boxed=SandboxSettings(enabled=True))
        with pytest.raises(RuntimeError, match="bwrap"):
            check_sandbox_prerequisites(config)
        # The error should name the offending agent so the operator can fix it.
        with pytest.raises(RuntimeError, match="boxed"):
            check_sandbox_prerequisites(config)

    def test_ok_when_bwrap_available(self, tmp_path, monkeypatch):
        from tsugite.daemon.gateway import check_sandbox_prerequisites

        monkeypatch.setattr(BubblewrapSandbox, "check_available", staticmethod(lambda: True))
        config = self._config(tmp_path, boxed=SandboxSettings(enabled=True))
        check_sandbox_prerequisites(config)  # must not raise

    def test_ok_when_no_sandbox_enabled(self, tmp_path, monkeypatch):
        from tsugite.daemon.gateway import check_sandbox_prerequisites

        monkeypatch.setattr(BubblewrapSandbox, "check_available", staticmethod(lambda: False))
        config = self._config(tmp_path, plain=SandboxSettings(enabled=False))
        check_sandbox_prerequisites(config)  # disabled sandbox => no requirement
