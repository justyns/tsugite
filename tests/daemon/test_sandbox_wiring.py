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

    def test_non_dict_override_ignored_falls_back_to_config(self):
        # A tampered/stray override (e.g. a string written via session_metadata)
        # must not disable the sandbox or crash - it's ignored, so the agent's
        # configured policy wins (fail closed to the daemon config).
        agent_sb = SandboxSettings(enabled=True, allow_domains=["github.com"])
        opts = resolve_sandbox_exec_options({"sandbox_override": "off"}, agent_sb)
        assert opts["sandbox"] is True
        assert opts["allow_domains"] == ["github.com"]


class TestResolveEffectiveSandbox:
    """Tighten-only frontmatter overrides layered on the daemon ceiling."""

    def _resolve(self, **kw):
        from tsugite.agent_runner.runner import resolve_effective_sandbox

        base = dict(
            daemon_enabled=False,
            daemon_domains=[],
            daemon_no_network=False,
            fm_network=None,
            fm_sandbox=None,
        )
        base.update(kw)
        return resolve_effective_sandbox(**base)

    def test_no_config_no_frontmatter_disabled(self):
        assert self._resolve() == (False, [], False)

    def test_frontmatter_can_enable_when_daemon_off(self):
        enabled, _, _ = self._resolve(fm_sandbox={"enabled": True})
        assert enabled is True

    def test_frontmatter_cannot_disable(self):
        # daemon on + frontmatter enabled:false must stay enabled (no loosening).
        enabled, _, _ = self._resolve(daemon_enabled=True, fm_sandbox={"enabled": False})
        assert enabled is True

    def test_frontmatter_can_force_no_network(self):
        _, _, no_net = self._resolve(daemon_enabled=True, fm_sandbox={"no_network": True})
        assert no_net is True

    def test_frontmatter_narrows_domains_within_ceiling(self):
        _, domains, _ = self._resolve(
            daemon_enabled=True,
            daemon_domains=["github.com", "pypi.org"],
            fm_sandbox={"allow_domains": ["github.com"]},
        )
        assert domains == ["github.com"]

    def test_empty_ceiling_means_frontmatter_caps_from_all(self):
        _, domains, _ = self._resolve(daemon_enabled=True, fm_network={"domains": ["github.com"]})
        assert domains == ["github.com"]

    def test_cannot_widen_beyond_ceiling(self):
        # daemon allows only github; frontmatter asking for pypi can't add it.
        _, domains, no_net = self._resolve(
            daemon_enabled=True, daemon_domains=["github.com"], fm_network={"domains": ["pypi.org"]}
        )
        assert "pypi.org" not in domains
        # Nothing in-ceiling remained -> no network granted (not all-allowed).
        assert domains == []
        assert no_net is True

    def test_no_frontmatter_leaves_daemon_domains(self):
        _, domains, _ = self._resolve(daemon_enabled=True, daemon_domains=["github.com"])
        assert domains == ["github.com"]

    def test_glob_ceiling_allows_matching_subdomain(self):
        # Daemon ceiling is a glob; a more specific agent domain is within it.
        _, domains, no_net = self._resolve(
            daemon_enabled=True,
            daemon_domains=["*.github.com"],
            fm_sandbox={"allow_domains": ["api.github.com"]},
        )
        assert domains == ["api.github.com"]
        assert no_net is False

    def test_wildcard_ceiling_allows_any_requested(self):
        _, domains, _ = self._resolve(
            daemon_enabled=True, daemon_domains=["*"], fm_sandbox={"allow_domains": ["api.github.com"]}
        )
        assert domains == ["api.github.com"]

    def test_cannot_widen_glob_beyond_specific_ceiling(self):
        # Ceiling is specific; agent asking for the broader glob can't widen.
        _, domains, no_net = self._resolve(
            daemon_enabled=True,
            daemon_domains=["api.github.com"],
            fm_sandbox={"allow_domains": ["*.github.com"]},
        )
        assert domains == []
        assert no_net is True

    def test_cannot_widen_port_beyond_default_ceiling(self):
        # daemon "github.com" allows only default 80/443; agent can't add port 22.
        _, domains, no_net = self._resolve(
            daemon_enabled=True,
            daemon_domains=["github.com"],
            fm_sandbox={"allow_domains": ["github.com:22"]},
        )
        assert domains == []
        assert no_net is True

    def test_wildcard_domain_still_caps_nonstandard_port(self):
        # daemon "*" allows any domain but only on default ports; ":22" is out.
        _, domains, no_net = self._resolve(
            daemon_enabled=True, daemon_domains=["*"], fm_sandbox={"allow_domains": ["evil.com:22"]}
        )
        assert domains == []
        assert no_net is True

    def test_explicit_port_ceiling_allows_matching_port(self):
        _, domains, no_net = self._resolve(
            daemon_enabled=True,
            daemon_domains=["github.com:22"],
            fm_sandbox={"allow_domains": ["github.com:22"]},
        )
        assert domains == ["github.com:22"]
        assert no_net is False

    def test_all_ports_ceiling_allows_any_port(self):
        _, domains, _ = self._resolve(
            daemon_enabled=True, daemon_domains=["*:*"], fm_sandbox={"allow_domains": ["github.com:22"]}
        )
        assert domains == ["github.com:22"]

    def test_split_port_ceiling_union_covers_default(self):
        # The proxy unions the allowlist: :80 + :443 together == default ports, so
        # an agent requesting "github.com" (default 80/443) is within the ceiling.
        _, domains, no_net = self._resolve(
            daemon_enabled=True,
            daemon_domains=["github.com:80", "github.com:443"],
            fm_sandbox={"allow_domains": ["github.com"]},
        )
        assert domains == ["github.com"]
        assert no_net is False

    def test_split_domain_and_port_ceiling_union(self):
        # Ports come from different matching ceiling patterns; their union covers it.
        _, domains, _ = self._resolve(
            daemon_enabled=True,
            daemon_domains=["*.github.com:443", "api.github.com:80"],
            fm_sandbox={"allow_domains": ["api.github.com"]},
        )
        assert domains == ["api.github.com"]

    def test_all_ports_request_rejected_by_finite_split_ceiling(self):
        # Agent asks for ALL ports (":*"); the split ceiling's union is the finite
        # {80, 443}, so an all-ports request can't be granted -> no network. Documents
        # the empty-port-set ("all ports") semantic.
        _, domains, no_net = self._resolve(
            daemon_enabled=True,
            daemon_domains=["github.com:80", "github.com:443"],
            fm_sandbox={"allow_domains": ["github.com:*"]},
        )
        assert domains == []
        assert no_net is True


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


class TestGatewaySessionSandboxResolver:
    """The terminal resolver must honor a session's inherited sandbox_override, not
    just the target agent's config - else a terminal opened for a sandboxed child
    session (whose agent has sandbox off) would run on the host."""

    def _gateway(self, tmp_path, agents):
        from tsugite.daemon.config import DaemonConfig
        from tsugite.daemon.gateway import Gateway
        from tsugite.daemon.session_store import SessionStore

        gw = Gateway(DaemonConfig(state_dir=tmp_path, agents=agents))
        gw._session_store = SessionStore(tmp_path / "s.json")
        return gw

    def test_inherited_override_wins_over_disabled_agent(self, tmp_path):
        from tsugite.daemon.config import AgentConfig
        from tsugite.daemon.session_store import Session, SessionSource

        gw = self._gateway(tmp_path, {"plain": AgentConfig(workspace_dir=tmp_path, agent_file="default")})
        gw._session_store.create_session(
            Session(
                id="child",
                agent="plain",  # agent config has sandbox OFF
                source=SessionSource.SPAWNED.value,
                metadata={"sandbox_override": {"enabled": True, "allow_domains": ["github.com"], "no_network": False}},
            )
        )
        ctx = gw._resolve_session_sandbox("child")
        assert ctx is not None
        assert ctx.allow_domains == ["github.com"]

    def test_agent_config_used_when_no_override(self, tmp_path):
        from tsugite.daemon.config import AgentConfig
        from tsugite.daemon.session_store import Session, SessionSource

        gw = self._gateway(
            tmp_path,
            {"boxed": AgentConfig(workspace_dir=tmp_path, agent_file="default", sandbox=SandboxSettings(enabled=True))},
        )
        gw._session_store.create_session(Session(id="s", agent="boxed", source=SessionSource.BACKGROUND.value))
        assert gw._resolve_session_sandbox("s") is not None

    def test_none_when_neither(self, tmp_path):
        from tsugite.daemon.config import AgentConfig
        from tsugite.daemon.session_store import Session, SessionSource

        gw = self._gateway(tmp_path, {"plain": AgentConfig(workspace_dir=tmp_path, agent_file="default")})
        gw._session_store.create_session(Session(id="s", agent="plain", source=SessionSource.BACKGROUND.value))
        assert gw._resolve_session_sandbox("s") is None


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
