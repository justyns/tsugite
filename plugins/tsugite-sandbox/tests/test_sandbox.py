"""Tests for bubblewrap sandbox wrapper."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import tsugite_sandbox
from tsugite_sandbox import BubblewrapSandbox

from tsugite.core.sandbox import SandboxConfig


class TestBuildCommand:
    def test_basic_command(self):
        config = SandboxConfig()
        sandbox = BubblewrapSandbox(
            config=config,
            workspace_dir=Path("/home/user/project"),
        )
        cmd = sandbox.build_command(["python3", "harness.py"])
        assert cmd[0] == "bwrap"
        assert "--unshare-pid" in cmd
        assert "--die-with-parent" in cmd
        # Workspace should be rw bind
        idx = cmd.index("--bind")
        assert cmd[idx + 1] == "/home/user/project"

    def test_network_unshare(self, tmp_path):
        config = SandboxConfig()
        sandbox = BubblewrapSandbox(
            config=config,
            proxy_socket=Path("/tmp/proxy.sock"),
            state_dir=tmp_path,
        )
        cmd = sandbox.build_command(["python3", "harness.py"])
        assert "--unshare-net" in cmd

    def test_proxy_socket_binding(self, tmp_path):
        config = SandboxConfig()
        sandbox = BubblewrapSandbox(
            config=config,
            proxy_socket=Path("/tmp/proxy.sock"),
            state_dir=tmp_path,
        )
        cmd = sandbox.build_command(["python3", "harness.py"])
        # Proxy socket should be bound to /run/proxy.sock
        ro_bind_indices = [i for i, v in enumerate(cmd) if v == "--ro-bind"]
        proxy_bound = any(
            cmd[i + 1] == "/tmp/proxy.sock" and cmd[i + 2] == "/run/proxy.sock"
            for i in ro_bind_indices
            if i + 2 < len(cmd)
        )
        assert proxy_bound

    def test_proxy_env_vars(self, tmp_path):
        config = SandboxConfig()
        sandbox = BubblewrapSandbox(
            config=config,
            proxy_socket=Path("/tmp/proxy.sock"),
            state_dir=tmp_path,
        )
        cmd = sandbox.build_command(["python3", "harness.py"])
        setenv_indices = [i for i, v in enumerate(cmd) if v == "--setenv"]
        http_proxy_set = any(cmd[i + 1] == "HTTP_PROXY" for i in setenv_indices if i + 1 < len(cmd))
        https_proxy_set = any(cmd[i + 1] == "HTTPS_PROXY" for i in setenv_indices if i + 1 < len(cmd))
        assert http_proxy_set
        assert https_proxy_set

    def test_proxy_scripts_written_to_state_dir(self, tmp_path):
        sandbox = BubblewrapSandbox(
            config=SandboxConfig(),
            proxy_socket=Path("/tmp/proxy.sock"),
            state_dir=tmp_path,
        )
        sandbox.build_command(["python3", "harness.py"])
        assert (tmp_path / "_bridge.py").exists()
        assert (tmp_path / "_ssh_proxy.py").exists()

    def test_ssh_proxy_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SandboxConfig()
            sandbox = BubblewrapSandbox(
                config=config,
                proxy_socket=Path("/tmp/proxy.sock"),
                state_dir=Path(tmpdir),
            )
            cmd = sandbox.build_command(["python3", "harness.py"])
            setenv_indices = [i for i, v in enumerate(cmd) if v == "--setenv"]
            git_ssh_set = any(cmd[i + 1] == "GIT_SSH_COMMAND" for i in setenv_indices if i + 1 < len(cmd))
            assert git_ssh_set

    def test_state_dir_binding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SandboxConfig()
            sandbox = BubblewrapSandbox(config=config, state_dir=Path(tmpdir))
            cmd = sandbox.build_command(["python3", "harness.py"])
            bind_indices = [i for i, v in enumerate(cmd) if v == "--bind"]
            state_bound = any(cmd[i + 1] == tmpdir for i in bind_indices if i + 1 < len(cmd))
            assert state_bound

    def test_no_network_mode(self):
        config = SandboxConfig(no_network=True)
        sandbox = BubblewrapSandbox(config=config)
        cmd = sandbox.build_command(["python3", "harness.py"])
        assert "--unshare-net" in cmd
        assert "HTTP_PROXY" not in cmd

    def test_unrestricted_network_mode(self):
        """No proxy + no_network=False: sandbox has full network access."""
        config = SandboxConfig(no_network=False)
        sandbox = BubblewrapSandbox(config=config)
        cmd = sandbox.build_command(["python3", "harness.py"])
        assert "--unshare-net" not in cmd
        assert "HTTP_PROXY" not in cmd

    @pytest.mark.parametrize("var", ["NODE_EXTRA_CA_CERTS", "CURL_CA_BUNDLE", "GIT_SSL_CAINFO"])
    def test_ca_env_vars_pass_through(self, monkeypatch, var):
        monkeypatch.setenv(var, "/opt/corp/ca.pem")
        cmd = BubblewrapSandbox(config=SandboxConfig()).build_command(["true"])
        setenvs = {cmd[i + 1]: cmd[i + 2] for i, v in enumerate(cmd) if v == "--setenv"}
        assert setenvs.get(var) == "/opt/corp/ca.pem"

    def test_every_cert_bundle_target_bound(self, monkeypatch, tmp_path):
        links = []
        for i, name in enumerate(("a", "b", "b")):
            (tmp_path / name).mkdir(exist_ok=True)
            (tmp_path / name / "bundle.pem").touch()
            link = tmp_path / f"link{i}.pem"
            link.symlink_to(tmp_path / name / "bundle.pem")
            links.append(str(link))
        monkeypatch.setattr(tsugite_sandbox, "_CA_BUNDLE_PATHS", tuple(links))
        cmd = BubblewrapSandbox(config=SandboxConfig()).build_command(["true"])
        ro_binds = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--ro-bind"]
        assert ro_binds.count(str(tmp_path / "a")) == 1
        assert ro_binds.count(str(tmp_path / "b")) == 1

    def test_extra_ro_binds(self):
        """extra_ro_binds paths appear as --ro-bind in command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extra = Path(tmpdir)
            config = SandboxConfig(extra_ro_binds=[extra])
            sandbox = BubblewrapSandbox(config=config, workspace_dir=Path("/home/user/project"))
            cmd = sandbox.build_command(["python3", "harness.py"])
            ro_binds = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--ro-bind" and i + 1 < len(cmd)]
            assert str(extra) in ro_binds

    def test_extra_rw_binds(self):
        """extra_rw_binds paths appear as --bind (rw) in command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extra = Path(tmpdir)
            config = SandboxConfig(extra_rw_binds=[extra])
            sandbox = BubblewrapSandbox(config=config, workspace_dir=Path("/home/user/project"))
            cmd = sandbox.build_command(["python3", "harness.py"])
            rw_binds = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--bind" and i + 1 < len(cmd)]
            assert str(extra) in rw_binds

    def test_use_proxy_property(self):
        """_use_proxy is True only when proxy_socket set and no_network is False."""
        config = SandboxConfig()
        assert BubblewrapSandbox(config=config, proxy_socket=Path("/tmp/p.sock"))._use_proxy is True
        assert BubblewrapSandbox(config=config)._use_proxy is False
        config_no_net = SandboxConfig(no_network=True)
        assert BubblewrapSandbox(config=config_no_net, proxy_socket=Path("/tmp/p.sock"))._use_proxy is False


class TestCheckAvailable:
    def test_available(self):
        with patch("shutil.which", return_value="/usr/bin/bwrap"):
            assert BubblewrapSandbox.check_available() is True

    def test_not_available(self):
        with patch("shutil.which", return_value=None):
            assert BubblewrapSandbox.check_available() is False


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bwrap not installed")
class TestSandboxedExecution:
    @pytest.mark.asyncio
    async def test_basic_sandboxed_run(self):
        """Integration: actually run a command under bwrap."""
        import subprocess

        config = SandboxConfig(no_network=True)
        sandbox = BubblewrapSandbox(config=config, workspace_dir=Path.cwd())
        cmd = sandbox.build_command(["python3", "-c", "print('hello from sandbox')"])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0
        assert "hello from sandbox" in result.stdout
