import socket
import subprocess
import sys
import threading

import hozo
import pytest
import tsugite_sandbox

from tsugite.core.sandbox import SandboxConfig, get_sandbox_class


def _build(config=None, **kwargs):
    return tsugite_sandbox.HozoSandbox(config=config or SandboxConfig(), **kwargs).build_command(["python3", "h.py"])


def _flag(cmd, name):
    """Values of a `--name=a,b` flag, [] for a bare `--name`, None when absent."""
    for tok in cmd[: cmd.index("--")]:
        if tok == name:
            return []
        if tok.startswith(name + "="):
            return tok.split("=", 1)[1].split(",")
    return None


def test_default_backend_builds_a_hozo_invocation(tmp_path):
    cmd = get_sandbox_class()(config=SandboxConfig(), workspace_dir=tmp_path).build_command(["python3", "h.py"])
    assert cmd[:5] == [sys.executable, "-m", "hozo", "--project", str(tmp_path)]
    assert cmd[-3:] == ["--", "python3", "h.py"]


class TestNetwork:
    def test_no_network_grants_none(self):
        assert _flag(_build(SandboxConfig(no_network=True)), "--allow-net") is None

    def test_allowed_domains_become_the_proxy_allowlist(self):
        cmd = _build(SandboxConfig(allowed_domains=["api.github.com", "*.example.com:22"]))
        assert _flag(cmd, "--allow-net") == ["api.github.com", "*.example.com:22"]

    def test_network_without_domains_proxies_everything(self):
        assert _flag(_build(), "--allow-net") == ["*:*"]


class TestBinds:
    def test_python_prefix_is_read_only(self):
        assert sys.prefix in _flag(_build(), "--allow-read")

    def test_interpreter_symlink_chain_is_readable(self, tmp_path, monkeypatch):
        # uv venvs link bin/python through a minor-version dir symlink to the real install.
        real = tmp_path / "cpython-3.11.15" / "bin"
        real.mkdir(parents=True)
        (real / "python3.11").touch()
        minor = tmp_path / "cpython-3.11"
        minor.symlink_to(real.parent)
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").symlink_to(minor / "bin" / "python3.11")
        monkeypatch.setattr(sys, "executable", str(venv_bin / "python"))
        assert str(minor) in _flag(_build(), "--allow-read")

    def test_workspace_on_sys_path_stays_writable(self, tmp_path, monkeypatch):
        monkeypatch.syspath_prepend(str(tmp_path))
        assert str(tmp_path) not in _flag(_build(workspace_dir=tmp_path), "--allow-read")

    def test_state_dir_and_extra_rw_binds_are_writable(self, tmp_path):
        extra = tmp_path / "extra"
        extra.mkdir()
        cmd = _build(SandboxConfig(extra_rw_binds=[extra]), state_dir=tmp_path)
        assert _flag(cmd, "--allow-write") == [str(tmp_path), str(extra)]

    def test_missing_extra_binds_are_skipped(self, tmp_path):
        extra = tmp_path / "extra"
        extra.mkdir()
        gone = tmp_path / "gone"
        cmd = _build(SandboxConfig(extra_ro_binds=[extra, gone], extra_rw_binds=[gone]))
        assert str(extra) in _flag(cmd, "--allow-read")
        assert str(gone) not in _flag(cmd, "--allow-read")
        assert _flag(cmd, "--allow-write") is None


def test_pass_env_names_are_allowed_through():
    names = _flag(_build(SandboxConfig(pass_env=["MY_VAR"])), "--allow-env")
    assert {"MY_VAR", "VIRTUAL_ENV", "SSL_CERT_FILE"} <= set(names)


@pytest.mark.skipif(not hozo.check_available(), reason="no sandbox runtime on this host")
class TestSandboxedExecution:
    def _run(self, config, code, tmp_path):
        cmd = tsugite_sandbox.HozoSandbox(config=config, workspace_dir=tmp_path).build_command(
            [sys.executable, "-c", code]
        )
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    def test_runs_in_the_workspace(self, tmp_path):
        r = self._run(SandboxConfig(no_network=True), "import os; print(os.getcwd())", tmp_path)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == str(tmp_path)

    def test_host_path_reaches_the_sandbox(self, tmp_path, monkeypatch):
        # cc-driver launches `claude` by bare name from a bound PATH dir.
        monkeypatch.setenv("PATH", f"/opt/tools/bin:{sys.prefix}/bin:/usr/bin:/bin")
        r = self._run(SandboxConfig(no_network=True), "import os; print(os.environ['PATH'])", tmp_path)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == f"/opt/tools/bin:{sys.prefix}/bin:/usr/bin:/bin"

    def test_proxy_enforces_the_domain_allowlist(self, tmp_path):
        server = socket.create_server(("127.0.0.1", 0))
        port = server.getsockname()[1]
        threading.Thread(target=lambda: server.accept()[0].close(), daemon=True).start()
        code = (
            "import os, socket, urllib.parse\n"
            "proxy = urllib.parse.urlsplit(os.environ['HTTPS_PROXY'])\n"
            "for host in ('localhost', 'blocked.invalid'):\n"
            "    s = socket.create_connection((proxy.hostname, proxy.port))\n"
            f"    s.sendall(f'CONNECT {{host}}:{port} HTTP/1.1\\r\\n\\r\\n'.encode())\n"
            "    print(host, s.recv(64).split(b'\\r\\n')[0].decode())\n"
        )
        try:
            r = self._run(SandboxConfig(allowed_domains=[f"localhost:{port}"]), code, tmp_path)
        finally:
            server.close()
        assert r.returncode == 0, r.stderr
        assert "localhost HTTP/1.1 200" in r.stdout
        assert "blocked.invalid HTTP/1.1 403" in r.stdout

    @pytest.mark.parametrize("no_network, tunneled", [(False, True), (True, False)])
    def test_git_ssh_tunnels_through_the_proxy_only_with_network(self, tmp_path, no_network, tunneled):
        code = "import os; print(os.environ.get('GIT_SSH_COMMAND', ''))"
        r = self._run(SandboxConfig(no_network=no_network), code, tmp_path)
        assert r.returncode == 0, r.stderr
        assert ("ProxyCommand" in r.stdout) is tunneled
