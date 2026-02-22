"""Bubblewrap (bwrap) sandbox wrapper.

Builds bwrap command lines for isolating agent subprocess execution
with filesystem restrictions and network namespace isolation.
"""

import shlex
import shutil
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""

    allowed_domains: List[str] = field(default_factory=list)
    no_network: bool = False
    extra_ro_binds: List[Path] = field(default_factory=list)
    extra_rw_binds: List[Path] = field(default_factory=list)


# TCP→UDS bridge script: listens on localhost, forwards to UDS proxy socket.
# Runs as a background process inside the sandbox so HTTP_PROXY works.
_TCP_UDS_BRIDGE = textwrap.dedent("""\
    import socket, threading, sys
    PROXY_SOCK = '/run/proxy.sock'
    LISTEN_PORT = 12345

    def handle(client):
        try:
            upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            upstream.connect(PROXY_SOCK)
            def relay(src, dst):
                try:
                    while True:
                        data = src.recv(8192)
                        if not data:
                            break
                        dst.sendall(data)
                except Exception:
                    pass
                finally:
                    try: dst.shutdown(socket.SHUT_WR)
                    except Exception: pass
            t1 = threading.Thread(target=relay, args=(client, upstream), daemon=True)
            t2 = threading.Thread(target=relay, args=(upstream, client), daemon=True)
            t1.start(); t2.start()
            t1.join(); t2.join()
        except Exception:
            pass
        finally:
            client.close()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', LISTEN_PORT))
    srv.listen(16)
    while True:
        c, _ = srv.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()
""")

# SSH ProxyCommand helper: connects to the HTTP CONNECT proxy to tunnel SSH.
# Usage: ssh -o ProxyCommand='python3 _ssh_proxy.py %h %p' user@host
_SSH_PROXY_CONNECT = textwrap.dedent("""\
    import socket, sys
    host, port = sys.argv[1], sys.argv[2]
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('127.0.0.1', 12345))
    s.sendall(f'CONNECT {host}:{port} HTTP/1.1\\r\\nHost: {host}:{port}\\r\\n\\r\\n'.encode())
    resp = b''
    while b'\\r\\n\\r\\n' not in resp:
        resp += s.recv(4096)
    if b'200' not in resp.split(b'\\r\\n')[0]:
        sys.exit(1)
    import os, select
    while True:
        r, _, _ = select.select([sys.stdin.buffer, s], [], [])
        if sys.stdin.buffer in r:
            data = os.read(sys.stdin.buffer.fileno(), 8192)
            if not data: break
            s.sendall(data)
        if s in r:
            data = s.recv(8192)
            if not data: break
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
""")


class BubblewrapSandbox:
    """Wraps a command with bubblewrap for filesystem and network isolation.

    Args:
        config: Sandbox configuration
        proxy_socket: Path to the HTTP proxy Unix domain socket
        workspace_dir: Working directory for the sandboxed process
        state_dir: Directory for harness temp files (read-write)
    """

    def __init__(
        self,
        config: SandboxConfig,
        proxy_socket: Optional[Path] = None,
        workspace_dir: Optional[Path] = None,
        state_dir: Optional[Path] = None,
    ):
        self.config = config
        self.proxy_socket = proxy_socket
        self.workspace_dir = workspace_dir or Path.cwd()
        self.state_dir = state_dir

    @property
    def _use_proxy(self) -> bool:
        """Whether to set up network proxy (socket provided and network not fully disabled)."""
        return bool(self.proxy_socket and not self.config.no_network)

    def build_command(self, inner_cmd: List[str]) -> List[str]:
        """Wrap inner command with bwrap arguments.

        Args:
            inner_cmd: The command to run inside the sandbox

        Returns:
            Full command list starting with 'bwrap'
        """
        cmd = ["bwrap"]

        cmd += ["--unshare-pid", "--die-with-parent"]
        if self.config.no_network or self.proxy_socket:
            cmd += ["--unshare-net"]

        # Basic filesystem
        cmd += ["--tmpfs", "/tmp", "--proc", "/proc", "--dev", "/dev"]

        # System read-only binds
        for sys_dir in ["/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc/ssl", "/etc/resolv.conf"]:
            p = Path(sys_dir)
            if p.exists():
                cmd += ["--ro-bind", sys_dir, sys_dir]

        # CA certificates: resolve symlinks to find the real path (varies by distro)
        for cert_path in ["/etc/ssl/certs/ca-certificates.crt", "/etc/ssl/certs/ca-bundle.crt", "/etc/pki/tls/certs/ca-bundle.crt"]:
            p = Path(cert_path)
            if p.exists():
                real = p.resolve().parent
                if real.exists() and not any(str(real).startswith(d) for d in ["/usr", "/etc/ssl"]):
                    cmd += ["--ro-bind", str(real), str(real)]
                break

        # Python venv (read-only)
        venv_path = Path(sys.prefix)
        if venv_path.exists():
            cmd += ["--ro-bind", str(venv_path), str(venv_path)]

        # If venv is a symlink or different from base prefix, bind base too
        base_prefix = Path(sys.base_prefix)
        if base_prefix != venv_path and base_prefix.exists():
            cmd += ["--ro-bind", str(base_prefix), str(base_prefix)]

        # Editable installs: venv .pth files reference the project source directory.
        # Bind all sys.path entries so imports work inside the sandbox.
        bound = {str(venv_path), str(base_prefix), str(self.workspace_dir)}
        for p in sys.path:
            if not p or p in bound:
                continue
            pp = Path(p)
            if pp.exists() and str(pp) not in bound:
                cmd += ["--ro-bind", str(pp), str(pp)]
                bound.add(str(pp))

        # Project/workspace directory (read-write)
        cmd += ["--bind", str(self.workspace_dir), str(self.workspace_dir)]

        # State directory for harness files (read-write)
        if self.state_dir:
            cmd += ["--bind", str(self.state_dir), str(self.state_dir)]

        if self._use_proxy:
            cmd += ["--ro-bind", str(self.proxy_socket), "/run/proxy.sock"]

        # Extra binds
        for path in self.config.extra_ro_binds:
            if path.exists():
                cmd += ["--ro-bind", str(path), str(path)]
        for path in self.config.extra_rw_binds:
            if path.exists():
                cmd += ["--bind", str(path), str(path)]

        # Working directory
        cmd += ["--chdir", str(self.workspace_dir)]

        if self._use_proxy:
            scripts_dir = self.state_dir or Path("/tmp")

            # Write bridge + SSH proxy scripts (state_dir is bind-mounted into sandbox)
            bridge_path = scripts_dir / "_bridge.py"
            bridge_path.write_text(_TCP_UDS_BRIDGE)
            ssh_proxy_path = scripts_dir / "_ssh_proxy.py"
            ssh_proxy_path.write_text(_SSH_PROXY_CONNECT)

            cmd += [
                "--setenv", "HTTP_PROXY", "http://127.0.0.1:12345",
                "--setenv", "HTTPS_PROXY", "http://127.0.0.1:12345",
                "--setenv", "GIT_SSH_COMMAND", f"ssh -o StrictHostKeyChecking=accept-new -o ProxyCommand='python3 {ssh_proxy_path} %h %p'",
            ]
            actual_cmd = shlex.join(inner_cmd)
            # Start bridge in background, wait for it to bind, then run harness
            cmd += [
                "sh", "-c",
                f"python3 {bridge_path} & sleep 0.1; exec {actual_cmd}",
            ]
        else:
            cmd += inner_cmd

        return cmd

    @staticmethod
    def check_available() -> bool:
        """Check if bwrap is installed."""
        return shutil.which("bwrap") is not None
