"""Tsugite plugin: sandbox backend that wraps commands in the `hozo` CLI (bubblewrap on
Linux, sandbox-exec on macOS). Registered as `hozo` in the `tsugite.sandbox` entry-point group.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

import hozo

from tsugite.core.sandbox import SandboxConfig

_CA_ENV = (
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
    "CURL_CA_BUNDLE",
    "GIT_SSL_CAINFO",
)

# Added to Hozo's own env allowlist (TERM, LANG, LC_*, TZ). Secrets reach sandboxed code
# only through the masked get_secret tool.
_ENV_ALLOWLIST = (
    "PATH",
    "LANGUAGE",
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONIOENCODING",
    "PYTHONDONTWRITEBYTECODE",
    "VIRTUAL_ENV",
    *_CA_ENV,
)


class HozoSandbox:
    def __init__(
        self,
        config: SandboxConfig,
        workspace_dir: Optional[Path] = None,
        state_dir: Optional[Path] = None,
    ):
        self.config = config
        self.workspace_dir = workspace_dir or Path.cwd()
        self.state_dir = state_dir

    def build_command(self, inner_cmd: List[str]) -> List[str]:
        cmd = [sys.executable, "-m", "hozo", "--project", str(self.workspace_dir)]

        if not self.config.no_network:
            # A bare --allow-net is host networking. `*:*` keeps the sandbox behind Hozo's proxy.
            cmd += ["--allow-net=" + ",".join(self.config.allowed_domains or ["*:*"]), "+ssh"]

        # sys.path covers editable installs, whose .pth files point at the source tree.
        # A read-only grant of the workspace would replace its writable bind.
        ca = [p for name in _CA_ENV for p in os.environ.get(name, "").split(":")]
        read = _existing(
            [
                sys.prefix,
                sys.base_prefix,
                *_interpreter_link_prefixes(sys.executable),
                *sys.path,
                *ca,
                *self.config.extra_ro_binds,
            ]
        )
        read = [p for p in read if p != str(self.workspace_dir)]
        write = _existing([self.state_dir, *self.config.extra_rw_binds])
        if read:
            cmd.append("--allow-read=" + ",".join(read))
        if write:
            cmd.append("--allow-write=" + ",".join(write))

        cmd.append("--allow-env=" + ",".join([*_ENV_ALLOWLIST, *self.config.pass_env]))
        return [*cmd, "--", *inner_cmd]

    @staticmethod
    def check_available() -> bool:
        return hozo.check_available()


def _interpreter_link_prefixes(path: str) -> List[str]:
    """Prefix (the dir above bin/) of each symlink hop from the interpreter, as the links spell it.

    Python searches for its stdlib under the prefix of the path it was launched through.
    """
    prefixes = []
    while os.path.islink(path):
        path = os.path.join(os.path.dirname(path), os.readlink(path))
        prefixes.append(os.path.dirname(os.path.dirname(path)))
    return prefixes


def _existing(paths) -> List[str]:
    out: List[str] = []
    for p in paths:
        if p and Path(p).exists() and str(p) not in out:
            out.append(str(p))
    return out
