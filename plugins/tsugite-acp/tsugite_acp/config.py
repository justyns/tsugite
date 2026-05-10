"""Resolve the ACP agent command, env, and cwd."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

DEFAULT_COMMAND: list[str] = ["npx", "-y", "@agentclientprotocol/claude-agent-acp"]


@dataclass(frozen=True)
class ACPCommandConfig:
    """Resolved command, env, and cwd for spawning an ACP agent subprocess."""

    command: str
    args: list[str]
    env: dict[str, str]
    cwd: str | None

    def argv(self) -> list[str]:
        return [self.command, *self.args]


def resolve_command(env_override: str | None = None, config: dict | None = None) -> ACPCommandConfig:
    """Resolve the ACP command using this precedence:

    1. `env_override` (or `TSUGITE_ACP_COMMAND`) parsed via shlex
    2. `config["command"]` from a workspace/user config dict
    3. `DEFAULT_COMMAND` (npx claude-agent-acp)
    """
    raw = env_override if env_override is not None else os.environ.get("TSUGITE_ACP_COMMAND")
    if raw:
        parts = shlex.split(raw)
    elif config and config.get("command"):
        c = config["command"]
        parts = shlex.split(c) if isinstance(c, str) else list(c)
    else:
        parts = list(DEFAULT_COMMAND)

    if not parts:
        raise ValueError("ACP command resolved to empty argv")

    cwd = (config or {}).get("cwd")
    env = _build_env((config or {}).get("env"))
    return ACPCommandConfig(command=parts[0], args=parts[1:], env=env, cwd=cwd)


def _build_env(extra: dict | None) -> dict[str, str]:
    """Inherit current env, then layer in any explicit entries from config."""
    env = dict(os.environ)
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


def workspace_cwd() -> str:
    """Best-effort cwd: tsugite workspace dir if set, else process cwd."""
    try:
        from tsugite.cli.helpers import get_workspace_dir
    except ImportError:
        return str(Path.cwd())
    ws = get_workspace_dir()
    return str(ws) if ws is not None else str(Path.cwd())
