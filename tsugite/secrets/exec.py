"""Exec-based secret backend.

Runs a command to fetch secrets. Supports any CLI secret tool:
  command: ["pass", "show", "tsugite/{{ name }}"]
  command: ["op", "read", "op://vault/{{ name }}"]
  command: "cat secrets/{{ name }}"
"""

import re
import subprocess
from typing import Optional, Union

import jinja2

_jinja_env = jinja2.Environment()
_SAFE_NAME = re.compile(r"[A-Za-z0-9_\-]+")


def _render_cmd(template: Union[str, list[str]], context: dict) -> Union[str, list[str]]:
    if isinstance(template, list):
        return [_jinja_env.from_string(part).render(context) for part in template]
    return _jinja_env.from_string(template).render(context)


class ExecSecretBackend:
    """Fetches secrets by running a command with Jinja-templated arguments."""

    def __init__(self, config: dict):
        command = config.get("command") or ""
        if not command:
            raise ValueError("exec secrets backend requires 'command' in config")
        self._command: Union[str, list[str]] = command
        self._list_command: Optional[Union[str, list[str]]] = config.get("list_command")

    def get(self, name: str) -> str | None:
        if not _SAFE_NAME.fullmatch(name):
            return None
        cmd = _render_cmd(self._command, {"name": name})
        try:
            result = subprocess.run(
                cmd,
                shell=isinstance(cmd, str),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return None
            return result.stdout.strip()
        except (subprocess.TimeoutExpired, OSError):
            return None

    def list_names(self) -> list[str]:
        if not self._list_command:
            return []
        cmd = _render_cmd(self._list_command, {})
        try:
            result = subprocess.run(
                cmd,
                shell=isinstance(cmd, str),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return []
            return [line.strip() for line in result.stdout.splitlines() if line.strip()]
        except (subprocess.TimeoutExpired, OSError):
            return []

    def set(self, name: str, value: str) -> None:
        raise NotImplementedError("Cannot set secrets via exec backend")

    def delete(self, name: str) -> bool:
        raise NotImplementedError("Cannot delete secrets via exec backend")
