"""Tmux session management tools for running interactive CLIs with automatic logging."""

import json
import os
import re
import shlex
import shutil
import subprocess
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tsugite.config import get_xdg_data_path
from tsugite.tools import tool

SESSION_PREFIX = "tsu-"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07|\x1b\(B")
SHELLS = {"bash", "zsh", "sh", "fish", "dash", "ksh", "csh", "tcsh"}
VALID_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _prefixed(name: str) -> str:
    return f"{SESSION_PREFIX}{name}"


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _validate_name(name: str) -> None:
    if not VALID_NAME_RE.match(name):
        raise ValueError(
            f"Invalid session name '{name}': only alphanumeric, hyphens, and underscores allowed"
        )


def _session_exists(prefixed_name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", prefixed_name],
        capture_output=True,
    )
    return result.returncode == 0


def _get_metadata_path() -> Path:
    return get_xdg_data_path("tmux") / "sessions.json"


def _load_metadata() -> dict:
    path = _get_metadata_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_metadata(data: dict) -> None:
    path = _get_metadata_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(str(tmp), str(path))


def _get_log_dir() -> Path:
    return get_xdg_data_path("tmux-logs")


def _get_pane_command(prefixed_name: str) -> str:
    result = subprocess.run(
        ["tmux", "list-panes", "-t", prefixed_name, "-F", "#{pane_current_command}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip().split("\n")[0]


def _get_session_status(prefixed_name: str) -> str:
    cmd = _get_pane_command(prefixed_name)
    if not cmd or cmd.lower() in SHELLS:
        return "idle"
    return f"active: {cmd}"


def _list_managed_sessions() -> list:
    """Fetch all managed sessions with status in a single batched subprocess call."""
    result = subprocess.run(
        ["tmux", "list-panes", "-a", "-F", "#{session_name}\t#{pane_current_command}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []

    pane_cmds = {}
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        session_name = parts[0]
        if session_name.startswith(SESSION_PREFIX) and session_name not in pane_cmds:
            pane_cmds[session_name] = parts[1] if len(parts) > 1 else ""

    if not pane_cmds:
        return []

    metadata = _load_metadata()
    sessions = []
    for prefixed, cmd in pane_cmds.items():
        name = prefixed[len(SESSION_PREFIX) :]
        status = "idle" if (not cmd or cmd.lower() in SHELLS) else f"active: {cmd}"
        meta = metadata.get(name, {})
        sessions.append(
            {
                "name": name,
                "status": status,
                "created_at": meta.get("created_at", "unknown"),
                "command": meta.get("command"),
                "log_file": meta.get("log_file"),
            }
        )

    return sessions


@tool
def tmux_create(name: str, command: Optional[str] = None) -> dict:
    """Create a named tmux session with automatic output logging.

    Args:
        name: Session name (alphanumeric, hyphens, underscores)
        command: Initial command to run in the session (e.g., "htop", "python3")

    Returns:
        Dict with session name, tmux session name, log file path, and status
    """
    _validate_name(name)
    prefixed = _prefixed(name)

    if _session_exists(prefixed):
        raise RuntimeError(
            f"Session '{name}' already exists. Kill it first with tmux_kill('{name}') or use a different name."
        )

    log_dir = _get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    cmd = ["tmux", "new-session", "-d", "-s", prefixed, "-x", "200", "-y", "50"]
    if command:
        cmd.append(command)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create tmux session: {result.stderr.strip()}")

    pipe_result = subprocess.run(
        ["tmux", "pipe-pane", "-t", prefixed, "-o", f"cat >> {shlex.quote(str(log_file))}"],
        capture_output=True,
        text=True,
    )
    if pipe_result.returncode != 0:
        subprocess.run(["tmux", "kill-session", "-t", prefixed], capture_output=True)
        raise RuntimeError(f"Failed to set up logging: {pipe_result.stderr.strip()}")

    metadata = _load_metadata()
    metadata[name] = {
        "prefixed_name": prefixed,
        "log_file": str(log_file),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
    }
    _save_metadata(metadata)

    return {
        "name": name,
        "tmux_session": prefixed,
        "log_file": str(log_file),
        "status": "created",
    }


@tool
def tmux_read(name: str, lines: int = 50, source: str = "pane") -> str:
    """Read output from a tmux session.

    Args:
        name: Session name (as given to tmux_create)
        lines: Number of lines to capture (default: 50, max: 5000)
        source: "pane" for current visible content, "log" for full pipe-pane log history

    Returns:
        Session output with ANSI escape codes stripped
    """
    _validate_name(name)
    prefixed = _prefixed(name)
    lines = max(1, min(lines, 5000))

    if source == "pane":
        if not _session_exists(prefixed):
            raise RuntimeError(f"Session '{name}' not found. Use tmux_list() to see active sessions.")

        result = subprocess.run(
            ["tmux", "capture-pane", "-t", prefixed, "-p", "-S", f"-{lines}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to capture pane: {result.stderr.strip()}")
        return _strip_ansi(result.stdout)

    elif source == "log":
        log_file = _get_log_dir() / f"{name}.log"
        if not log_file.exists():
            raise RuntimeError(f"No log file found for session '{name}'.")
        with open(log_file) as f:
            tail = deque(f, maxlen=lines)
        return _strip_ansi("".join(tail))

    else:
        raise ValueError(f"Invalid source '{source}': must be 'pane' or 'log'")


@tool
def tmux_send(name: str, keys: str, enter: bool = True) -> str:
    """Send keystrokes to a tmux session.

    Args:
        name: Session name (as given to tmux_create)
        keys: Keys to send (text command or tmux key names like "C-c", "Enter", "q")
        enter: Whether to send Enter after the keys (default: True). Set False for
               single-key interactive inputs like "q" to quit, "y" to confirm, etc.

    Returns:
        Confirmation message
    """
    _validate_name(name)
    prefixed = _prefixed(name)

    if not _session_exists(prefixed):
        raise RuntimeError(f"Session '{name}' not found. Use tmux_list() to see active sessions.")

    cmd = ["tmux", "send-keys", "-t", prefixed, keys]
    if enter:
        cmd.append("Enter")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to send keys: {result.stderr.strip()}")

    return f"Sent {'keys' if not enter else 'command'} to session '{name}'"


@tool
def tmux_list() -> list:
    """List all tsugite-managed tmux sessions with their current status.

    Returns:
        List of dicts with name, status, created_at, command, and log_file for each session
    """
    return _list_managed_sessions()


@tool
def tmux_kill(name: str) -> str:
    """Terminate a tmux session and clean up its metadata.

    Args:
        name: Session name (as given to tmux_create)

    Returns:
        Confirmation message
    """
    _validate_name(name)
    prefixed = _prefixed(name)

    if not _session_exists(prefixed):
        raise RuntimeError(f"Session '{name}' not found. Use tmux_list() to see active sessions.")

    result = subprocess.run(
        ["tmux", "kill-session", "-t", prefixed],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to kill session: {result.stderr.strip()}")

    metadata = _load_metadata()
    metadata.pop(name, None)
    _save_metadata(metadata)

    return f"Session '{name}' terminated. Log file preserved at {_get_log_dir() / f'{name}.log'}"


def get_tmux_sessions() -> list:
    """Get active tsugite-managed tmux sessions for use in Jinja2 templates.

    Returns a list of dicts with name, status, and command for each session.
    Returns empty list if tmux is not installed or no managed sessions exist.
    """
    if not shutil.which("tmux"):
        return []
    return [
        {"name": s["name"], "status": s["status"], "command": s["command"]}
        for s in _list_managed_sessions()
    ]
