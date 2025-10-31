"""Shell command execution tools for Tsugite agents."""

import shlex
import subprocess

from tsugite.tools import tool
from tsugite.utils import execute_shell_command

DANGEROUS_SHELL_SUBSTRINGS = (
    "rm -rf /",
    "sudo rm",
    "dd if=",
    "mkfs",
    "format",
    "> /dev/",
)


BLOCKED_SAFE_MODE_COMMANDS = {
    "rm",
    "rmdir",
    "del",
    "format",
    "fdisk",
    "mkfs",
    "sudo",
    "su",
    "chmod",
    "chown",
    "passwd",
    "wget",
    "curl",
    "nc",
    "netcat",
    "ssh",
    "scp",
    "python",
    "perl",
    "ruby",
    "node",
    "bash",
    "sh",
}


@tool
def run(command: str, timeout: int = 30, shell: bool = True) -> str:
    """Execute a shell command and return its output.

    Args:
        command: Shell command to execute
        timeout: Maximum execution time in seconds (default: 30)
        shell: Whether to use shell execution (default: True)
    """
    # Basic safety check for dangerous patterns
    for pattern in DANGEROUS_SHELL_SUBSTRINGS:
        if pattern in command.lower():
            raise ValueError(f"Dangerous command pattern detected: {pattern}")

    return execute_shell_command(command, timeout=timeout, shell=shell)


@tool
def run_safe(command: str, timeout: int = 30) -> str:
    """Execute a shell command with additional safety checks.

    This version disables shell=True for safer execution and has stricter
    command validation.

    Args:
        command: Command to execute (will be parsed safely)
        timeout: Maximum execution time in seconds (default: 30)

    Returns:
        Command output

    Raises:
        RuntimeError: If command execution fails or is deemed unsafe
    """
    try:
        cmd_parts = shlex.split(command)
        if not cmd_parts:
            raise ValueError("Empty command")

        command_name = cmd_parts[0].lower()

        if command_name in BLOCKED_SAFE_MODE_COMMANDS:
            raise ValueError(f"Command '{command_name}' is not allowed in safe mode")

        return run(command, timeout=timeout, shell=False)

    except Exception as e:
        raise RuntimeError(f"Safe command execution failed: {e}") from e


@tool
def get_system_info() -> str:
    """Get basic system information.

    Returns:
        System information including OS, hostname, and current directory
    """
    try:
        info_commands = [
            ("OS", "uname -s"),
            ("Hostname", "hostname"),
            ("Current Directory", "pwd"),
            ("Date", "date"),
            ("Uptime", "uptime"),
        ]

        results = []
        for label, cmd in info_commands:
            try:
                result = subprocess.run(
                    cmd.split(),
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                )
                results.append(f"{label}: {result.stdout.strip()}")
            except Exception:
                results.append(f"{label}: [unavailable]")

        return "\n".join(results)

    except Exception as e:
        return f"Failed to get system info: {e}"
