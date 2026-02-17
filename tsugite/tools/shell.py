"""Shell command execution tools for Tsugite agents."""

import subprocess

from tsugite.tools import tool
from tsugite.utils import execute_shell_command


@tool
def run(command: str, timeout: int = 30, shell: bool = True) -> str:
    """Execute a shell command and return its output.

    Args:
        command: Shell command to execute
        timeout: Maximum execution time in seconds (default: 30)
        shell: Whether to use shell execution (default: True)
    """
    return execute_shell_command(command, timeout=timeout, shell=shell)


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
