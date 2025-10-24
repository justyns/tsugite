"""Docker CLI wrapper entry points.

This module provides console script entry points for tsugite-docker and
tsugite-docker-session. The implementation is kept simple to maintain
the principle of zero coupling with tsugite core.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from uuid import uuid4


def docker_main():
    """Entry point for tsugite-docker wrapper."""
    # Parse wrapper-specific flags
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--network", default="host", help="Docker network mode")
    parser.add_argument("--keep", action="store_true", help="Keep container running")
    parser.add_argument("--container", help="Use existing container or create named container")
    parser.add_argument("--image", default="tsugite/runtime", help="Docker image to use")

    args, tsugite_args = parser.parse_known_args()

    # If using existing container, exec into it
    if args.container:
        # Check if container exists
        check = subprocess.run(
            ["docker", "ps", "-a", "-q", "-f", f"name=^{args.container}$"],
            capture_output=True,
            text=True,
            check=False,
        )

        if check.stdout.strip():
            # Container exists - exec into it (tsugite command available in PATH)
            cmd = ["docker", "exec", "-it", "-w", "/workspace", args.container, "tsugite"] + tsugite_args
        else:
            # Container doesn't exist - create it with this name
            cmd = _build_run_command(args, tsugite_args, container_name=args.container)
    else:
        # New container
        container_name = f"tsugite-{uuid4().hex[:8]}" if args.keep else None
        cmd = _build_run_command(args, tsugite_args, container_name)

    # Execute
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


def _build_run_command(args, tsugite_args, container_name=None):
    """Build docker run command."""
    import os

    cmd = ["docker", "run", "-it"]

    # Container lifecycle
    if container_name:
        cmd.extend(["--name", container_name])
    else:
        cmd.append("--rm")  # Auto-remove

    # Network
    cmd.extend(["--network", args.network])

    # Volume mounts
    # 1. Workspace (read-only for security)
    cmd.extend(["-v", f"{Path.cwd()}:/workspace:ro"])

    # 2. Config directory (read-only) - for MCP configs, model aliases, etc
    config_dir = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / "tsugite"
    if config_dir.exists():
        cmd.extend(["-v", f"{config_dir}:/root/.config/tsugite:ro"])

    # 3. Cache directory (read-write) - for attachment cache
    cache_dir = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "tsugite"
    if cache_dir.exists():
        cmd.extend(["-v", f"{cache_dir}:/root/.cache/tsugite"])

    # 4. Forward API keys and important env vars
    env_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GITHUB_TOKEN"]
    for var in env_vars:
        if var in os.environ:
            cmd.extend(["-e", var])

    # Working directory
    cmd.extend(["-w", "/workspace"])

    # Image and command (ENTRYPOINT already has "tsugite")
    cmd.append(args.image)
    cmd.extend(tsugite_args)

    return cmd


def session_main():
    """Entry point for tsugite-docker-session wrapper."""
    if len(sys.argv) < 2:
        _print_session_usage()
        sys.exit(1)

    command = sys.argv[1]

    if command == "start":
        _start_session(sys.argv[2:])
    elif command == "stop":
        _stop_session(sys.argv[2:])
    elif command == "list":
        _list_sessions()
    elif command == "exec":
        _exec_session(sys.argv[2:])
    else:
        print(f"Unknown command: {command}")
        _print_session_usage()
        sys.exit(1)


def _start_session(args):
    """Start a persistent container."""
    import os

    if not args:
        print("Error: session name required")
        print("Usage: tsugite-docker-session start NAME [--network NETWORK] [--image IMAGE]")
        sys.exit(1)

    name = args[0]
    network = "host"  # Default to host for consistency
    image = "tsugite/runtime"

    # Parse optional flags
    i = 1
    while i < len(args):
        if args[i] == "--network" and i + 1 < len(args):
            network = args[i + 1]
            i += 2
        elif args[i] == "--image" and i + 1 < len(args):
            image = args[i + 1]
            i += 2
        else:
            i += 1

    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--network",
        network,
    ]

    # Volume mounts (same as run command)
    cmd.extend(["-v", f"{Path.cwd()}:/workspace"])

    # Config directory
    config_dir = Path(os.getenv("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "tsugite"
    if config_dir.exists():
        cmd.extend(["-v", f"{config_dir}:/root/.config/tsugite:ro"])

    # Cache directory
    cache_dir = Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "tsugite"
    if cache_dir.exists():
        cmd.extend(["-v", f"{cache_dir}:/root/.cache/tsugite"])

    # Forward API keys
    env_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GITHUB_TOKEN"]
    for var in env_vars:
        if var in os.environ:
            cmd.extend(["-e", var])

    cmd.extend(["-w", "/workspace", image, "tail", "-f", "/dev/null"])

    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        print(f"✓ Session '{name}' started")
        print(f'  Use: tsugite-docker --container {name} run agent.md "task"')
    sys.exit(result.returncode)


def _stop_session(args):
    """Stop a container."""
    if not args:
        print("Error: session name required")
        print("Usage: tsugite-docker-session stop NAME [--remove]")
        sys.exit(1)

    name = args[0]
    remove = "--remove" in args

    # Stop container
    result = subprocess.run(["docker", "stop", name], check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)

    # Remove if requested
    if remove:
        result = subprocess.run(["docker", "rm", name], check=False)
        if result.returncode == 0:
            print(f"✓ Session '{name}' stopped and removed")
    else:
        print(f"✓ Session '{name}' stopped")

    sys.exit(result.returncode)


def _list_sessions():
    """List all tsugite containers."""
    subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            "name=tsugite-",
            "--format",
            "table {{.Names}}\t{{.Status}}\t{{.CreatedAt}}",
        ],
        check=False,
    )


def _exec_session(args):
    """Execute command in container."""
    if len(args) < 2:
        print("Error: session name and command required")
        print("Usage: tsugite-docker-session exec NAME COMMAND [ARGS...]")
        sys.exit(1)

    name = args[0]
    command = args[1:]

    # Check if running, start if needed
    check = subprocess.run(["docker", "ps", "-q", "-f", f"name=^{name}$"], capture_output=True, text=True, check=False)

    if not check.stdout.strip():
        print(f"Starting stopped session '{name}'...")
        subprocess.run(["docker", "start", name], check=False)

    # Execute
    result = subprocess.run(["docker", "exec", "-it", name] + command, check=False)
    sys.exit(result.returncode)


def _print_session_usage():
    """Print session management usage."""
    print(
        """Usage: tsugite-docker-session COMMAND [ARGS...]

Commands:
    start NAME [--network NETWORK] [--image IMAGE]
        Start a new persistent session

    stop NAME [--remove]
        Stop a session (optionally remove it)

    list
        List all tsugite sessions

    exec NAME COMMAND [ARGS...]
        Execute command in session

Examples:
    tsugite-docker-session start my-work
    tsugite-docker --container my-work run agent.md "task"
    tsugite-docker-session exec my-work bash
    tsugite-docker-session stop my-work
"""
    )
