"""Agent orchestration tools for spawning and managing sub-agents."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..tools import tool
from ..utils import parse_yaml_frontmatter


@tool
def spawn_agent(
    agent_path: str,
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
    timeout: int = 300,
) -> str:
    """Spawn subagent as subprocess.

    Args:
        agent_path: Path to agent .md file
        prompt: Task for the subagent
        context: Optional context dict (must be JSON-serializable)
        model_override: Optional model to use
        timeout: Timeout in seconds (default: 5 minutes)

    Returns:
        Subagent's final result as string

    Raises:
        ValueError: If agent not found or context not JSON-serializable
        RuntimeError: If subagent fails, times out, or errors
    """
    import json
    import subprocess

    from ..agent_runner import get_current_agent

    # Validate agent path
    agent_file = Path(agent_path)
    if not agent_file.is_absolute():
        agent_file = Path.cwd() / agent_file
    if not agent_file.exists():
        raise ValueError(f"Agent not found: {agent_path}")

    # Prepare context
    context_data = {
        "prompt": prompt,
        "context": {
            **(context or {}),
            "parent_agent": get_current_agent(),
            "is_subagent": True,
        },
    }

    # Validate JSON serializability early
    try:
        context_json = json.dumps(context_data)
    except (TypeError, ValueError) as e:
        # Try to identify problematic value
        bad_type = "unknown"
        if hasattr(e, "__context__") and e.__context__:
            bad_type = str(type(e.__context__)).split("'")[1]
        raise ValueError(
            f"Context contains non-JSON-serializable data (type: {bad_type}). "
            "Only use dicts, lists, strings, numbers, bools, and None."
        ) from e

    # Build command
    cmd = ["uv", "run", "tsu", "run", str(agent_file), "--subagent-mode"]
    if model_override:
        cmd.extend(["--model", model_override])

    # Set up progress spinner
    import queue
    import threading

    from ..ui_context import get_progress, get_ui_handler

    progress = get_progress()
    ui_handler = get_ui_handler()
    agent_name = agent_file.stem

    # Show initial message through event system
    if ui_handler and not progress:
        from ..events import EventBus, InfoEvent

        event_bus = EventBus()
        event_bus.subscribe(ui_handler.handle_event)
        event_bus.emit(InfoEvent(message=f"🚀 Spawning subagent: [cyan]{agent_name}[/cyan]..."))

    try:
        # Spawn subprocess with line buffering
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered for real-time output
            cwd=Path.cwd(),  # Subagent inherits parent's working directory
        )

        # Write context to stdin then close it
        proc.stdin.write(context_json)
        proc.stdin.close()

        # Queue for passing lines from reader thread to main thread
        line_queue = queue.Queue()
        reader_exception = None

        def read_stdout():
            """Read stdout lines in separate thread and put in queue."""
            nonlocal reader_exception
            try:
                for line in proc.stdout:
                    line_queue.put(line)
                line_queue.put(None)  # Signal EOF
            except Exception as e:
                reader_exception = e
                line_queue.put(None)

        # Start reader thread
        reader_thread = threading.Thread(target=read_stdout, daemon=True)
        reader_thread.start()

        # Read JSONL stream and collect events
        final_result = None
        errors = []

        while True:
            # Try to get line from queue with timeout for periodic updates
            try:
                line = line_queue.get(timeout=0.5)
            except queue.Empty:
                # No data yet - just waiting for subprocess output
                # Don't update progress here to avoid too many updates
                continue

            # Check for EOF or reader thread exception
            if line is None:
                if reader_exception:
                    raise reader_exception
                break

            # Process JSONL event
            try:
                event = json.loads(line.strip())

                # Skip non-dict events (e.g., if line is just a number)
                if not isinstance(event, dict):
                    continue

                event_type = event.get("type")

                # Update progress spinner for key events only
                if ui_handler:
                    if event_type == "turn_start":
                        ui_handler.update_progress(f"🚀 {agent_name}: Turn {event['turn']}")
                    elif event_type == "tool_call":
                        ui_handler.update_progress(f"🚀 {agent_name}: {event['tool']}(...)")
                    elif event_type == "code":
                        ui_handler.update_progress(f"🚀 {agent_name}: Running code...")

                # Collect results/errors
                if event_type == "final_result":
                    final_result = event["result"]
                elif event_type == "error":
                    errors.append(event)

            except json.JSONDecodeError:
                continue  # Skip malformed lines

        # Wait for reader thread to finish
        reader_thread.join(timeout=1.0)

        # Wait for completion
        try:
            return_code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise RuntimeError(f"Subagent timed out after {timeout}s")

        # Check for failures
        if return_code != 0:
            stderr = proc.stderr.read()
            error_msg = f"Subagent failed with exit code {return_code}"
            if stderr:
                error_msg += f": {stderr}"
            if errors:
                error_msg += f"\nErrors: {[e['error'] for e in errors]}"
            raise RuntimeError(error_msg)

        if errors and final_result is None:
            # Errors occurred and no result was returned
            error_details = errors[-1]  # Use most recent error
            raise RuntimeError(
                f"Subagent error at step {error_details.get('step', 'unknown')}: {error_details['error']}"
            )

        if final_result is None:
            raise RuntimeError("Subagent did not return a result")

        return final_result

    finally:
        # Restore progress to parent agent state
        if ui_handler:
            ui_handler.update_progress("Agent running...")


def _show_progress(message: str):
    """Show subagent progress in parent UI."""
    from ..events import InfoEvent
    from ..ui_context import get_ui_handler

    ui = get_ui_handler()
    if ui:
        ui.handle_event(InfoEvent(message=f"[Subagent] {message}"))


@tool
def list_agents() -> str:
    """List all available agents for delegation.

    Scans standard agent directories and returns information about
    available specialized agents. Use this to discover which agents
    are available for delegation.

    Returns:
        Formatted list of available agents with their descriptions.
        Returns empty string if no agents are found.
    """
    from ..agent_inheritance import get_builtin_agents_path, get_global_agents_paths
    from ..agent_runner import get_current_agent

    agents_info: List[Dict[str, str]] = []
    seen_names = set()

    # Get current agent name to exclude it from the list
    current_agent_name = get_current_agent()

    # Define search paths in priority order
    search_paths = [
        Path.cwd() / ".tsugite" / "agents",
        Path.cwd() / "agents",
    ]

    # Add built-in agents directory
    builtin_path = get_builtin_agents_path()
    search_paths.append(builtin_path)

    # Add global paths
    search_paths.extend(get_global_agents_paths())

    # Scan each directory for agent files
    for search_dir in search_paths:
        if not search_dir.exists() or not search_dir.is_dir():
            continue

        is_builtin_dir = search_dir == builtin_path

        for agent_file in search_dir.glob("*.md"):
            # Skip if we've already seen this agent name (higher priority paths win)
            if agent_file.stem in seen_names:
                continue

            try:
                content = agent_file.read_text(encoding="utf-8")
                frontmatter, _ = parse_yaml_frontmatter(content, str(agent_file))

                name = frontmatter.get("name", agent_file.stem)
                description = frontmatter.get("description", "No description")

                # Skip the currently running agent to prevent self-spawning
                if current_agent_name and name == current_agent_name:
                    continue

                # Store relative path from cwd if possible, otherwise use name for built-ins
                if is_builtin_dir:
                    display_path = name
                else:
                    try:
                        display_path = str(agent_file.relative_to(Path.cwd()))
                    except ValueError:
                        display_path = str(agent_file)

                # Add marker for built-in agents
                description_with_marker = f"{description} (built-in)" if is_builtin_dir else description

                agents_info.append(
                    {
                        "name": name,
                        "description": description_with_marker,
                        "path": display_path,
                    }
                )

                seen_names.add(agent_file.stem)
            except Exception:
                # Skip files that can't be parsed
                continue

    if not agents_info:
        return ""

    # Format as a simple markdown list
    lines = []
    for agent in agents_info:
        lines.append(f"- **{agent['name']}** (`{agent['path']}`): {agent['description']}")

    return "\n".join(lines)
