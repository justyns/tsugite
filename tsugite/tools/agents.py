"""Agent orchestration tools for spawning and managing sub-agents."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..cli.helpers import get_workspace_dir, resolve_workspace_path
from ..tools import tool
from ..utils import parse_yaml_frontmatter


def _effective_cwd() -> Path:
    return get_workspace_dir() or Path.cwd()


def _build_subagent_cmd(agent_file: Path, model_override: Optional[str], sandbox_ctx: Optional[Any]) -> List[str]:
    """Build the `tsu run --subagent-mode` command, propagating the sandbox policy.

    When the spawning agent runs sandboxed (sandbox_ctx is set), the child gets
    `--sandbox` plus the same network policy so it re-enters the sandbox branch
    with its own bwrap - inheritance, so spawn_agent can't be used to escape.
    """
    cmd = ["uv", "run", "tsu", "run", str(agent_file), "--subagent-mode"]
    if model_override:
        cmd.extend(["--model", model_override])
    if sandbox_ctx is not None:
        cmd.append("--sandbox")
        if sandbox_ctx.no_network:
            cmd.append("--no-network")
        for domain in sandbox_ctx.allow_domains:
            cmd.extend(["--allow-domain", domain])
    return cmd


def resolve_agent_path(agent_path: str) -> Optional[Path]:
    """Resolve an agent reference (path or name) to a file. Returns None if missing.

    Used by `spawn_agent` at runtime and by multi-step pre-flight validation, so
    both paths agree on what is resolvable.
    """
    from ..agent_inheritance import find_agent_file

    candidate = resolve_workspace_path(agent_path)
    if candidate.exists():
        return candidate
    return find_agent_file(agent_path, current_agent_dir=_effective_cwd())


@tool(parent_only=True)
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

    from ..agent_runner import get_allowed_agents, get_current_agent, get_sandbox_context

    agent_file = resolve_agent_path(agent_path)
    if agent_file is None:
        raise ValueError(f"Agent not found: {agent_path}")

    # Parse agent config to check visibility and spawnable
    content = agent_file.read_text()
    frontmatter, _ = parse_yaml_frontmatter(content, str(agent_file))
    agent_name = frontmatter.get("name", agent_file.stem)
    visibility = frontmatter.get("visibility", "public")
    spawnable = frontmatter.get("spawnable", True)

    # Check spawnable flag (hard block - cannot be overridden)
    if not spawnable:
        raise ValueError(
            f"Agent '{agent_name}' is marked as non-spawnable (spawnable: false). "
            f"This agent cannot be spawned by other agents."
        )

    # Check allowed agents list and visibility
    allowed_agents = get_allowed_agents()
    is_explicitly_allowed = allowed_agents is not None and agent_name in allowed_agents

    # If allowed list exists and agent is not in it
    if allowed_agents is not None and not is_explicitly_allowed:
        raise ValueError(
            f"Agent '{agent_name}' is not in the allowed agents list. "
            f"Allowed: {', '.join(allowed_agents)}. "
            f"To spawn this agent, add it to the run command: "
            f'tsugite run +{get_current_agent() or "primary"} +{agent_name} "task"'
        )

    # Check visibility (only if not explicitly allowed via multi-agent mode)
    if not is_explicitly_allowed and visibility in ["private", "internal"]:
        # No allowed list means unrestricted, but respect visibility
        if allowed_agents is None:
            raise ValueError(
                f"Agent '{agent_name}' has visibility '{visibility}' and cannot be spawned. "
                f"Only 'public' agents can be spawned without explicit permission. "
                f"To spawn this agent, use multi-agent mode: "
                f'tsugite run +{get_current_agent() or "primary"} +{agent_name} "task"'
            )

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

    # Build command. Inherit the sandbox: if this (parent) agent runs sandboxed,
    # the subagent must too, otherwise spawn_agent is a trivial escape.
    cmd = _build_subagent_cmd(agent_file, model_override, get_sandbox_context())

    # Set up progress spinner
    import queue
    import threading

    from ..ui_context import get_progress, get_ui_handler

    progress = get_progress()
    ui_handler = get_ui_handler()
    agent_name = agent_file.stem

    # Show initial message through event system
    if ui_handler and not progress:
        from ..events.helpers import emit_info_event

        emit_info_event(f"🚀 Spawning subagent: [cyan]{agent_name}[/cyan]...")

    try:
        # Spawn subprocess with line buffering
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered for real-time output
            cwd=str(_effective_cwd()),
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
    from ..events.helpers import emit_info_event

    emit_info_event(f"[Subagent] {message}")


def discover_agents() -> List[Dict[str, str]]:
    """Discover available agents across standard agent directories.

    Returns a structured list (used by `list_agents` for markdown formatting and
    by `list_available_agents` as the on-demand discovery tool). Excludes the
    currently running agent.
    """
    from ..agent_inheritance import AgentDirSource, iter_agent_search_paths
    from ..agent_runner import get_current_agent

    agents_info: List[Dict[str, str]] = []
    seen_names: set = set()
    current_agent_name = get_current_agent()

    effective_cwd = _effective_cwd()
    readonly_suffix = {
        AgentDirSource.BUILTIN: " (built-in)",
        AgentDirSource.PLUGIN: " (plugin)",
    }

    # include_local_roots=False: discovery globs *.md, and the workspace cwd of
    # a daemon agent can be a notes vault where every file has frontmatter.
    for entry in iter_agent_search_paths(current_agent_dir=effective_cwd, include_local_roots=False):
        if not entry.path.is_dir():
            continue

        for agent_file in entry.path.glob("*.md"):
            if agent_file.stem in seen_names:
                continue
            try:
                content = agent_file.read_text(encoding="utf-8")
                frontmatter, _ = parse_yaml_frontmatter(content, str(agent_file))

                name = frontmatter.get("name", agent_file.stem)
                description = frontmatter.get("description", "No description")

                if current_agent_name and name == current_agent_name:
                    continue

                suffix = readonly_suffix.get(entry.source)
                if suffix:
                    display_path = name
                    description = f"{description}{suffix}"
                else:
                    try:
                        display_path = str(agent_file.relative_to(effective_cwd))
                    except ValueError:
                        display_path = str(agent_file)

                agents_info.append({"name": name, "description": description, "path": display_path})
                seen_names.add(agent_file.stem)
            except Exception:
                continue

    return agents_info


def format_agents_markdown(agents: List[Dict[str, str]]) -> str:
    """Format the structured agent list as the markdown bullet list templates expect."""
    if not agents:
        return ""
    return "\n".join(f"- **{a['name']}** (`{a['path']}`): {a['description']}" for a in agents)


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
    return format_agents_markdown(discover_agents())


@tool
def list_available_agents() -> List[Dict[str, str]]:
    """Discover sub-agents that can be delegated to via spawn_agent().

    Call this on demand when you need to delegate; the result is intentionally
    not always present in your context. Returns one dict per agent with `name`,
    `path`, and `description` keys. The currently running agent is excluded so
    you cannot spawn yourself.
    """
    return discover_agents()
