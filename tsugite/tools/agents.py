"""Agent orchestration tools for spawning and managing sub-agents."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..tools import tool
from ..utils import parse_yaml_frontmatter, validation_error


@tool
def spawn_agent(
    agent_path: str, prompt: str, context: Optional[Dict[str, Any]] = None, model_override: Optional[str] = None
) -> str:
    """Spawn a sub-agent and return its result.

    Args:
        agent_path: Path to the agent markdown file (relative to current working directory)
        prompt: Task/prompt to give the sub-agent
        context: Optional context variables to pass to the sub-agent
        model_override: Optional model to override the agent's default model

    Returns:
        The sub-agent's execution result as a string

    Raises:
        ValueError: If agent file doesn't exist or is invalid
        RuntimeError: If sub-agent execution fails
    """
    # Convert to absolute path and validate
    agent_file = Path(agent_path)
    if not agent_file.is_absolute():
        agent_file = Path.cwd() / agent_file

    if not agent_file.exists():
        raise validation_error("agent file", str(agent_path), "not found")

    if not agent_file.suffix == ".md":
        raise validation_error("agent file", str(agent_path), "must be a .md file")

    # Prepare context for sub-agent
    sub_context = context or {}

    try:
        # Import here to avoid circular imports
        from ..agent_runner import _get_current_agent, run_agent

        # Inject subagent context
        sub_context["is_subagent"] = True
        parent = _get_current_agent()
        if parent:
            sub_context["parent_agent"] = parent

        # Run the sub-agent
        result = run_agent(
            agent_path=agent_file, prompt=prompt, context=sub_context, model_override=model_override, debug=False
        )
        return result
    except Exception as e:
        raise RuntimeError(f"Sub-agent execution failed: {e}")


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
    from ..agent_inheritance import get_global_agents_paths

    agents_info: List[Dict[str, str]] = []
    seen_names = set()

    # Define search paths in priority order
    search_paths = [
        Path.cwd() / ".tsugite" / "agents",
        Path.cwd() / "agents",
    ]

    # Add global paths
    search_paths.extend(get_global_agents_paths())

    # Scan each directory for agent files
    for search_dir in search_paths:
        if not search_dir.exists() or not search_dir.is_dir():
            continue

        for agent_file in search_dir.glob("*.md"):
            # Skip builtin agents to avoid confusion
            if agent_file.stem.startswith("builtin-"):
                continue

            # Skip if we've already seen this agent name (higher priority paths win)
            if agent_file.stem in seen_names:
                continue

            try:
                content = agent_file.read_text(encoding="utf-8")
                frontmatter, _ = parse_yaml_frontmatter(content, str(agent_file))

                name = frontmatter.get("name", agent_file.stem)
                description = frontmatter.get("description", "No description")

                # Store relative path from cwd if possible, otherwise absolute
                try:
                    display_path = str(agent_file.relative_to(Path.cwd()))
                except ValueError:
                    display_path = str(agent_file)

                agents_info.append(
                    {
                        "name": name,
                        "description": description,
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
