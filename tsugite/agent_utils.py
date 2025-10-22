"""Utility functions for agent management."""

from pathlib import Path
from typing import List, Tuple


def _parse_agent_from_path(path: Path):
    """Parse an agent from a file path.

    Args:
        path: Path to agent file (can be a real path or builtin marker like <builtin-default>)

    Returns:
        Parsed Agent object
    """
    from tsugite.md_agents import parse_agent

    # Handle built-in agents
    path_str = str(path)
    if path_str.startswith("<builtin-"):
        from tsugite.builtin_agents import get_builtin_chat_assistant, get_builtin_default_agent

        if "builtin-default" in path_str:
            return get_builtin_default_agent()
        elif "builtin-chat-assistant" in path_str:
            return get_builtin_chat_assistant()
        else:
            raise ValueError(f"Unknown built-in agent: {path_str}")

    content = path.read_text(encoding="utf-8")
    return parse_agent(content, path)


def _is_valid_agent_file(path: Path) -> bool:
    """Check if a file is a valid agent file.

    Args:
        path: Path to check

    Returns:
        True if file is a valid agent with a name field
    """
    try:
        agent = _parse_agent_from_path(path)
        return bool(agent.config.name)
    except Exception:
        return False


def build_inheritance_chain(agent_path: Path) -> List[Tuple[str, Path]]:
    """Build the inheritance chain for an agent.

    Args:
        agent_path: Path to the agent file

    Returns:
        List of (agent_name, agent_path) tuples in inheritance order (parent to child)
    """
    from tsugite.agent_inheritance import _get_default_base_agent_name, find_agent_file

    chain = []
    visited = set()

    current_path = agent_path.resolve()
    visited.add(current_path)

    current_agent = _parse_agent_from_path(current_path)

    if current_agent.config.extends and current_agent.config.extends != "none":
        extends_chain = _get_parent_chain(current_agent.config.extends, current_path, visited.copy())
        chain.extend(extends_chain)
    elif current_agent.config.extends != "none":
        # Auto-inherit from default base agent
        # Priority: 1) Local .tsugite/default.md, 2) Config's default_base_agent
        user_default_path = current_path.parent / ".tsugite" / "default.md"
        default_path = None
        default_name = None

        if user_default_path.exists() and user_default_path.resolve() != current_path:
            # Use local default.md if it exists
            default_path = user_default_path
            default_name = "default"
        else:
            # Fall back to config's default_base_agent
            default_base_name = _get_default_base_agent_name()
            if default_base_name:
                default_path = find_agent_file(default_base_name, current_path.parent)
                default_name = default_base_name

        if default_path and default_path.resolve() != current_path:
            chain.append((default_name, default_path))
            visited.add(default_path.resolve())

            default_agent = _parse_agent_from_path(default_path)
            if default_agent.config.extends and default_agent.config.extends != "none":
                parent_chain = _get_parent_chain(default_agent.config.extends, default_path, visited.copy())
                # Insert at beginning (parents come before children)
                chain = parent_chain + chain

    chain.append((current_agent.config.name, current_path))

    return chain


def _get_parent_chain(extends_ref: str, current_path: Path, visited: set) -> List[Tuple[str, Path]]:
    """Recursively get parent chain.

    Args:
        extends_ref: Reference to parent agent
        current_path: Current agent path
        visited: Set of already visited paths (for cycle detection)

    Returns:
        List of (agent_name, agent_path) tuples
    """
    from tsugite.agent_inheritance import find_agent_file

    chain = []

    parent_path = find_agent_file(extends_ref, current_path.parent)
    if not parent_path:
        return chain

    parent_resolved = parent_path.resolve()
    if parent_resolved in visited:
        return chain

    visited.add(parent_resolved)

    parent_agent = _parse_agent_from_path(parent_path)

    if parent_agent.config.extends and parent_agent.config.extends != "none":
        grandparent_chain = _get_parent_chain(parent_agent.config.extends, parent_path, visited.copy())
        chain.extend(grandparent_chain)

    chain.append((parent_agent.config.name, parent_path))

    return chain


def list_local_agents(base_path: Path = None) -> dict[str, List[Path]]:
    """List agents in local directories.

    Args:
        base_path: Base directory to search from (defaults to cwd)

    Returns:
        Dictionary mapping location names to list of agent paths
    """
    if base_path is None:
        base_path = Path.cwd()

    results = {}

    # Add built-in agents first
    results["Built-in"] = [Path("<builtin-default>")]

    locations = [
        ("Current directory", base_path),
        (".tsugite/", base_path / ".tsugite"),
        ("agents/", base_path / "agents"),
    ]

    for location_name, location_path in locations:
        if location_path.exists() and location_path.is_dir():
            all_md_files = sorted(location_path.glob("*.md"))
            agent_files = [f for f in all_md_files if _is_valid_agent_file(f)]

            if agent_files:
                results[location_name] = agent_files

    return results
