"""Built-in agent definitions.

Built-in agents are now stored as .md files in the builtin_agents/ directory
and are discovered automatically through the agent resolution system.
"""


def is_builtin_agent_path(path) -> bool:
    """Check if a path represents a builtin agent.

    Args:
        path: Path object or string to check

    Returns:
        True if the path is within the builtin_agents directory
    """
    from pathlib import Path

    builtin_dir = Path(__file__).parent / "builtin_agents"
    try:
        path_obj = Path(path).resolve()
        return builtin_dir.resolve() in path_obj.parents or path_obj.parent == builtin_dir.resolve()
    except (ValueError, OSError):
        return False
