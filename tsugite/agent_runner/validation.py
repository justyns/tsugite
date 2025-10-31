"""Agent validation and information utilities."""

from pathlib import Path
from typing import Any, Dict

from tsugite.md_agents import parse_agent_file, validate_agent_execution


def validate_agent_file(agent_path: Path) -> tuple[bool, str]:
    """Validate that an agent file can be executed.

    Args:
        agent_path: Path to agent markdown file (or builtin agent path like <builtin-default>)

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Parse agent with inheritance resolution
        agent = parse_agent_file(agent_path)

        # Use centralized validation
        return validate_agent_execution(agent)

    except Exception as e:
        return False, f"Agent file validation failed: {e}"


def get_agent_info(agent_path: Path) -> Dict[str, Any]:
    """Get information about an agent without executing it.

    Args:
        agent_path: Path to agent markdown file (or builtin agent path like <builtin-default>)

    Returns:
        Dictionary with agent information
    """
    try:
        # Parse agent with inheritance resolution
        agent = parse_agent_file(agent_path)
        agent_config = agent.config

        model_display = agent_config.model
        if not model_display:
            from tsugite.config import load_config

            config = load_config()
            if config.default_model:
                model_display = f"{config.default_model} (default)"
            else:
                model_display = "not set"

        return {
            "name": agent_config.name,
            "description": getattr(agent_config, "description", "No description"),
            "model": model_display,
            "max_turns": agent_config.max_turns,
            "tools": agent_config.tools,
            "prefetch_count": (len(agent_config.prefetch) if agent_config.prefetch else 0),
            "attachments": agent_config.attachments,
            "auto_context": getattr(agent_config, "auto_context", None),
            "permissions_profile": getattr(agent_config, "permissions_profile", None),
            "valid": validate_agent_file(agent_path)[0],
            "instructions": getattr(agent_config, "instructions", ""),
        }
    except Exception as e:
        return {
            "error": str(e),
            "valid": False,
        }
