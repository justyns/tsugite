"""Agent orchestration tools for spawning and managing sub-agents."""

from pathlib import Path
from typing import Any, Dict, Optional

from ..tools import tool
from ..utils import validation_error


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
        from ..agent_runner import run_agent

        # Run the sub-agent
        result = run_agent(
            agent_path=agent_file, prompt=prompt, context=sub_context, model_override=model_override, debug=False
        )
        return result
    except Exception as e:
        raise RuntimeError(f"Sub-agent execution failed: {e}")
