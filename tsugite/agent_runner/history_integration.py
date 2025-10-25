"""History integration for agent runs."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def save_run_to_history(
    agent_path: Path,
    agent_name: str,
    prompt: str,
    result: str,
    model: str,
    token_count: Optional[int] = None,
    cost: Optional[float] = None,
    execution_steps: Optional[list] = None,
) -> Optional[str]:
    """Save a single agent run to history.

    Args:
        agent_path: Path to agent file
        agent_name: Name of the agent
        prompt: User prompt/task
        result: Agent's final answer
        model: Model used
        token_count: Number of tokens used
        cost: Cost of execution
        execution_steps: List of execution steps (from agent memory)

    Returns:
        Conversation ID if saved, None if history disabled or failed
    """
    try:
        from tsugite.config import load_config
        from tsugite.ui.chat_history import save_chat_turn, start_conversation

        # Check if history is enabled
        config = load_config()
        if not getattr(config, "history_enabled", True):
            return None

        # Check agent-level disable_history
        try:
            from tsugite.md_agents import parse_agent_file

            agent = parse_agent_file(agent_path)
            if getattr(agent.config, "disable_history", False):
                return None
        except Exception:
            # If we can't parse agent, assume history is enabled
            pass

        # Start conversation
        timestamp = datetime.now(timezone.utc)
        conv_id = start_conversation(
            agent_name=agent_name,
            model=model,
            timestamp=timestamp,
        )

        # Save as a single turn (run mode is single-shot)
        save_chat_turn(
            conversation_id=conv_id,
            user_message=prompt,
            agent_response=result,
            tool_calls=_extract_tool_calls(execution_steps) if execution_steps else [],
            token_count=token_count,
            cost=cost,
            timestamp=timestamp,
        )

        return conv_id

    except Exception as e:
        # Don't fail the run if history fails
        import sys

        print(f"Warning: Failed to save run to history: {e}", file=sys.stderr)
        return None


def _extract_tool_calls(execution_steps: list) -> list[str]:
    """Extract list of tool names called during execution.

    Args:
        execution_steps: List of execution step objects

    Returns:
        List of unique tool names called
    """
    tool_calls = set()
    for step in execution_steps:
        if hasattr(step, "tools_called") and step.tools_called:
            tool_calls.update(step.tools_called)
    return sorted(list(tool_calls))
