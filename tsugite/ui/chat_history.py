"""Chat history integration helpers."""

import socket
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from tsugite.config import load_config
from tsugite.history import (
    generate_conversation_id,
    save_turn_to_history,
    update_index,
)


def get_machine_name() -> str:
    """Get machine name for conversation tracking.

    Checks config for custom machine_name, falls back to hostname.

    Returns:
        Machine name string
    """
    config = load_config()

    # Check config for override
    if hasattr(config, "machine_name") and config.machine_name:
        return config.machine_name

    # Auto-detect from hostname
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def start_conversation(
    agent_name: str,
    model: str,
    timestamp: Optional[datetime] = None,
) -> str:
    """Start a new conversation and save metadata.

    Creates conversation ID and saves initial metadata line to JSONL file.

    Args:
        agent_name: Name of the agent
        model: Model identifier (e.g., "openai:gpt-4o")
        timestamp: Optional timestamp (defaults to now)

    Returns:
        Conversation ID

    Raises:
        RuntimeError: If save fails
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    conversation_id = generate_conversation_id(agent_name, timestamp)
    machine = get_machine_name()

    # Save metadata line to JSONL
    metadata = {
        "type": "metadata",
        "id": conversation_id,
        "agent": agent_name,
        "model": model,
        "machine": machine,
        "created_at": timestamp.isoformat(),
        "timestamp": timestamp.isoformat(),
    }

    save_turn_to_history(conversation_id, metadata)

    # Initialize index entry
    index_metadata = {
        "agent": agent_name,
        "model": model,
        "machine": machine,
        "created_at": timestamp.isoformat(),
        "updated_at": timestamp.isoformat(),
        "turn_count": 0,
        "total_tokens": 0,
        "total_cost": 0.0,
    }
    update_index(conversation_id, index_metadata)

    return conversation_id


def save_chat_turn(
    conversation_id: str,
    user_message: str,
    agent_response: str,
    tool_calls: List[str],
    token_count: Optional[int] = None,
    cost: Optional[float] = None,
    timestamp: Optional[datetime] = None,
) -> None:
    """Save a chat turn to history.

    Appends turn to JSONL file and updates index.

    Args:
        conversation_id: Conversation ID
        user_message: User's message
        agent_response: Agent's response
        tool_calls: List of tool calls made
        token_count: Number of tokens used
        cost: Cost of the turn
        timestamp: Optional timestamp (defaults to now)

    Raises:
        RuntimeError: If save fails
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    # Save turn to JSONL
    turn_data = {
        "type": "turn",
        "timestamp": timestamp.isoformat(),
        "user": user_message,
        "assistant": agent_response,
        "tools": tool_calls,
        "tokens": token_count or 0,
        "cost": cost or 0.0,
    }

    save_turn_to_history(conversation_id, turn_data)

    # Update index with cumulative stats
    from tsugite.history import get_conversation_metadata

    metadata = get_conversation_metadata(conversation_id)

    if metadata:
        metadata["turn_count"] = metadata.get("turn_count", 0) + 1
        metadata["total_tokens"] = metadata.get("total_tokens", 0) + (token_count or 0)
        metadata["total_cost"] = metadata.get("total_cost", 0.0) + (cost or 0.0)
        metadata["updated_at"] = timestamp.isoformat()
        update_index(conversation_id, metadata)


def format_conversation_for_display(turns: List[Dict[str, Any]]) -> str:
    """Format conversation turns for display.

    Args:
        turns: List of turn dictionaries from load_conversation()

    Returns:
        Formatted string for display
    """
    lines = []

    for turn in turns:
        turn_type = turn.get("type")

        if turn_type == "metadata":
            # Header
            lines.append("=" * 60)
            lines.append(f"Conversation: {turn.get('id', 'unknown')}")
            lines.append(f"Agent: {turn.get('agent', 'unknown')}")
            lines.append(f"Model: {turn.get('model', 'unknown')}")
            lines.append(f"Machine: {turn.get('machine', 'unknown')}")
            lines.append(f"Created: {turn.get('created_at', 'unknown')}")
            lines.append("=" * 60)
            lines.append("")

        elif turn_type == "turn":
            # Turn
            timestamp = turn.get("timestamp", "")
            user = turn.get("user", "")
            assistant = turn.get("assistant", "")
            tools = turn.get("tools", [])
            tokens = turn.get("tokens", 0)
            cost = turn.get("cost", 0.0)

            lines.append(f"[{timestamp}]")
            lines.append(f"User: {user}")
            lines.append("")
            lines.append(f"Assistant: {assistant}")

            if tools:
                lines.append(f"  Tools: {', '.join(tools)}")

            lines.append(f"  Tokens: {tokens} | Cost: ${cost:.4f}")
            lines.append("")
            lines.append("-" * 60)
            lines.append("")

    return "\n".join(lines)
