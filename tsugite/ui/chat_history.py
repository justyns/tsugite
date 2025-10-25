"""Chat history integration helpers."""

import socket
from datetime import datetime, timezone
from typing import List, Optional, Union

from tsugite.config import load_config
from tsugite.history import (
    ConversationMetadata,
    IndexEntry,
    Turn,
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

    # Create ConversationMetadata model
    metadata = ConversationMetadata(
        id=conversation_id,
        agent=agent_name,
        model=model,
        machine=machine,
        created_at=timestamp,
        timestamp=timestamp,
    )

    save_turn_to_history(conversation_id, metadata)

    # Initialize index entry using IndexEntry model
    index_metadata = IndexEntry(
        agent=agent_name,
        model=model,
        machine=machine,
        created_at=timestamp,
        updated_at=timestamp,
        turn_count=0,
        total_tokens=0,
        total_cost=0.0,
    )
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

    # Create Turn model
    turn = Turn(
        timestamp=timestamp,
        user=user_message,
        assistant=agent_response,
        tools=tool_calls,
        tokens=token_count,
        cost=cost,
    )

    save_turn_to_history(conversation_id, turn)

    # Update index with cumulative stats
    from tsugite.history import get_conversation_metadata

    metadata = get_conversation_metadata(conversation_id)

    if metadata:
        # metadata is always IndexEntry
        updated_metadata = IndexEntry(
            agent=metadata.agent,
            model=metadata.model,
            machine=metadata.machine,
            created_at=metadata.created_at,
            updated_at=timestamp,
            turn_count=metadata.turn_count + 1,
            total_tokens=(metadata.total_tokens or 0) + (token_count or 0),
            total_cost=(metadata.total_cost or 0.0) + (cost or 0.0),
        )
        update_index(conversation_id, updated_metadata)


def format_conversation_for_display(turns: List[Union[ConversationMetadata, Turn]]) -> str:
    """Format conversation turns for display.

    Args:
        turns: List of ConversationMetadata/Turn models from load_conversation()

    Returns:
        Formatted string for display
    """
    lines = []

    for turn in turns:
        if isinstance(turn, ConversationMetadata):
            # Header from ConversationMetadata model
            lines.append("=" * 60)
            lines.append(f"Conversation: {turn.id}")
            lines.append(f"Agent: {turn.agent}")
            lines.append(f"Model: {turn.model}")
            lines.append(f"Machine: {turn.machine}")
            lines.append(f"Created: {turn.created_at}")
            lines.append("=" * 60)
            lines.append("")

        elif isinstance(turn, Turn):
            # Turn from Turn model
            lines.append(f"[{turn.timestamp}]")
            lines.append(f"User: {turn.user}")
            lines.append("")
            lines.append(f"Assistant: {turn.assistant}")

            if turn.tools:
                lines.append(f"  Tools: {', '.join(turn.tools)}")

            lines.append(f"  Tokens: {turn.tokens or 0} | Cost: ${turn.cost or 0.0:.4f}")
            lines.append("")
            lines.append("-" * 60)
            lines.append("")

    return "\n".join(lines)
