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
    load_conversation,
    query_index,
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
    execution_steps: Optional[list] = None,
    messages: Optional[list] = None,
    metadata: Optional[dict] = None,
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
        execution_steps: Optional list of execution step objects (StepResult)
        messages: Optional full LiteLLM message history
        metadata: Optional channel routing metadata

    Raises:
        RuntimeError: If save fails
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    # Convert execution_steps to dicts if provided
    steps_dicts = None
    if execution_steps:
        steps_dicts = []
        for step in execution_steps:
            if hasattr(step, "__dict__"):
                # Convert dataclass/object to dict
                step_dict = {
                    "step_number": getattr(step, "step_number", None),
                    "thought": getattr(step, "thought", ""),
                    "code": getattr(step, "code", ""),
                    "output": getattr(step, "output", ""),
                    "error": getattr(step, "error", None),
                    "tools_called": getattr(step, "tools_called", []),
                    "xml_observation": getattr(step, "xml_observation", None),
                }
                steps_dicts.append(step_dict)
            elif isinstance(step, dict):
                steps_dicts.append(step)

    # Create Turn model
    turn = Turn(
        timestamp=timestamp,
        user=user_message,
        assistant=agent_response,
        tools=tool_calls,
        tokens=token_count,
        cost=cost,
        steps=steps_dicts,
        messages=messages,
        metadata=metadata,
    )

    save_turn_to_history(conversation_id, turn)

    # Update index with cumulative stats
    from tsugite.history import get_conversation_metadata

    index_metadata = get_conversation_metadata(conversation_id)

    if index_metadata:
        # index_metadata is always IndexEntry
        # Check if this turn marks the conversation as daemon-managed
        is_daemon = metadata and metadata.get("is_daemon_managed", False) if metadata else False
        # Once a conversation is daemon-managed, it stays that way
        daemon_managed = index_metadata.is_daemon_managed or is_daemon

        updated_metadata = IndexEntry(
            agent=index_metadata.agent,
            model=index_metadata.model,
            machine=index_metadata.machine,
            created_at=index_metadata.created_at,
            updated_at=timestamp,
            turn_count=index_metadata.turn_count + 1,
            total_tokens=(index_metadata.total_tokens or 0) + (token_count or 0),
            total_cost=(index_metadata.total_cost or 0.0) + (cost or 0.0),
            is_daemon_managed=daemon_managed,
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

            # Display execution steps if available
            if turn.steps:
                lines.append("")
                lines.append("  Execution Steps:")
                for step in turn.steps:
                    step_num = step.get("step_number", "?")
                    thought = step.get("thought", "").strip()
                    code = step.get("code", "").strip()
                    output = step.get("output", "").strip()
                    error = step.get("error")
                    tools_called = step.get("tools_called", [])

                    lines.append(f"    Step {step_num}:")
                    if thought:
                        lines.append(f"      Thought: {thought[:100]}{'...' if len(thought) > 100 else ''}")
                    if tools_called:
                        lines.append(f"      Tools: {', '.join(tools_called)}")
                    if code:
                        # Show first few lines of code
                        code_lines = code.split("\n")
                        if len(code_lines) <= 3:
                            lines.append(f"      Code: {code}")
                        else:
                            lines.append(f"      Code: {code_lines[0]}")
                            lines.append(f"            ... ({len(code_lines) - 1} more lines)")
                    if output:
                        output_preview = output[:150].replace("\n", " ")
                        lines.append(f"      Output: {output_preview}{'...' if len(output) > 150 else ''}")
                    if error:
                        lines.append(f"      Error: {error}")

            lines.append(f"  Tokens: {turn.tokens or 0} | Cost: ${turn.cost or 0.0:.4f}")
            lines.append("")
            lines.append("-" * 60)
            lines.append("")

    return "\n".join(lines)


def get_latest_conversation() -> Optional[str]:
    """Get the most recent conversation ID.

    Returns:
        Conversation ID of the most recent conversation, or None if no conversations exist

    Raises:
        RuntimeError: If query fails
    """
    try:
        results = query_index(limit=1)
        if results:
            return results[0].get("conversation_id")
        return None
    except Exception as e:
        raise RuntimeError(f"Failed to query conversation index: {e}")


def load_conversation_history(conversation_id: str) -> List[Turn]:
    """Load conversation history turns (without metadata).

    Args:
        conversation_id: Conversation ID to load

    Returns:
        List of Turn objects (excludes ConversationMetadata)

    Raises:
        FileNotFoundError: If conversation doesn't exist
        RuntimeError: If load fails
    """
    records = load_conversation(conversation_id)

    # Filter out metadata, return only Turn objects
    return [record for record in records if isinstance(record, Turn)]
