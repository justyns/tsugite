"""History tools for agents to access conversation data."""

from typing import Any, Dict, List, Optional

from ..history import load_conversation
from ..history.index import query_index
from . import tool


@tool
def read_conversation(conversation_id: str) -> Dict[str, Any]:
    """Read a complete conversation from history.

    Args:
        conversation_id: Unique conversation identifier (e.g., "20251024_103000_chat_abc123")

    Returns:
        Dictionary containing:
        - metadata: Conversation metadata (id, agent, model, machine, created_at)
        - turns: List of conversation turns with user/assistant messages, tools used, tokens, costs
        - summary: High-level statistics

    Raises:
        ValueError: If conversation_id is not found
    """
    try:
        records = load_conversation(conversation_id)
    except FileNotFoundError as e:
        raise ValueError(f"Conversation '{conversation_id}' not found: {e}")

    if not records:
        raise ValueError(f"Conversation '{conversation_id}' is empty")

    # First record is always metadata
    metadata_record = records[0]
    turn_records = records[1:]

    # Extract metadata
    metadata = {
        "conversation_id": metadata_record.id,
        "agent": metadata_record.agent,
        "model": metadata_record.model,
        "machine": metadata_record.machine,
        "created_at": metadata_record.created_at.isoformat(),
    }

    # Extract turns
    turns = []
    total_tokens = 0
    total_cost = 0.0

    for turn in turn_records:
        turn_data = {
            "timestamp": turn.timestamp.isoformat(),
            "user": turn.user,
            "assistant": turn.assistant,
            "tools": turn.tools or [],
            "tokens": turn.tokens,
            "cost": turn.cost,
        }

        # Include execution steps if available (thought/code/observation)
        if turn.steps:
            turn_data["steps"] = turn.steps

        # Include full message history if available
        if turn.messages:
            turn_data["messages"] = turn.messages

        turns.append(turn_data)

        if turn.tokens:
            total_tokens += turn.tokens
        if turn.cost:
            total_cost += turn.cost

    # Summary statistics
    summary = {
        "turn_count": len(turns),
        "total_tokens": total_tokens if total_tokens > 0 else None,
        "total_cost": round(total_cost, 4) if total_cost > 0 else None,
        "tools_used": list(set(tool for turn in turn_records for tool in (turn.tools or []))),
    }

    return {"metadata": metadata, "turns": turns, "summary": summary}


@tool
def list_conversations(
    limit: int = 10,
    agent: Optional[str] = None,
    machine: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List recent conversations with optional filtering.

    Args:
        limit: Maximum number of conversations to return (default: 10, max: 100)
        agent: Filter by agent name (optional)
        machine: Filter by machine name (optional)

    Returns:
        List of conversation summaries, sorted by most recent first.
        Each entry contains: conversation_id, agent, model, machine, created_at, updated_at,
        turn_count, total_tokens, total_cost
    """
    if limit < 1 or limit > 100:
        raise ValueError("Limit must be between 1 and 100")

    results = query_index(machine=machine, agent=agent, limit=limit)

    conversations = []
    for result in results:
        conversations.append(
            {
                "conversation_id": result["conversation_id"],
                "agent": result["agent"],
                "model": result["model"],
                "machine": result["machine"],
                "created_at": result["created_at"],
                "updated_at": result["updated_at"],
                "turn_count": result.get("turn_count", 0),
                "total_tokens": result.get("total_tokens"),
                "total_cost": result.get("total_cost"),
            }
        )

    return conversations
