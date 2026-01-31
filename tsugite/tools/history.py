"""History tools for agents to access conversation data."""

from typing import Any, Dict, List, Optional

from ..history import SessionStorage, Turn, get_history_dir, list_session_files
from . import tool


@tool
def read_conversation(conversation_id: str) -> Dict[str, Any]:
    """Read a complete conversation from history.

    Args:
        conversation_id: Unique conversation identifier (e.g., "20251024_103000_chat_abc123")

    Returns:
        Dictionary containing:
        - metadata: Conversation metadata (id, agent, model, machine, created_at)
        - turns: List of conversation turns with messages, functions called, tokens, costs
        - summary: High-level statistics

    Raises:
        ValueError: If conversation_id is not found
    """
    session_path = get_history_dir() / f"{conversation_id}.jsonl"

    try:
        storage = SessionStorage.load(session_path)
    except FileNotFoundError as e:
        raise ValueError(f"Conversation '{conversation_id}' not found: {e}")

    records = storage.load_records()
    if not records:
        raise ValueError(f"Conversation '{conversation_id}' is empty")

    metadata = {
        "conversation_id": storage.session_id,
        "agent": storage.agent,
        "model": storage.model,
        "machine": storage.machine,
        "created_at": storage.created_at.isoformat() if storage.created_at else None,
    }

    turns = []
    total_tokens = 0
    total_cost = 0.0
    all_functions = set()

    for record in records:
        if isinstance(record, Turn):
            turn_data = {
                "timestamp": record.timestamp.isoformat(),
                "user_summary": record.user_summary,
                "final_answer": record.final_answer,
                "functions_called": record.functions_called or [],
                "tokens": record.tokens,
                "cost": record.cost,
                "messages": record.messages,
            }

            turns.append(turn_data)

            if record.tokens:
                total_tokens += record.tokens
            if record.cost:
                total_cost += record.cost
            if record.functions_called:
                all_functions.update(record.functions_called)

    summary = {
        "turn_count": len(turns),
        "total_tokens": total_tokens if total_tokens > 0 else None,
        "total_cost": round(total_cost, 4) if total_cost > 0 else None,
        "functions_used": sorted(list(all_functions)),
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
        Each entry contains: conversation_id, agent, model, machine, created_at,
        turn_count, total_tokens, total_cost
    """
    if limit < 1 or limit > 100:
        raise ValueError("Limit must be between 1 and 100")

    session_files = list_session_files()

    conversations = []
    for session_file in session_files:
        if len(conversations) >= limit:
            break

        try:
            storage = SessionStorage.load(session_file)

            # Apply filters
            if agent and storage.agent != agent:
                continue
            if machine and storage.machine != machine:
                continue

            conversations.append(
                {
                    "conversation_id": storage.session_id,
                    "agent": storage.agent,
                    "model": storage.model,
                    "machine": storage.machine,
                    "created_at": storage.created_at.isoformat() if storage.created_at else None,
                    "turn_count": storage.turn_count,
                    "total_tokens": storage.total_tokens,
                    "total_cost": round(storage.total_cost, 4) if storage.total_cost else None,
                }
            )

        except Exception:
            continue

    return conversations
