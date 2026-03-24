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
        - metadata: Conversation metadata (id, agent, model, machine, created_at, status)
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
        "status": storage.status,
    }

    turns = []
    for record in records:
        if isinstance(record, Turn):
            turns.append(
                {
                    "timestamp": record.timestamp.isoformat(),
                    "user_summary": record.user_summary,
                    "final_answer": record.final_answer,
                    "functions_called": record.functions_called or [],
                    "tokens": record.tokens,
                    "cost": record.cost,
                    "duration_ms": record.duration_ms,
                    "messages": record.messages,
                }
            )

    summary = {
        "turn_count": storage.turn_count,
        "total_tokens": storage.total_tokens if storage.total_tokens > 0 else None,
        "total_cost": round(storage.total_cost, 4) if storage.total_cost > 0 else None,
        "total_duration_ms": storage.total_duration_ms if storage.total_duration_ms > 0 else None,
        "functions_used": storage.all_functions_called,
    }

    return {"metadata": metadata, "turns": turns, "summary": summary}


@tool
def list_conversations(
    limit: int = 10,
    agent: Optional[str] = None,
    machine: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List recent conversations with optional filtering.

    Args:
        limit: Maximum number of conversations to return (default: 10, max: 100)
        agent: Filter by agent name (optional)
        machine: Filter by machine name (optional)
        status: Filter by status: success, error, interrupted (optional)

    Returns:
        List of conversation summaries, sorted by most recent first.
        Each entry contains: conversation_id, agent, model, machine, created_at,
        turn_count, total_tokens, total_cost, status, duration_ms, functions_used
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

            if agent and storage.agent != agent:
                continue
            if machine and storage.machine != machine:
                continue
            if status and storage.status != status:
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
                    "status": storage.status,
                    "duration_ms": storage.total_duration_ms if storage.total_duration_ms > 0 else None,
                    "functions_used": storage.all_functions_called,
                }
            )

        except Exception:
            continue

    return conversations


@tool
def search_conversations(
    query: str,
    limit: int = 10,
    agent: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search conversation history by text content.

    Searches across user prompts, agent outputs, and tool/function names.

    Args:
        query: Text to search for (case-insensitive)
        limit: Maximum number of results (default: 10, max: 50)
        agent: Filter by agent name (optional)

    Returns:
        List of matching conversations with match context snippet.
    """
    if limit < 1 or limit > 50:
        raise ValueError("Limit must be between 1 and 50")

    from tsugite.cli.history import _search_turns

    session_files = list_session_files()
    query_lower = query.lower()
    results = []

    for session_file in session_files:
        if len(results) >= limit:
            break

        try:
            storage = SessionStorage.load(session_file)

            if agent and storage.agent != agent:
                continue

            match_snippet = _search_turns(storage.load_records(), query_lower)
            if not match_snippet:
                continue

            results.append(
                {
                    "conversation_id": storage.session_id,
                    "agent": storage.agent,
                    "match": match_snippet,
                    "status": storage.status,
                    "created_at": storage.created_at.isoformat() if storage.created_at else None,
                    "total_tokens": storage.total_tokens,
                    "total_cost": round(storage.total_cost, 4) if storage.total_cost else None,
                }
            )

        except Exception:
            continue

    return results
