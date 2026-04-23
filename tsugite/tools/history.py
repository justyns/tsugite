"""History tools for agents to access conversation data.

Reads the new per-event session JSONL format and surfaces it as turn-shaped
summaries (one turn = one user_input → next user_input span).
"""

from typing import Any, Dict, List, Optional

from ..history import Event, SessionStorage, SessionSummary, get_history_dir, list_session_files
from . import tool


def _summary_dict(events: List[Event]) -> Dict[str, Any]:
    """Adapt SessionSummary into the tool's dict shape."""
    s = SessionSummary.from_events(events)
    return {
        "turn_count": s.turn_count,
        "total_tokens": s.total_tokens,
        "total_cost": s.total_cost,
        "total_duration_ms": s.total_duration_ms,
        "functions_used": sorted(s.functions_called),
        "last_response": s.last_response_text,
        "status": s.status,
        "error_message": s.error_message,
    }


def _group_by_user_input(events: List[Event]) -> List[Dict[str, Any]]:
    """Walk events; produce one dict per user_input → next user_input span."""
    turns: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for event in events:
        if event.type == "user_input":
            if current:
                turns.append(current)
            current = {
                "timestamp": event.ts.isoformat(),
                "user": event.data.get("text", ""),
                "responses": [],
                "tools_called": [],
                "tokens": 0,
                "cost": 0.0,
                "duration_ms": 0,
            }
        elif current is None:
            continue
        elif event.type == "model_response":
            current["responses"].append(event.data.get("raw_content", ""))
            usage = event.data.get("usage") or {}
            if isinstance(usage, dict):
                current["tokens"] += int(usage.get("total_tokens") or 0)
            cost = event.data.get("cost")
            if cost:
                current["cost"] += float(cost)
        elif event.type == "code_execution":
            current["duration_ms"] += int(event.data.get("duration_ms") or 0)
            current["tools_called"].extend(event.data.get("tools_called") or [])

    if current:
        turns.append(current)

    for t in turns:
        t["assistant"] = t["responses"][-1] if t["responses"] else ""
        t["tools_called"] = sorted(set(t["tools_called"]))
    return turns


@tool
def read_conversation(conversation_id: str) -> Dict[str, Any]:
    """Read a complete conversation from history.

    Args:
        conversation_id: Unique conversation identifier (e.g., "20251024_103000_chat_abc123")

    Returns:
        Dict with metadata, turns (list), and summary (totals).

    Raises:
        ValueError: If conversation_id is not found
    """
    session_path = get_history_dir() / f"{conversation_id}.jsonl"
    try:
        storage = SessionStorage.load(session_path)
    except FileNotFoundError as e:
        raise ValueError(f"Conversation '{conversation_id}' not found: {e}")

    events = storage.load_events()
    if not events:
        raise ValueError(f"Conversation '{conversation_id}' is empty")

    summary = _summary_dict(events)
    start = events[0]
    metadata = {
        "conversation_id": storage.session_id,
        "agent": start.data.get("agent"),
        "model": start.data.get("model"),
        "machine": start.data.get("machine"),
        "created_at": start.ts.isoformat(),
        "status": summary["status"],
    }

    return {
        "metadata": metadata,
        "turns": _group_by_user_input(events),
        "summary": summary,
    }


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
    """
    if limit < 1 or limit > 100:
        raise ValueError("Limit must be between 1 and 100")

    out: List[Dict[str, Any]] = []
    for path in list_session_files():
        if len(out) >= limit:
            break
        try:
            meta = SessionStorage.load_meta_fast(path)
            if not meta:
                continue
            if agent and meta.data.get("agent") != agent:
                continue
            if machine and meta.data.get("machine") != machine:
                continue

            storage = SessionStorage.load(path)
            events = storage.load_events()
            summary = _summary_dict(events)

            if status and summary["status"] != status:
                continue

            out.append(
                {
                    "conversation_id": storage.session_id,
                    "agent": meta.data.get("agent"),
                    "model": meta.data.get("model"),
                    "machine": meta.data.get("machine"),
                    "created_at": meta.ts.isoformat(),
                    "turn_count": summary["turn_count"],
                    "total_tokens": summary["total_tokens"] or None,
                    "total_cost": round(summary["total_cost"], 4) if summary["total_cost"] else None,
                    "status": summary["status"],
                    "duration_ms": summary["total_duration_ms"] or None,
                    "functions_used": summary["functions_used"],
                }
            )
        except Exception:
            continue
    return out


@tool
def search_conversations(
    query: str,
    limit: int = 10,
    agent: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search conversation history by text content.

    Searches across user inputs and assistant responses.
    """
    if limit < 1 or limit > 50:
        raise ValueError("Limit must be between 1 and 50")

    q = query.lower()
    out: List[Dict[str, Any]] = []
    for path in list_session_files():
        if len(out) >= limit:
            break
        try:
            meta = SessionStorage.load_meta_fast(path)
            if not meta:
                continue
            if agent and meta.data.get("agent") != agent:
                continue

            storage = SessionStorage.load(path)
            events = storage.load_events()
            snippet = _snippet_for_query(events, q)
            if not snippet:
                continue

            summary = _summary_dict(events)
            out.append(
                {
                    "conversation_id": storage.session_id,
                    "agent": meta.data.get("agent"),
                    "match": snippet,
                    "status": summary["status"],
                    "created_at": meta.ts.isoformat(),
                    "total_tokens": summary["total_tokens"],
                    "total_cost": round(summary["total_cost"], 4) if summary["total_cost"] else None,
                }
            )
        except Exception:
            continue
    return out


def _snippet_for_query(events: List[Event], q: str) -> Optional[str]:
    for event in events:
        if event.type == "user_input":
            text = event.data.get("text", "")
            if q in text.lower():
                return f"User: {_window(text, q)}"
        elif event.type == "model_response":
            text = event.data.get("raw_content", "")
            if q in text.lower():
                return f"Assistant: {_window(text, q)}"
    return None


def _window(text: str, q: str, radius: int = 60) -> str:
    idx = text.lower().find(q)
    if idx < 0:
        return text[:120]
    start = max(0, idx - radius)
    end = min(len(text), idx + len(q) + radius)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"
