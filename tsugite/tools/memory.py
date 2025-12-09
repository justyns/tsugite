"""Memory tools for persistent semantic search across agent sessions."""

import json
from typing import Any, Dict, List, Optional

from . import tool


def _get_manager():
    """Get memory manager, raising helpful error if not available."""
    try:
        from tsugite.memory import get_memory_manager

        return get_memory_manager()
    except ImportError as e:
        raise RuntimeError(
            "Memory system requires optional dependencies. Install with: pip install tsugite[memory]"
        ) from e


@tool
def memory_store(
    content: str,
    memory_type: str = "note",
    tags: str = "",
    agent_name: Optional[str] = None,
    metadata: Optional[str] = None,
) -> Dict[str, Any]:
    """Store information in persistent memory with semantic embedding.

    Args:
        content: The information to remember
        memory_type: Type of memory (fact, event, instruction, note)
        tags: Comma-separated tags for categorization
        agent_name: Optional namespace (None = global memory)
        metadata: JSON string with type-specific data (e.g. '{"source": "user", "event_date": "2024-12-01"}')

    Returns:
        Dict with memory id and confirmation
    """
    manager = _get_manager()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    metadata_dict = json.loads(metadata) if metadata else None
    memory_id = manager.store(content, memory_type, tag_list, agent_name, metadata_dict)
    return {"id": memory_id, "status": "stored", "content_preview": content[:100]}


@tool
def memory_search(
    query: str,
    limit: int = 5,
    agent_name: Optional[str] = None,
    tags: Optional[str] = None,
    memory_type: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search memories using semantic similarity.

    Args:
        query: Search query (natural language)
        limit: Maximum results to return (default 5)
        agent_name: Filter by agent namespace (None = search all)
        tags: Comma-separated tags to filter by
        memory_type: Filter by memory type (fact, event, instruction, note)
        since: Only memories created after this date (ISO format or relative: "7d", "1w", "30d")
        until: Only memories created before this date (ISO format or relative: "7d", "1w", "30d")

    Returns:
        List of matching memories with scores
    """
    manager = _get_manager()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    memories = manager.search(
        query, limit=limit, agent_name=agent_name, tags=tag_list, memory_type=memory_type, since=since, until=until
    )
    return [
        {
            "id": m.id,
            "content": m.content,
            "type": m.memory_type,
            "agent": m.agent_name,
            "tags": m.tags,
            "metadata": m.metadata,
            "score": round(m.score, 4) if m.score else None,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in memories
    ]


@tool
def memory_list(
    limit: int = 20,
    agent_name: Optional[str] = None,
    memory_type: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List recent memories, optionally filtered.

    Args:
        limit: Maximum results (default 20)
        agent_name: Filter by agent namespace (None = all)
        memory_type: Filter by type (fact, event, instruction, note)
        since: Only memories created after this date (ISO format or relative: "7d", "1w", "30d")
        until: Only memories created before this date (ISO format or relative: "7d", "1w", "30d")

    Returns:
        List of recent memories ordered by creation time
    """
    manager = _get_manager()
    memories = manager.list_recent(
        limit=limit, agent_name=agent_name, memory_type=memory_type, since=since, until=until
    )
    return [
        {
            "id": m.id,
            "content": m.content,
            "type": m.memory_type,
            "agent": m.agent_name,
            "tags": m.tags,
            "metadata": m.metadata,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in memories
    ]


@tool
def memory_get(memory_id: int) -> Dict[str, Any]:
    """Get a specific memory by ID.

    Args:
        memory_id: ID of the memory to retrieve

    Returns:
        Memory details or error if not found
    """
    manager = _get_manager()
    memory = manager.get(memory_id)
    if memory is None:
        return {"error": f"Memory {memory_id} not found"}
    return {
        "id": memory.id,
        "content": memory.content,
        "type": memory.memory_type,
        "agent": memory.agent_name,
        "tags": memory.tags,
        "metadata": memory.metadata,
        "created_at": memory.created_at.isoformat() if memory.created_at else None,
        "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
    }


@tool
def memory_update(memory_id: int, content: str) -> Dict[str, Any]:
    """Update an existing memory's content (re-generates embedding).

    Args:
        memory_id: ID of the memory to update
        content: New content

    Returns:
        Confirmation or error if not found
    """
    manager = _get_manager()
    success = manager.update(memory_id, content)
    if success:
        return {"status": "updated", "id": memory_id}
    return {"error": f"Memory {memory_id} not found"}


@tool
def memory_delete(memory_id: int) -> Dict[str, Any]:
    """Delete a memory by ID.

    Args:
        memory_id: ID of the memory to delete

    Returns:
        Confirmation or error if not found
    """
    manager = _get_manager()
    success = manager.delete(memory_id)
    if success:
        return {"status": "deleted", "id": memory_id}
    return {"error": f"Memory {memory_id} not found"}


@tool
def memory_count(agent_name: Optional[str] = None) -> Dict[str, int]:
    """Count total memories.

    Args:
        agent_name: Filter by agent namespace (None = all)

    Returns:
        Dict with count
    """
    manager = _get_manager()
    count = manager.count(agent_name)
    return {"count": count, "agent": agent_name}
