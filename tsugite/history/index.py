"""JSON index for fast conversation metadata lookup."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .storage import get_history_dir, list_conversation_files, load_conversation


def _get_index_path() -> Path:
    """Get path to index.json file.

    Returns:
        Path to index file
    """
    return get_history_dir() / "index.json"


def load_index() -> Dict[str, Dict[str, Any]]:
    """Load conversation index from JSON file.

    Returns:
        Dictionary mapping conversation IDs to metadata
        Empty dict if index doesn't exist
    """
    index_path = _get_index_path()

    if not index_path.exists():
        return {}

    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        # Index corrupted, return empty dict
        # Will be rebuilt on next update
        return {}


def save_index(index: Dict[str, Dict[str, Any]]) -> None:
    """Save conversation index to JSON file.

    Args:
        index: Index data to save

    Raises:
        RuntimeError: If save fails
    """
    index_path = _get_index_path()

    # Ensure directory exists
    index_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
    except IOError as e:
        raise RuntimeError(f"Failed to save index to {index_path}: {e}")


def update_index(conversation_id: str, metadata: Dict[str, Any]) -> None:
    """Update index entry for a conversation.

    Creates or updates the index entry with provided metadata.
    Preserves created_at timestamp for existing entries.

    Args:
        conversation_id: Conversation ID
        metadata: Metadata to store (agent, model, machine, etc.)

    Raises:
        RuntimeError: If save fails
    """
    index = load_index()

    # Preserve created_at if entry exists
    if conversation_id in index:
        existing_created_at = index[conversation_id].get("created_at")
        if existing_created_at:
            metadata["created_at"] = existing_created_at

    # Ensure updated_at is set
    if "updated_at" not in metadata:
        metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

    index[conversation_id] = metadata
    save_index(index)


def remove_from_index(conversation_id: str) -> bool:
    """Remove conversation from index.

    Args:
        conversation_id: Conversation ID to remove

    Returns:
        True if removed, False if not in index
    """
    index = load_index()

    if conversation_id not in index:
        return False

    del index[conversation_id]
    save_index(index)
    return True


def rebuild_index() -> int:
    """Rebuild index from all conversation files.

    Scans all JSONL files and rebuilds the index from scratch.
    Useful for recovering from index corruption or manual file changes.

    Returns:
        Number of conversations indexed

    Raises:
        RuntimeError: If rebuild fails
    """
    conversation_files = list_conversation_files()
    new_index = {}

    for file_path in conversation_files:
        conversation_id = file_path.stem  # Filename without .jsonl extension

        try:
            # Load conversation and extract metadata
            turns = load_conversation(conversation_id)

            if not turns:
                continue

            # Extract metadata from first turn (should be metadata line)
            first_turn = turns[0]
            last_turn = turns[-1]

            # Build metadata from turns
            metadata = {
                "agent": first_turn.get("agent", "unknown"),
                "model": first_turn.get("model", "unknown"),
                "machine": first_turn.get("machine", "unknown"),
                "created_at": first_turn.get("timestamp", "unknown"),
                "updated_at": last_turn.get("timestamp", "unknown"),
                "turn_count": len([t for t in turns if t.get("type") == "turn"]),
                "total_tokens": sum(t.get("tokens", 0) for t in turns),
                "total_cost": sum(t.get("cost", 0.0) for t in turns),
            }

            new_index[conversation_id] = metadata

        except Exception as e:
            # Skip files that can't be read
            print(f"Warning: Failed to index {conversation_id}: {e}")
            continue

    save_index(new_index)
    return len(new_index)


def query_index(
    machine: Optional[str] = None,
    agent: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Query conversation index with filters.

    Args:
        machine: Filter by machine name
        agent: Filter by agent name
        limit: Maximum number of results

    Returns:
        List of conversation metadata (sorted by updated_at, newest first)
    """
    index = load_index()

    # Convert to list of dicts with conversation_id included
    results = [{"conversation_id": conv_id, **metadata} for conv_id, metadata in index.items()]

    # Apply filters
    if machine:
        results = [r for r in results if r.get("machine") == machine]

    if agent:
        results = [r for r in results if r.get("agent") == agent]

    # Sort by updated_at (newest first)
    results.sort(key=lambda r: r.get("updated_at", ""), reverse=True)

    # Apply limit
    if limit:
        results = results[:limit]

    return results


def get_conversation_metadata(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Get metadata for a specific conversation from index.

    Args:
        conversation_id: Conversation ID

    Returns:
        Metadata dict if found, None otherwise
    """
    index = load_index()
    return index.get(conversation_id)
