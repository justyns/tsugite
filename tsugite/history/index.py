"""JSON index for fast conversation metadata lookup."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import portalocker

from .models import ConversationMetadata, IndexEntry, Turn
from .storage import get_history_dir, list_conversation_files, load_conversation


def _get_index_path() -> Path:
    """Get path to index.json file.

    Returns:
        Path to index file
    """
    return get_history_dir() / "index.json"


def load_index() -> Dict[str, IndexEntry]:
    """Load conversation index from JSON file.

    Returns:
        Dictionary mapping conversation IDs to IndexEntry models
        Empty dict if index doesn't exist
    """
    index_path = _get_index_path()

    if not index_path.exists():
        return {}

    try:
        with open(index_path, "r", encoding="utf-8") as f:
            raw_index = json.load(f)

        return {conv_id: IndexEntry.model_validate(metadata) for conv_id, metadata in raw_index.items()}

    except (json.JSONDecodeError, IOError):
        # Index corrupted, return empty dict
        # Will be rebuilt on next update
        return {}


def save_index(index: Dict[str, IndexEntry]) -> None:
    """Save conversation index to JSON file.

    Args:
        index: Index data to save (IndexEntry models)

    Raises:
        RuntimeError: If save fails or lock timeout
    """
    index_path = _get_index_path()

    # Ensure directory exists
    index_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        serializable_index = {
            conv_id: entry.model_dump(mode="json", exclude_none=True) for conv_id, entry in index.items()
        }

        # Use file locking to prevent concurrent write corruption
        with open(index_path, "w", encoding="utf-8") as f:
            # Acquire exclusive lock
            portalocker.lock(f, portalocker.LOCK_EX)
            try:
                json.dump(serializable_index, f, indent=2, ensure_ascii=False)
                f.flush()  # Ensure data is written
            finally:
                portalocker.unlock(f)
    except portalocker.exceptions.LockException:
        raise RuntimeError(f"Failed to acquire lock on {index_path} (timeout after 5s)")
    except IOError as e:
        raise RuntimeError(f"Failed to save index to {index_path}: {e}")


def update_index(conversation_id: str, metadata: IndexEntry) -> None:
    """Update index entry for a conversation.

    Creates or updates the index entry with provided metadata.
    Preserves created_at timestamp for existing entries.

    Args:
        conversation_id: Conversation ID
        metadata: IndexEntry model with metadata

    Raises:
        RuntimeError: If save fails
    """
    index = load_index()

    metadata_dict = metadata.model_dump(mode="json")

    # Preserve created_at if entry exists
    if conversation_id in index:
        metadata_dict["created_at"] = index[conversation_id].created_at.isoformat()

    # Ensure updated_at is set
    if "updated_at" not in metadata_dict:
        metadata_dict["updated_at"] = datetime.now(timezone.utc).isoformat()

    index[conversation_id] = IndexEntry.model_validate(metadata_dict)
    save_index(index)


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
            records = load_conversation(conversation_id)

            if not records:
                continue

            first_record = records[0]
            last_record = records[-1]

            # Build metadata from records
            turns = [r for r in records if isinstance(r, Turn)]
            metadata_dict = {
                "agent": first_record.agent if isinstance(first_record, ConversationMetadata) else "unknown",
                "model": first_record.model if isinstance(first_record, ConversationMetadata) else "unknown",
                "machine": first_record.machine if isinstance(first_record, ConversationMetadata) else "unknown",
                "created_at": (
                    first_record.timestamp.isoformat()
                    if hasattr(first_record, "timestamp")
                    else datetime.now(timezone.utc).isoformat()
                ),
                "updated_at": (
                    last_record.timestamp.isoformat()
                    if hasattr(last_record, "timestamp")
                    else datetime.now(timezone.utc).isoformat()
                ),
                "turn_count": len(turns),
                "total_tokens": sum(r.tokens or 0 for r in turns),
                "total_cost": sum(r.cost or 0.0 for r in turns),
            }

            new_index[conversation_id] = IndexEntry.model_validate(metadata_dict)

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
) -> List[Dict[str, str]]:
    """Query conversation index with filters.

    Args:
        machine: Filter by machine name
        agent: Filter by agent name
        limit: Maximum number of results

    Returns:
        List of conversation metadata dicts (sorted by updated_at, newest first)
    """
    index = load_index()

    # Convert to list of dicts with conversation_id included
    results = []
    for conv_id, entry in index.items():
        entry_dict = entry.model_dump(mode="json")
        entry_dict["conversation_id"] = conv_id
        results.append(entry_dict)

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


def get_conversation_metadata(conversation_id: str) -> Optional[IndexEntry]:
    """Get metadata for a specific conversation from index.

    Args:
        conversation_id: Conversation ID

    Returns:
        IndexEntry model if found, None otherwise
    """
    index = load_index()
    return index.get(conversation_id)
