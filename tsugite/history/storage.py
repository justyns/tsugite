"""JSONL-based conversation history storage."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tsugite.xdg import get_xdg_data_path


def get_history_dir() -> Path:
    """Get path to conversation history directory.

    Returns:
        Path to history directory in XDG data location
        (~/.local/share/tsugite/history/)
    """
    return get_xdg_data_path("history")


def generate_conversation_id(agent_name: str, timestamp: Optional[datetime] = None) -> str:
    """Generate unique conversation ID.

    Format: YYYYMMDD_HHMMSS_{agent}_{hash}
    Example: 20251024_103000_chat_abc123

    Args:
        agent_name: Name of the agent
        timestamp: Optional timestamp (defaults to now)

    Returns:
        Unique conversation ID
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    # Format: YYYYMMDD_HHMMSS
    date_str = timestamp.strftime("%Y%m%d_%H%M%S")

    # Clean agent name (remove special chars)
    clean_agent = "".join(c if c.isalnum() or c == "-" else "_" for c in agent_name)
    clean_agent = clean_agent[:20]  # Limit length

    # Generate short hash from timestamp + agent name
    hash_input = f"{timestamp.isoformat()}_{agent_name}".encode()
    hash_str = hashlib.sha256(hash_input).hexdigest()[:6]

    return f"{date_str}_{clean_agent}_{hash_str}"


def _get_conversation_path(conversation_id: str) -> Path:
    """Get path to conversation JSONL file.

    Args:
        conversation_id: Conversation ID

    Returns:
        Path to JSONL file
    """
    history_dir = get_history_dir()
    return history_dir / f"{conversation_id}.jsonl"


def save_turn_to_history(conversation_id: str, turn_data: Dict[str, Any]) -> None:
    """Append a turn to conversation history.

    Creates the history directory and file if they don't exist.
    Appends turn as JSONL line.

    Args:
        conversation_id: Conversation ID
        turn_data: Turn data to append (will be JSON serialized)

    Raises:
        RuntimeError: If write fails
    """
    conversation_path = _get_conversation_path(conversation_id)

    # Ensure directory exists
    conversation_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(conversation_path, "a", encoding="utf-8") as f:
            json.dump(turn_data, f, ensure_ascii=False)
            f.write("\n")
    except IOError as e:
        raise RuntimeError(f"Failed to save turn to {conversation_path}: {e}")


def load_conversation(conversation_id: str) -> List[Dict[str, Any]]:
    """Load full conversation from JSONL file.

    Args:
        conversation_id: Conversation ID

    Returns:
        List of turn data dictionaries (in order)

    Raises:
        FileNotFoundError: If conversation doesn't exist
        RuntimeError: If read fails
    """
    conversation_path = _get_conversation_path(conversation_id)

    if not conversation_path.exists():
        raise FileNotFoundError(f"Conversation not found: {conversation_id}")

    turns = []
    try:
        with open(conversation_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                try:
                    turn = json.loads(line)
                    turns.append(turn)
                except json.JSONDecodeError as e:
                    # Skip malformed lines but log warning
                    print(f"Warning: Skipping malformed line {line_num} in {conversation_id}: {e}")
                    continue

        return turns
    except IOError as e:
        raise RuntimeError(f"Failed to load conversation {conversation_id}: {e}")


def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation file.

    Args:
        conversation_id: Conversation ID to delete

    Returns:
        True if conversation was deleted, False if it didn't exist

    Raises:
        RuntimeError: If deletion fails
    """
    conversation_path = _get_conversation_path(conversation_id)

    if not conversation_path.exists():
        return False

    try:
        conversation_path.unlink()
        return True
    except IOError as e:
        raise RuntimeError(f"Failed to delete conversation {conversation_id}: {e}")


def list_conversation_files() -> List[Path]:
    """List all conversation JSONL files.

    Returns:
        List of conversation file paths (sorted by modification time, newest first)
    """
    history_dir = get_history_dir()

    if not history_dir.exists():
        return []

    try:
        files = list(history_dir.glob("*.jsonl"))
        # Sort by modification time (newest first)
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files
    except OSError:
        return []


def prune_conversations(
    keep_count: Optional[int] = None,
    older_than_days: Optional[int] = None,
) -> int:
    """Delete old conversations based on retention policy.

    Args:
        keep_count: Keep only this many most recent conversations
        older_than_days: Delete conversations older than this many days

    Returns:
        Number of conversations deleted

    Raises:
        ValueError: If no pruning criteria specified
    """
    if keep_count is None and older_than_days is None:
        raise ValueError("Must specify either keep_count or older_than_days")

    deleted_count = 0
    files = list_conversation_files()

    # Prune by count
    if keep_count is not None and len(files) > keep_count:
        for file_path in files[keep_count:]:
            try:
                file_path.unlink()
                deleted_count += 1
            except OSError:
                pass

    # Prune by age
    if older_than_days is not None:
        cutoff_time = datetime.now(timezone.utc).timestamp() - (older_than_days * 24 * 3600)

        for file_path in files:
            try:
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
            except OSError:
                pass

    return deleted_count


def list_conversations(
    machine: Optional[str] = None,
    agent: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """List conversations with optional filtering.

    This is a simple implementation that reads metadata from each file.
    For better performance with many conversations, use the index module.

    Args:
        machine: Filter by machine name
        agent: Filter by agent name
        limit: Maximum number of conversations to return

    Returns:
        List of conversation metadata dictionaries
    """
    from .index import query_index

    return query_index(machine=machine, agent=agent, limit=limit)
