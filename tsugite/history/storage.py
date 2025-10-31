"""JSONL-based conversation history storage."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

from tsugite.xdg import get_xdg_data_path

from .models import ConversationMetadata, Turn


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


def save_turn_to_history(conversation_id: str, turn_data: Union[Turn, ConversationMetadata]) -> None:
    """Append a turn to conversation history.

    Creates the history directory and file if they don't exist.
    Appends turn as JSONL line.

    Args:
        conversation_id: Conversation ID
        turn_data: Turn or ConversationMetadata model to append

    Raises:
        RuntimeError: If write fails
    """
    conversation_path = _get_conversation_path(conversation_id)

    # Ensure directory exists
    conversation_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(conversation_path, "a", encoding="utf-8") as f:
            f.write(turn_data.model_dump_json(exclude_none=True))
            f.write("\n")
    except IOError as e:
        raise RuntimeError(f"Failed to save turn to {conversation_path}: {e}")


def load_conversation(conversation_id: str) -> List[Union[ConversationMetadata, Turn]]:
    """Load full conversation from JSONL file.

    Args:
        conversation_id: Conversation ID

    Returns:
        List of ConversationMetadata/Turn models

    Raises:
        FileNotFoundError: If conversation doesn't exist
        RuntimeError: If read fails
    """
    conversation_path = _get_conversation_path(conversation_id)

    if not conversation_path.exists():
        raise FileNotFoundError(f"Conversation not found: {conversation_id}")

    records = []
    try:
        with open(conversation_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                try:
                    data = json.loads(line)
                    record_type = data.get("type")

                    if record_type == "metadata":
                        records.append(ConversationMetadata.model_validate(data))
                    elif record_type == "turn":
                        records.append(Turn.model_validate(data))
                    else:
                        print(f"Warning: Unknown record type '{record_type}' at line {line_num}")

                except json.JSONDecodeError as e:
                    # Skip malformed lines but log warning
                    print(f"Warning: Skipping malformed JSON at line {line_num} in {conversation_id}: {e}")
                    continue

        return records
    except IOError as e:
        raise RuntimeError(f"Failed to load conversation {conversation_id}: {e}")


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
