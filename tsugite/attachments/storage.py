"""Attachment management for reusable context."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

from tsugite.xdg import get_xdg_config_path


def get_attachments_path() -> Path:
    """Get path to attachments.json file.

    Returns:
        Path to attachments.json in tsugite config directory
    """
    return get_xdg_config_path("attachments.json")


def load_attachments() -> Dict[str, Dict[str, str]]:
    """Load attachments from JSON file.

    Returns:
        Dictionary of attachments, empty dict if file doesn't exist
    """
    attachments_path = get_attachments_path()

    if not attachments_path.exists():
        return {}

    try:
        with open(attachments_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("attachments", {})
    except (json.JSONDecodeError, IOError) as e:
        raise RuntimeError(f"Failed to load attachments from {attachments_path}: {e}")


def save_attachments(attachments: Dict[str, Dict[str, str]]) -> None:
    """Save attachments to JSON file.

    Args:
        attachments: Dictionary of attachment data to save

    Raises:
        RuntimeError: If save fails
    """
    attachments_path = get_attachments_path()

    # Ensure directory exists
    attachments_path.parent.mkdir(parents=True, exist_ok=True)

    data = {"attachments": attachments}

    try:
        with open(attachments_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        raise RuntimeError(f"Failed to save attachments to {attachments_path}: {e}")


def add_attachment(alias: str, source: str, content: Optional[str] = None) -> None:
    """Add or update an attachment.

    For inline text (stdin), provide both source="inline" and content.
    For file/URL references, provide only source (content will be fetched on demand).

    Args:
        alias: Unique identifier for the attachment
        source: Source reference (file path, URL, or "inline" for text)
        content: Text content (only for inline attachments)

    Raises:
        ValueError: If alias is empty or invalid parameters
        RuntimeError: If save fails
    """
    if not alias or not alias.strip():
        raise ValueError("Attachment alias cannot be empty")

    # Validate inline vs reference
    is_inline = source.lower() in ("inline", "text")
    if is_inline and not content:
        raise ValueError("Inline attachments require content")

    attachments = load_attachments()
    now = datetime.now(timezone.utc).isoformat()

    # Build attachment entry
    entry = {
        "source": source,
        "updated": now,
    }

    # Only store content for inline attachments
    if is_inline:
        entry["content"] = content

    # Add created timestamp for new attachments
    if alias not in attachments:
        entry["created"] = now
    else:
        # Preserve original created timestamp
        entry["created"] = attachments[alias].get("created", now)

    attachments[alias] = entry
    save_attachments(attachments)


def get_attachment(alias: str) -> Optional[Tuple[str, Optional[str]]]:
    """Get an attachment by alias.

    Args:
        alias: Attachment identifier

    Returns:
        Tuple of (source, content) if found, None otherwise.
        For inline attachments, content is the stored text.
        For file/URL references, content is None (fetch on demand).
    """
    attachments = load_attachments()

    if alias not in attachments:
        return None

    attachment = attachments[alias]
    source = attachment["source"]
    content = attachment.get("content")  # None for references
    return source, content


def list_attachments() -> Dict[str, Dict[str, str]]:
    """List all attachments.

    Returns:
        Dictionary of all attachment data
    """
    return load_attachments()


def remove_attachment(alias: str) -> bool:
    """Remove an attachment.

    Args:
        alias: Attachment identifier to remove

    Returns:
        True if attachment was removed, False if it didn't exist

    Raises:
        RuntimeError: If save fails
    """
    attachments = load_attachments()

    if alias not in attachments:
        return False

    del attachments[alias]
    save_attachments(attachments)
    return True


def search_attachments(query: str) -> Dict[str, Dict[str, str]]:
    """Search attachments by alias or source.

    Args:
        query: Search term (case-insensitive)

    Returns:
        Dictionary of matching attachments
    """
    attachments = load_attachments()
    query_lower = query.lower()

    return {
        alias: data
        for alias, data in attachments.items()
        if query_lower in alias.lower() or query_lower in data.get("source", "").lower()
    }
