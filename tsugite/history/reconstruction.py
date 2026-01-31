"""Message reconstruction from session storage V2.

This module reconstructs the full message history from session records,
including dereferencing cached content for multi-modal messages.
"""

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional

from tsugite.cache import get_content_by_hash, get_content_by_hash_as_base64

from .models import (
    CompactionSummary,
    ContextSnapshot,
    ContextUpdate,
    Turn,
)
from .storage import SessionStorage


def reconstruct_messages(session_path: Path) -> List[Dict[str, Any]]:
    """Reconstruct full message history from session.

    System prompt is NOT included - it should be added by the caller.

    Args:
        session_path: Path to session JSONL file

    Returns:
        List of message dicts for LLM
    """
    storage = SessionStorage.load(session_path)
    records = storage.load_records()
    messages = []

    context = None
    compaction_summary = None

    for record in records:
        if isinstance(record, ContextSnapshot):
            context = record
        elif isinstance(record, CompactionSummary):
            compaction_summary = record

    if context:
        context_msg = _build_context_xml(context)
        if context_msg:
            messages.append({"role": "user", "content": context_msg})
            messages.append({"role": "assistant", "content": "Context loaded."})

    if compaction_summary:
        messages.append(
            {
                "role": "user",
                "content": f"<previous_conversation>\n{compaction_summary.summary}\n</previous_conversation>",
            }
        )
        messages.append({"role": "assistant", "content": "I've reviewed our previous conversation."})

    for record in records:
        if isinstance(record, Turn):
            dereferenced = dereference_cached_content(record.messages)
            messages.extend(dereferenced)
        elif isinstance(record, ContextUpdate):
            update_msg = _build_context_update_xml(record)
            if update_msg:
                messages.append({"role": "user", "content": update_msg})
                messages.append({"role": "assistant", "content": "Context updated."})

    return messages


def _build_context_xml(context: ContextSnapshot) -> Optional[str]:
    """Build XML representation of initial context."""
    if not context.attachments and not context.skills:
        return None

    parts = ["<context>"]

    if context.attachments:
        parts.append("<attachments>")
        for name, ref in context.attachments.items():
            if ref.source == "url":
                parts.append(f'  <attachment name="{name}" type="{ref.type}" url="{ref.url}"/>')
            else:
                content = get_content_by_hash(ref.hash, is_binary=(ref.type != "text"))
                if ref.type == "text" and content:
                    parts.append(f'  <attachment name="{name}" type="{ref.type}">')
                    parts.append(f"    {content}")
                    parts.append("  </attachment>")
                else:
                    parts.append(f'  <attachment name="{name}" type="{ref.type}" cached="true"/>')
        parts.append("</attachments>")

    if context.skills:
        parts.append("<skills>")
        for skill in context.skills:
            parts.append(f"  <skill>{skill}</skill>")
        parts.append("</skills>")

    parts.append("</context>")
    return "\n".join(parts)


def _build_context_update_xml(update: ContextUpdate) -> Optional[str]:
    """Build XML representation of context update."""
    if not update.changed and not update.removed and not update.added_skills and not update.removed_skills:
        return None

    parts = ["<context_update>"]

    if update.changed:
        parts.append("<changed>")
        for name, ref in update.changed.items():
            if ref.source == "url":
                parts.append(f'  <attachment name="{name}" type="{ref.type}" url="{ref.url}"/>')
            else:
                content = get_content_by_hash(ref.hash, is_binary=(ref.type != "text"))
                if ref.type == "text" and content:
                    parts.append(f'  <attachment name="{name}" type="{ref.type}">')
                    parts.append(f"    {content}")
                    parts.append("  </attachment>")
                else:
                    parts.append(f'  <attachment name="{name}" type="{ref.type}" cached="true"/>')
        parts.append("</changed>")

    if update.removed:
        parts.append("<removed>")
        for name in update.removed:
            parts.append(f'  <attachment name="{name}"/>')
        parts.append("</removed>")

    if update.added_skills:
        parts.append("<added_skills>")
        for skill in update.added_skills:
            parts.append(f"  <skill>{skill}</skill>")
        parts.append("</added_skills>")

    if update.removed_skills:
        parts.append("<removed_skills>")
        for skill in update.removed_skills:
            parts.append(f"  <skill>{skill}</skill>")
        parts.append("</removed_skills>")

    parts.append("</context_update>")
    return "\n".join(parts)


def dereference_cached_content(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Replace cache_key references with actual content.

    For images, this converts cache references to base64 data URIs.
    For text, this retrieves the cached content.

    Args:
        messages: List of message dicts

    Returns:
        Messages with dereferenced content
    """
    result = []

    for msg in messages:
        content = msg.get("content")

        if isinstance(content, list):
            new_content = []
            for block in content:
                if not isinstance(block, dict):
                    new_content.append(block)
                    continue

                block_type = block.get("type")

                if block_type == "image_url":
                    source = block.get("source")
                    if source == "file" and "cache_key" in block:
                        cache_key = block["cache_key"]
                        mime_type = block.get("mime_type", "image/png")
                        image_data = get_content_by_hash_as_base64(cache_key)

                        if image_data is None and block.get("original_path"):
                            image_data = _read_and_encode_file(block["original_path"])

                        if image_data:
                            new_content.append(
                                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}
                            )
                        else:
                            new_content.append(block)
                    elif source == "url" and "url" in block:
                        new_content.append({"type": "image_url", "image_url": {"url": block["url"]}})
                    elif "image_url" in block:
                        new_content.append(block)
                    else:
                        new_content.append(block)
                else:
                    new_content.append(block)

            result.append({**msg, "content": new_content})
        else:
            result.append(msg)

    return result


def _read_and_encode_file(path: str) -> Optional[str]:
    """Read file and return base64 encoded content."""
    try:
        file_path = Path(path)
        if file_path.exists():
            content = file_path.read_bytes()
            return base64.b64encode(content).decode("ascii")
    except Exception:
        pass
    return None


def get_current_context(session_path: Path) -> Dict[str, Any]:
    """Get current context state by replaying context records.

    Args:
        session_path: Path to session JSONL file

    Returns:
        Dict with 'attachments' and 'skills' keys
    """
    storage = SessionStorage.load(session_path)
    return {
        "attachments": {name: ref.model_dump(exclude_none=True) for name, ref in storage._current_attachments.items()},
        "skills": list(storage._current_skills),
        "context_hash": storage._current_context_hash,
    }


def get_turns(session_path: Path) -> List[Turn]:
    """Get all turn records from session.

    Args:
        session_path: Path to session JSONL file

    Returns:
        List of Turn records
    """
    storage = SessionStorage.load(session_path)
    return [r for r in storage.load_records() if isinstance(r, Turn)]


def apply_cache_control_to_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply cache control markers to all conversation messages.

    Following industry best practices from Anthropic and OpenAI, we cache
    all conversation history.

    Args:
        messages: List of message dicts

    Returns:
        Messages with cache_control added
    """
    if not messages:
        return messages

    return [{**msg, "cache_control": {"type": "ephemeral"}} for msg in messages]


def load_and_apply_history(session_path: Path) -> List[Dict[str, Any]]:
    """Load session and apply cache control markers.

    Args:
        session_path: Path to session JSONL file

    Returns:
        Messages with cache control applied

    Raises:
        FileNotFoundError: If session not found
    """
    messages = reconstruct_messages(session_path)
    if messages:
        messages = apply_cache_control_to_messages(messages)
    return messages
