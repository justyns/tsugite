"""History integration for agent runs.

This module provides the interface between agent execution and
the session storage system.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from tsugite.attachments.base import Attachment
from tsugite.config import load_config
from tsugite.history import (
    SessionStorage,
    apply_cache_control_to_messages,
    get_history_dir,
    reconstruct_messages,
)


def load_conversation_messages(conversation_id: str) -> List[Dict[str, Any]]:
    """Load conversation history as message list for LLM.

    System messages are skipped as they will be reconstructed with current context.

    Args:
        conversation_id: Session/conversation ID to load

    Returns:
        List of message dicts including all tool calls and observations

    Raises:
        FileNotFoundError: If conversation doesn't exist
    """
    session_path = get_history_dir() / f"{conversation_id}.jsonl"
    messages = reconstruct_messages(session_path)

    # Filter out system messages (will be reconstructed)
    return [msg for msg in messages if msg.get("role") != "system"]


def load_and_apply_history(conversation_id: str) -> List[Dict[str, Any]]:
    """Load conversation history and apply cache control markers.

    Args:
        conversation_id: Session/conversation ID to load

    Returns:
        List of message dicts with cache control applied

    Raises:
        ValueError: If conversation not found
    """
    try:
        messages = load_conversation_messages(conversation_id)
        if messages:
            messages = apply_cache_control_to_messages(messages)
        return messages
    except FileNotFoundError:
        raise ValueError(f"Conversation not found: {conversation_id}")


def save_run_to_history(
    agent_path: Path,
    agent_name: str,
    prompt: str,
    result: str,
    model: str,
    token_count: Optional[int] = None,
    cost: Optional[float] = None,
    execution_steps: Optional[list] = None,
    continue_conversation_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
    attachments: Optional[List[Attachment]] = None,
    channel_metadata: Optional[dict] = None,
    duration_ms: Optional[int] = None,
    claude_code_session_id: Optional[str] = None,
) -> Optional[str]:
    """Save a single agent run to history.

    Args:
        agent_path: Path to agent file
        agent_name: Name of the agent
        prompt: User prompt/task
        result: Agent's final answer
        model: Model used
        token_count: Number of tokens used
        cost: Cost of execution
        execution_steps: List of execution steps (from agent memory)
        continue_conversation_id: Optional session ID to continue
        system_prompt: System prompt sent to LLM (not stored in V2)
        attachments: List of Attachment objects
        channel_metadata: Optional channel routing metadata
        duration_ms: Execution duration in milliseconds
        claude_code_session_id: Claude Code session ID to store in turn metadata

    Returns:
        Session ID if saved, None if history disabled or failed
    """
    if os.environ.get("TSUGITE_SUBAGENT_MODE") == "1":
        return None

    try:
        config = load_config()
        if not getattr(config, "history_enabled", True):
            return None

        try:
            from tsugite.md_agents import parse_agent_file

            agent = parse_agent_file(agent_path)
            if getattr(agent.config, "disable_history", False):
                return None
        except Exception as e:
            print(f"Warning: Could not check agent history settings: {e}", file=sys.stderr)

        if continue_conversation_id:
            session_path = get_history_dir() / f"{continue_conversation_id}.jsonl"
            if session_path.exists():
                storage = SessionStorage.load(session_path)
                if attachments:
                    skills = []  # TODO: track loaded skills
                    storage.check_and_record_context_change(attachments, skills)
            else:
                # First message for this conversation ID (e.g., daemon mode)
                storage = SessionStorage.create(
                    agent_name=agent_name,
                    model=model,
                    session_path=session_path,
                )
                if attachments:
                    skills = []  # TODO: track loaded skills
                    storage.record_initial_context(attachments, skills)
        else:
            storage = SessionStorage.create(agent_name=agent_name, model=model)

            if attachments:
                skills = []  # TODO: track loaded skills
                storage.record_initial_context(attachments, skills)

        messages = _build_turn_messages(prompt, result, execution_steps)

        functions_called = _extract_functions_called(execution_steps) if execution_steps else []

        # Merge claude_code_session_id into metadata if present
        metadata = dict(channel_metadata) if channel_metadata else {}
        if claude_code_session_id:
            metadata["claude_code_session_id"] = claude_code_session_id

        storage.record_turn(
            messages=messages,
            final_answer=result,
            tokens=token_count,
            cost=cost,
            model=model,
            duration_ms=duration_ms,
            functions_called=functions_called,
            metadata=metadata or None,
        )

        return storage.session_id

    except Exception as e:
        print(f"Warning: Failed to save run to history: {e}", file=sys.stderr)
        return None


def _build_turn_messages(
    prompt: str,
    result: str,
    execution_steps: Optional[list] = None,
) -> List[Dict[str, Any]]:
    """Build message list for a turn."""
    messages = []

    messages.append({"role": "user", "content": prompt})

    if execution_steps:
        for step in execution_steps:
            code = getattr(step, "code", "")
            xml_observation = getattr(step, "xml_observation", "")

            if code:
                messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
            if xml_observation:
                messages.append({"role": "user", "content": xml_observation})

    if result:
        messages.append({"role": "assistant", "content": result})

    return messages


def _extract_functions_called(execution_steps: list) -> List[str]:
    """Extract list of function/tool names called during execution."""
    functions = set()
    for step in execution_steps:
        if hasattr(step, "tools_called") and step.tools_called:
            functions.update(step.tools_called)
    return sorted(list(functions))


def get_claude_code_session_id(conversation_id: str) -> Optional[str]:
    """Get the Claude Code session ID from a conversation's history metadata.

    Args:
        conversation_id: Session/conversation ID to look up

    Returns:
        Claude Code session ID if found, None otherwise
    """
    try:
        from tsugite.history import SessionStorage, Turn

        session_path = get_history_dir() / f"{conversation_id}.jsonl"
        if not session_path.exists():
            return None

        storage = SessionStorage.load(session_path)
        records = storage.load_records()

        # Search turns in reverse for the most recent session ID
        for record in reversed(records):
            if isinstance(record, Turn) and record.metadata:
                session_id = record.metadata.get("claude_code_session_id")
                if session_id:
                    return session_id

        return None
    except Exception:
        return None


def get_latest_conversation() -> Optional[str]:
    """Get the most recent conversation/session ID.

    Returns:
        Session ID of the most recent session, or None if none exist
    """
    from tsugite.history import list_session_files

    files = list_session_files()
    if files:
        return files[0].stem
    return None
