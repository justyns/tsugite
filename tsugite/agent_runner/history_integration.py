"""History integration for agent runs.

This module provides the interface between agent execution and
the session storage system.
"""

import os
import sys
from datetime import datetime, timezone
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
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    cost: Optional[float] = None,
    execution_steps: Optional[list] = None,
    continue_conversation_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
    attachments: Optional[List[Attachment]] = None,
    channel_metadata: Optional[dict] = None,
    duration_ms: Optional[int] = None,
    claude_code_session_id: Optional[str] = None,
    schedule_id: Optional[str] = None,
) -> Optional[str]:
    """Save a single agent run to history.

    Args:
        agent_path: Path to agent file
        agent_name: Name of the agent
        prompt: User prompt/task
        result: Agent's final answer
        model: Model used
        token_count: Number of tokens used
        input_tokens: Input/prompt tokens used
        output_tokens: Output/completion tokens used
        cost: Cost of execution
        execution_steps: List of execution steps (from agent memory)
        continue_conversation_id: Optional session ID to continue
        system_prompt: System prompt sent to LLM (not stored in V2)
        attachments: List of Attachment objects
        channel_metadata: Optional channel routing metadata
        duration_ms: Execution duration in milliseconds
        claude_code_session_id: Claude Code session ID to store in turn metadata
        schedule_id: Schedule ID if this was a scheduled run

    Returns:
        Session ID if saved, None if history disabled or failed
    """
    if os.environ.get("TSUGITE_SUBAGENT_MODE") == "1":
        return None

    session_id = None
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
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            model=model,
            duration_ms=duration_ms,
            functions_called=functions_called,
            metadata=metadata or None,
        )

        session_id = storage.session_id

    except Exception as e:
        print(f"Warning: Failed to save run to history: {e}", file=sys.stderr)

    # Record to usage store (even if history save failed, try to record usage)
    _record_usage(
        agent_name=agent_name,
        model=model,
        session_id=session_id,
        schedule_id=schedule_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=token_count,
        cost=cost,
        duration_ms=duration_ms,
    )

    return session_id


def _record_usage(
    agent_name: str,
    model: str,
    session_id: Optional[str],
    schedule_id: Optional[str],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    total_tokens: Optional[int],
    cost: Optional[float],
    duration_ms: Optional[int],
) -> None:
    """Record usage to the SQLite usage store. Best-effort, never raises."""
    try:
        from tsugite.usage import UsageRecord, get_usage_store

        store = get_usage_store()
        store.record(
            UsageRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                agent=agent_name,
                model=model,
                session_id=session_id,
                schedule_id=schedule_id,
                input_tokens=input_tokens or 0,
                output_tokens=output_tokens or 0,
                total_tokens=total_tokens or 0,
                cost=cost or 0.0,
                duration_ms=duration_ms,
            )
        )
    except Exception as e:
        print(f"Warning: Failed to record usage: {e}", file=sys.stderr)


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
