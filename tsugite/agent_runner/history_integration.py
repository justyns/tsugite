"""History integration for agent runs.

The agent loop now emits per-event records as it runs (model_request,
model_response, code_execution, etc.). This module handles session lifecycle
around the agent run: creating the session, recording user_input + session_end
events, exposing helpers to load past sessions for continuation.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tsugite.attachments.base import Attachment
from tsugite.config import load_config
from tsugite.history import (
    SessionStorage,
    events_to_messages,
    get_history_dir,
    last_index_of,
    list_session_files,
)


def load_conversation_messages(conversation_id: str) -> List[Dict[str, Any]]:
    """Load conversation history as a messages list for an LLM call."""
    session_path = get_history_dir() / f"{conversation_id}.jsonl"
    if not session_path.exists():
        raise FileNotFoundError(f"Conversation not found: {conversation_id}")
    storage = SessionStorage.load(session_path)
    return events_to_messages(storage.iter_events())


def load_and_apply_history(conversation_id: str) -> List[Dict[str, Any]]:
    """Load conversation history, raising ValueError if not found."""
    try:
        return load_conversation_messages(conversation_id)
    except FileNotFoundError:
        raise ValueError(f"Conversation not found: {conversation_id}")


def open_or_create_session(
    *,
    agent_path: Path,
    agent_name: str,
    model: str,
    continue_conversation_id: Optional[str] = None,
    workspace: Optional[str] = None,
) -> Optional[SessionStorage]:
    """Open an existing session or create a new one.

    Returns None if history is disabled (config flag, agent flag, or subagent mode).
    """
    if os.environ.get("TSUGITE_SUBAGENT_MODE") == "1":
        return None

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
            return SessionStorage.load(session_path)
        return SessionStorage.create(
            agent_name=agent_name,
            model=model,
            workspace=workspace,
            session_path=session_path,
        )

    return SessionStorage.create(agent_name=agent_name, model=model, workspace=workspace)


def record_user_input(
    storage: SessionStorage,
    text: str,
    attachments: Optional[List[Attachment]] = None,
    channel_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Record a user_input event at the start of a turn."""
    data: Dict[str, Any] = {"text": text}
    if attachments:
        data["attachments"] = [
            {"name": a.name, "type": a.content_type.value, "source_url": a.source_url} for a in attachments
        ]
    if channel_metadata:
        data["channel"] = channel_metadata
    storage.record("user_input", **data)


def record_session_end(
    storage: SessionStorage,
    status: str = "success",
    error_message: Optional[str] = None,
) -> None:
    """Record a session_end event with final status."""
    storage.record("session_end", status=status, error_message=error_message)


def save_run_to_history(
    *,
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
    channel_metadata: Optional[Dict[str, Any]] = None,
    duration_ms: Optional[int] = None,
    provider_state: Optional[Dict[str, Any]] = None,
    status: str = "success",
    error_message: Optional[str] = None,
    raw_events: Optional[list] = None,
) -> Optional[str]:
    """Persist a completed run as events.

    Used by call sites that don't pass a SessionStorage to the agent; this
    function opens or continues a session, replays the run as a sequence of
    events (user_input, code_execution per step, model_response with the final
    text, session_end), and returns the session_id.
    """
    storage = open_or_create_session(
        agent_path=agent_path,
        agent_name=agent_name,
        model=model,
        continue_conversation_id=continue_conversation_id,
    )
    if storage is None:
        return None

    # If the agent loop already recorded events live (storage was threaded
    # through TsugiteAgent), avoid duplicating. Detect this by looking for any
    # model_response after the most recent session_start: that's a signal the
    # agent ran to completion against this storage.
    existing = list(storage.iter_events())
    last_session_start = -1
    last_model_response = -1
    last_session_end = -1
    for i, e in enumerate(existing):
        if e.type == "session_start":
            last_session_start = i
        elif e.type == "model_response":
            last_model_response = i
        elif e.type == "session_end":
            last_session_end = i

    agent_already_recorded = last_model_response > last_session_start

    if agent_already_recorded:
        if last_session_end <= last_session_start:
            record_session_end(storage, status=status, error_message=error_message)
        return storage.session_id

    record_user_input(storage, prompt, attachments=attachments, channel_metadata=channel_metadata)

    for step in execution_steps or []:
        code = getattr(step, "code", "") or ""
        if code:
            storage.record(
                "code_execution",
                code=code,
                output=getattr(step, "output", "") or "",
                error=getattr(step, "error", None),
                tools_called=list(getattr(step, "tools_called", []) or []),
            )

    state_delta = provider_state if provider_state else None
    storage.record(
        "model_response",
        provider=model.split(":", 1)[0] if ":" in model else None,
        model=model,
        raw_content=result or "",
        usage={"total_tokens": token_count} if token_count else None,
        cost=cost,
        duration_ms=duration_ms,
        state_delta=state_delta,
    )

    record_session_end(storage, status=status, error_message=error_message)
    return storage.session_id


@dataclass
class ClaudeCodeSessionInfo:
    """Info about the Claude Code session for a conversation."""

    session_id: str
    compacted: bool = False


def get_claude_code_session_info(conversation_id: str) -> Optional[ClaudeCodeSessionInfo]:
    """Find the most recent Claude Code session_id from a conversation's events.

    Returns None if no session_id was recorded (non-claude_code model used) or
    if a compaction happened after the last session_id (the Claude Code session
    is stale at that point).
    """
    try:
        session_path = get_history_dir() / f"{conversation_id}.jsonl"
        if not session_path.exists():
            return None
        storage = SessionStorage.load(session_path)
        events = storage.load_events()

        # Find the last compaction. Anything before it is stale.
        compaction_idx = last_index_of(events, "compaction")

        for i in range(len(events) - 1, -1, -1):
            event = events[i]
            if event.type != "model_response":
                continue
            state = event.data.get("state_delta") or {}
            session_id = state.get("session_id")
            if not session_id:
                continue
            if compaction_idx is not None and i <= compaction_idx:
                continue
            return ClaudeCodeSessionInfo(
                session_id=session_id,
                compacted=bool(state.get("compacted", False)),
            )
        return None
    except Exception:
        return None


def get_latest_conversation() -> Optional[str]:
    files = list_session_files()
    return files[0].stem if files else None
