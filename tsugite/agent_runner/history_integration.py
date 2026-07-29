"""History integration for agent runs.

The agent loop now emits per-event records as it runs (model_request,
model_response, code_execution, etc.). This module handles session lifecycle
around the agent run: creating the session, recording user_input + session_end
events, exposing helpers to load past sessions for continuation.
"""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tsugite.attachments.base import Attachment
from tsugite.config import load_config
from tsugite.history import (
    Session,
    events_to_messages,
    get_history_backend,
    last_index_of,
)

logger = logging.getLogger(__name__)


def load_conversation_messages(conversation_id: str) -> List[Dict[str, Any]]:
    """Load conversation history as a messages list for an LLM call."""
    backend = get_history_backend()
    if not backend.exists(conversation_id):
        raise FileNotFoundError(f"Conversation not found: {conversation_id}")
    storage = backend.load(conversation_id)
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
) -> Optional[Session]:
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

    backend = get_history_backend()
    if continue_conversation_id:
        if backend.exists(continue_conversation_id):
            return backend.load(continue_conversation_id)
        return backend.create(
            agent_name=agent_name,
            model=model,
            workspace=workspace,
            session_id=continue_conversation_id,
        )

    return backend.create(agent_name=agent_name, model=model, workspace=workspace)


def _current_turn_already_has_user_input(storage: Session) -> bool:
    """True if the in-progress turn already recorded a user_input.

    A turn runs from its user_input to its session_end. A live-recorded turn
    that failed (or retried) before completing leaves a user_input with no
    session_end after it; the post-hoc save_run_to_history on the error path
    would otherwise record a second, identical one (a duplicate user bubble).
    A genuinely repeated message opens a new turn *after* the prior session_end,
    so this stays False for it.
    """
    for event in reversed(list(storage.iter_events(types=("user_input", "session_end")))):
        if event.type == "session_end":
            return False
        if event.type == "user_input":
            return True
    return False


def record_user_input(
    storage: Session,
    text: str,
    attachments: Optional[List[Attachment]] = None,
    channel_metadata: Optional[Dict[str, Any]] = None,
    client_context_items: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Record a user_input event at the start of a turn.

    Idempotent within a turn: the live runner and the error-path
    save_run_to_history can both reach here for the same message with no shared
    in-memory flag, so a turn that failed before completing is deduped from the
    session's own events rather than by content-matching (which would collapse
    legitimate repeats).
    """
    if _current_turn_already_has_user_input(storage):
        return
    data: Dict[str, Any] = {"text": text}
    # Only files the user actually uploaded ride here: this field is display-only
    # (the web UI renders them as clickable `uploads/<name>` chips). The agent's
    # auto-included context (workspace memory files like USER.md, config
    # attachments) is not a user attachment and lives outside uploads/, so listing
    # it produced dead "file not found" chips on every message.
    uploads = [a for a in attachments if getattr(a, "user_upload", False)] if attachments else []
    if uploads:
        data["attachments"] = [
            {"name": a.name, "type": a.content_type.value, "source_url": a.source_url} for a in uploads
        ]
    if channel_metadata:
        data["channel"] = channel_metadata
    # Structured client context, recorded alongside the folded <client_context> XML
    # in `text` so the UI reads it back without re-parsing the prompt string.
    if client_context_items:
        data["client_context"] = client_context_items
    storage.record("user_input", **data)


def record_final_result(
    storage: Session,
    result: str,
    result_data: Any = None,
    turns: Optional[int] = None,
    tokens: Optional[int] = None,
    cost: Optional[float] = None,
) -> None:
    """Record the turn's final answer as a durable final_result event.

    Interactive daemon turns may already have one for this turn, persisted by
    the live SSE handler during FinalAnswerEvent dispatch (which runs before
    the agent reaches its end-of-run recording). Skip in that case so the
    conversation view doesn't render the answer twice.
    """
    for event in reversed(list(storage.iter_events(types=("user_input", "final_result")))):
        if event.type == "user_input":
            break
        if event.type == "final_result":
            return
    storage.record(
        "final_result",
        result=result,
        result_data=result_data,
        turns=turns,
        tokens=tokens,
        cost=cost,
    )


def record_session_end(
    storage: Session,
    status: str = "success",
    error_message: Optional[str] = None,
) -> None:
    """Record a session_end event with final status.

    Note: this fires at the end of every agent run, i.e. once per turn for a
    daemon interactive session. So the history DB recording a session as
    ``status=success`` with an ``ended_at`` only means the most recent *turn*
    finished - it does NOT mean the conversation is closed. The daemon session
    store deliberately keeps such a session ``active`` because the user can still
    send more messages; the store is the source of truth for conversation
    liveness. The two stores tracking different things (last-turn outcome vs
    conversation open/closed) is expected, not a desync.
    """
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
class ResumableSessionState:
    """A session-owning provider's resume state recorded for a conversation."""

    session_id: str
    compacted: bool = False


def get_resumable_session_state(conversation_id: str) -> Optional[ResumableSessionState]:
    """Find the most recent resumable provider session from a conversation's events.

    Reads the ``session_id`` a session-owning provider recorded in its ``state_delta``.
    Returns None if none was recorded (a stateless provider was used) or if a compaction
    happened after the last one (the provider session is stale at that point).
    """
    try:
        backend = get_history_backend()
        if not backend.exists(conversation_id):
            return None
        storage = backend.load(conversation_id)
        events = storage.load_events()

        # Find the last boundary past which a recorded provider session is stale: a
        # compaction (history was summarized) or a resume_reset (the session was
        # severed because its transcript could no longer be resumed).
        compaction_idx = last_index_of(events, "compaction")
        reset_idx = last_index_of(events, "resume_reset")
        boundary_idx = max((idx for idx in (compaction_idx, reset_idx) if idx is not None), default=None)

        for i in range(len(events) - 1, -1, -1):
            event = events[i]
            if event.type != "model_response":
                continue
            state = event.data.get("state_delta") or {}
            session_id = state.get("session_id")
            if not session_id:
                continue
            if boundary_idx is not None and i <= boundary_idx:
                continue
            return ResumableSessionState(
                session_id=session_id,
                compacted=bool(state.get("compacted", False)),
            )
        return None
    except Exception:
        return None


def record_resume_reset(conversation_id: str, *, reason: str = "poisoned_transcript") -> Optional[Dict[str, Any]]:
    """Sever a conversation's resumable provider session by recording a boundary.

    A session-owning provider transcript can become unresumable (e.g. a Claude Code
    sidecar that picked up an empty text block and 400s on every resume). This
    boundary makes get_resumable_session_state stop returning the stale session id,
    so the next turn starts fresh from serialized history; the frontend renders it
    as a notice. Best-effort: a recording failure must not mask the triggering turn.

    Returns the recorded event data (so the caller can also surface it live on the
    same turn) when persisted, else None (empty id, no history, or a record error).
    """
    if not conversation_id:
        return None
    data: Dict[str, Any] = {
        "reason": reason,
        "message": (
            "The chat's resumable model session was reset because it could no "
            "longer be resumed; continuing from saved history."
        ),
    }
    try:
        backend = get_history_backend()
        if not backend.exists(conversation_id):
            return None
        backend.load(conversation_id).record("resume_reset", **data)
    except Exception:
        logger.warning("Failed to record resume_reset for %s", conversation_id, exc_info=True)
        return None
    return data


def get_latest_conversation() -> Optional[str]:
    sessions = get_history_backend().list_sessions(limit=1)
    return sessions[0] if sessions else None
