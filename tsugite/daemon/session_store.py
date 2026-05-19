"""Unified session store — replaces SessionManager and AgentSessionStore.

Single global metadata layer for all session types (interactive, schedule,
webhook, background, spawned). Conversation data stays in JSONL history files.
"""

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import uuid4

from tsugite.daemon.memory import DEFAULT_CONTEXT_LIMIT
from tsugite.history import SessionStorage, generate_session_id, get_history_dir


def _parse_ts(ts: str | None) -> datetime | None:
    """Parse an ISO timestamp string to datetime, handling Z and +00:00 formats."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


logger = logging.getLogger(__name__)


_SESSION_END_EVENT_TYPES = frozenset(
    {"session_complete", "session_error", "session_cancelled", "final_result", "error", "cancelled", "session_end"}
)


def _progress_status_text(event: dict) -> Optional[str]:
    """Render a short status label for a mid-session progress event.

    Handles both the broadcast event names (turn_start, tool_result,
    llm_wait_progress, ...) seen via SSE and the persisted event names
    (model_request, code_execution, tool_invocation, ...) replayed when
    rebuilding the sidebar status after a page refresh.
    """
    etype = event.get("type")
    if etype == "session_start":
        return "Starting..."
    if etype == "init":
        agent = event.get("agent")
        return f"Agent: {agent}" if agent else "Starting..."
    if etype == "turn_start":
        turn = event.get("turn")
        return f"Turn {turn}..." if turn is not None else "Working..."
    if etype == "thought":
        return "Thinking..."
    if etype == "reasoning_content":
        return "Reasoning..."
    if etype == "tool_result":
        return f"Tool: {event['tool']}" if _is_real_tool_event(event) else None
    if etype == "tool_call":
        name = event.get("tool")
        return f"Tool: {name}" if name else None
    if etype == "tool_invocation":
        name = event.get("name")
        return f"Tool: {name}" if name else None
    if etype == "code_execution":
        return "Running code..."
    if etype == "model_request":
        return "Waiting on LLM..."
    if etype == "hook_status":
        return event.get("message")
    if etype == "llm_wait_progress":
        secs = event.get("elapsed_seconds")
        return f"Waiting on LLM ({secs}s)" if secs else "Waiting on LLM..."
    return None


def _is_real_tool_event(event: dict) -> bool:
    """True for events that count toward the tool counter — broadcast tool_result
    with a named tool, or persisted tool_invocation. tool_call is NOT counted here
    because the matching tool_result fires later for the same invocation."""
    etype = event.get("type")
    if etype == "tool_result":
        return (event.get("tool") or "unknown") != "unknown"
    if etype == "tool_invocation":
        return bool(event.get("name"))
    return False


def _empty_progress() -> dict:
    return {
        "turn_count": 0,
        "tool_count": 0,
        "status_text": "Starting...",
        "last_event_time": None,
    }


def _apply_event_to_progress(progress: dict, event: dict) -> None:
    """Fold one event into a progress dict in place.

    Mirrors `_progress_from_events` so the cache can be primed from a full
    event list and then updated incrementally without reloading the file.
    """
    etype = event.get("type")
    progress["last_event_time"] = event.get("timestamp") or progress.get("last_event_time")
    if etype in _SESSION_END_EVENT_TYPES:
        progress["turn_count"] = 0
        progress["tool_count"] = 0
        progress["status_text"] = ""
        return
    if etype == "turn_start":
        turn = event.get("turn")
        if isinstance(turn, int) and turn > progress.get("turn_count", 0):
            progress["turn_count"] = turn
    elif _is_real_tool_event(event):
        progress["tool_count"] = progress.get("tool_count", 0) + 1
    label = _progress_status_text(event)
    if label:
        progress["status_text"] = label


def _progress_from_events(events: list[dict]) -> dict:
    """Compute a progress summary dict from the raw event list.

    A session/turn-end event clears live progress fields so the sidebar doesn't
    re-render a stale label between turns of an active session.
    """
    progress = _empty_progress()
    for event in events:
        _apply_event_to_progress(progress, event)
    return progress


READ_ONLY_METADATA_KEYS = frozenset(
    {
        "source",
        "user_id",
        "thread_id",
        "channel_id",
        "parent_session_id",
        "created_at",
        "started_at",
    }
)

METADATA_SESSION_NAME = "session_name"
METADATA_PRIMARY_FLAG = "is_primary"

# Metadata keys preserved across compaction in addition to READ_ONLY ones. session_name
# anchors named-route adapters (e.g. Discord session_name) to the successor session;
# is_primary makes the user's chosen primary session "follow" compaction.
COMPACTION_PRESERVED_METADATA_KEYS = READ_ONLY_METADATA_KEYS | frozenset({METADATA_SESSION_NAME, METADATA_PRIMARY_FLAG})

TOPIC_MAX_LENGTH = 160


class SessionSource(str, Enum):
    INTERACTIVE = "interactive"
    SCHEDULE = "schedule"

    BACKGROUND = "background"
    SPAWNED = "spawned"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"

    RUNNING = "running"
    FAILED = "failed"
    CANCELLED = "cancelled"


FINISHED_STATUSES = (SessionStatus.CANCELLED.value, SessionStatus.COMPLETED.value, SessionStatus.FAILED.value)


@dataclass
class Session:
    id: str
    agent: str
    source: str = SessionSource.INTERACTIVE.value
    status: str = SessionStatus.ACTIVE.value
    parent_id: Optional[str] = None
    user_id: Optional[str] = None
    created_at: str = ""
    last_active: str = ""
    metadata: dict = field(default_factory=dict)

    # Token tracking (for compaction)
    cumulative_tokens: int = 0
    message_count: int = 0

    # Background session fields
    prompt: Optional[str] = None
    error: Optional[str] = None
    result: Optional[str] = None
    model: Optional[str] = None
    agent_file: Optional[str] = None
    notify: list[str] = field(default_factory=list)

    title: Optional[str] = None
    scratchpad: str = ""

    pinned: bool = False
    pin_position: Optional[int] = None
    last_viewed_at: str = ""
    superseded_by: Optional[str] = None

    # Per-session UI / runtime state. Persisted so values survive daemon restart.
    sticky_skills: dict[str, int] = field(default_factory=dict)
    suppressed_skills: list[str] = field(default_factory=list)
    reasoning_effort: Optional[str] = None
    model_override: Optional[str] = None
    compacting: bool = False
    # Provider-reported context window for this session's model. None until the
    # first turn reports it; consumers fall back to the agent-wide default via
    # SessionStore.get_session_context_limit. Per-session so a compact-model
    # call (or any other secondary-model side effect) can't clobber the value
    # other sessions are reading.
    context_limit: Optional[int] = None

    @property
    def is_primary(self) -> bool:
        return bool(self.metadata.get(METADATA_PRIMARY_FLAG))

    def __post_init__(self):
        if not self.id:
            self.id = f"session-{uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.last_active:
            self.last_active = now


class SessionStore:
    """Global unified session metadata store.

    One instance shared across all agents and adapters.
    Persists to {state_dir}/session_store.json.
    """

    def __init__(
        self,
        store_path: Path,
        context_limits: Optional[dict[str, int]] = None,
        history_dir: Optional[Path] = None,
    ):
        self._path = store_path
        # Where per-session event logs live. Defaults to `<store_path parent>/history`
        # so tests using tmp_path/session_store.json get isolated tmp_path/history/,
        # while production callers can pass the XDG history dir explicitly.
        self._history_dir = history_dir if history_dir is not None else (store_path.parent / "history")
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._dirty = False
        self._save_dir_created = False

        # Per-agent context limits for compaction
        self._context_limits: dict[str, int] = context_limits or {}

        # Index: platform_thread_id -> session_id for fast thread lookup
        self._thread_index: dict[str, str] = {}

        # Index: (channel_id, agent) -> session_id for channel session lookup
        self._channel_index: dict[tuple[str, str], str] = {}

        # Per-(user_id, agent) compaction synchronization. Event is unset while
        # compaction is in progress, set when done. Per-session compacting state
        # is stored on Session itself (Session.compacting); this map only gates
        # concurrent begin_compaction calls.
        self._compaction_events: dict[tuple[str, str], threading.Event] = {}

        # Hot caches keyed by session_id, populated lazily on first read and
        # then updated incrementally inside `append_event`. Without these,
        # `session_progress_summary` and `event_count` would re-parse the
        # full .jsonl on every sidebar refresh — at 800+ sessions and
        # multi-MB active session files that's tens of MB of file I/O per
        # SSE-driven update.
        self._progress_cache: dict[str, dict] = {}
        self._event_count_cache: dict[str, int] = {}
        self._cache_lock = threading.Lock()

        self._load()
        self._migrate_legacy()
        self._recover_stale_sessions()

    # ── Context limit management ──

    def get_context_limit(self, agent: str) -> int:
        return self._context_limits.get(agent, DEFAULT_CONTEXT_LIMIT)

    def get_compaction_threshold(self, agent: str) -> int:
        return int(self.get_context_limit(agent) * 0.8)

    def update_context_limit(self, agent: str, limit: int) -> None:
        self._context_limits[agent] = limit

    def get_session_context_limit(self, session_id: str) -> int:
        """Return the session's tracked context window, falling back to the
        agent-wide default when the session hasn't completed a turn yet.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return DEFAULT_CONTEXT_LIMIT
            if session.context_limit is not None:
                return session.context_limit
            return self._context_limits.get(session.agent, DEFAULT_CONTEXT_LIMIT)

    def update_session_context_limit(self, session_id: str, limit: int) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.context_limit == limit:
                # Guard against write amplification: providers report a static
                # context_window per model, so every turn after the first writes
                # the same value. Without this, _mark_dirty triggers a JSON
                # rewrite on every turn for no reason.
                return
            session.context_limit = limit
            self._mark_dirty()

    def get_session_compaction_threshold(self, session_id: str) -> int:
        return int(self.get_session_context_limit(session_id) * 0.8)

    # ── Compaction locking ──

    def begin_compaction(self, user_id: str, agent: str, session_id: str | None = None) -> bool:
        """Try to start compaction. Returns True if this caller should compact.

        If another caller is already compacting this session, returns False.
        Pass session_id so per-session UI state (e.g. the "compacting…" chip)
        is also scoped via Session.compacting.
        """
        with self._lock:
            key = (user_id, agent)
            if key in self._compaction_events:
                return False
            self._compaction_events[key] = threading.Event()
            if session_id:
                session = self._sessions.get(session_id)
                if session:
                    session.compacting = True
                    self._mark_dirty()
            return True

    def end_compaction(self, user_id: str, agent: str, session_id: str | None = None) -> None:
        """Signal that compaction is complete. Wakes all waiters."""
        with self._lock:
            key = (user_id, agent)
            event = self._compaction_events.pop(key, None)
            if session_id:
                session = self._sessions.get(session_id)
                if session:
                    session.compacting = False
                    self._mark_dirty()
        if event:
            event.set()

    def wait_for_compaction(self, user_id: str, agent: str, timeout: float = 300) -> bool:
        """Block until an in-progress compaction finishes. Returns True if done, False on timeout."""
        with self._lock:
            event = self._compaction_events.get((user_id, agent))
        if event is None:
            return True
        return event.wait(timeout=timeout)

    def is_compacting(self, user_id: str, agent: str, session_id: str | None = None) -> bool:
        """Check if a session is currently being compacted.

        With session_id: per-session answer from Session.compacting.
        Without: per-(user, agent) lock state.
        """
        with self._lock:
            if session_id is not None:
                session = self._sessions.get(session_id)
                return bool(session and session.compacting)
            return (user_id, agent) in self._compaction_events

    # ── Per-session skill / model / effort state (lives on Session) ──

    def suppress_skill(self, session_id: str, skill_name: str) -> None:
        """Mark a skill as suppressed for the given session.

        AgentPreparer will skip this skill on subsequent turns so it does not
        reload from auto_load_skills or trigger matches.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session and skill_name not in session.suppressed_skills:
                session.suppressed_skills.append(skill_name)
                self._mark_dirty()

    def unsuppress_skill(self, session_id: str, skill_name: str) -> None:
        """Remove a skill from the session's suppression set."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session and skill_name in session.suppressed_skills:
                session.suppressed_skills.remove(skill_name)
                self._mark_dirty()

    def get_suppressed_skills(self, session_id: str) -> set[str]:
        """Return a copy of the session's suppressed skill names."""
        with self._lock:
            session = self._sessions.get(session_id)
            return set(session.suppressed_skills) if session else set()

    def mark_sticky(self, session_id: str, skill_name: str) -> None:
        """Mark a skill as sticky for the session and reset its unused-turn counter.

        Called when the skill is first trigger-matched or dynamically loaded, and
        again any time it's referenced (so the counter restarts at 0).
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.sticky_skills[skill_name] = 0
                self._mark_dirty()

    def drop_sticky(self, session_id: str, skill_name: str) -> None:
        """Remove a skill from the sticky set."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session and skill_name in session.sticky_skills:
                del session.sticky_skills[skill_name]
                self._mark_dirty()

    def get_sticky_skills(self, session_id: str) -> dict[str, int]:
        """Return a copy of the session's sticky skill counters."""
        with self._lock:
            session = self._sessions.get(session_id)
            return dict(session.sticky_skills) if session else {}

    def set_reasoning_effort(self, session_id: str, value: str | None) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.reasoning_effort = value or None
                self._mark_dirty()

    def get_reasoning_effort(self, session_id: str) -> str | None:
        with self._lock:
            session = self._sessions.get(session_id)
            return session.reasoning_effort if session else None

    def freeze_session_models_to_current(self, agent: str, current_model: str | None) -> None:
        """Pin every active, non-superseded session for `agent` that doesn't
        already have a `model_override` to `current_model`. Used when the agent
        default model is about to change, so existing sessions stay on whatever
        model they were resolving to instead of silently switching on their
        next turn.

        No-op when `current_model` is falsy (nothing to pin to).
        """
        if not current_model:
            return
        with self._lock:
            for session in self._sessions.values():
                if session.agent != agent:
                    continue
                if session.status in FINISHED_STATUSES:
                    continue
                if session.superseded_by:
                    continue
                if session.model_override:
                    continue
                session.model_override = current_model
            self._mark_dirty()

    def set_model_override(self, session_id: str, value: str | None) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.model_override = value or None
                self._mark_dirty()

    def get_model_override(self, session_id: str) -> str | None:
        with self._lock:
            session = self._sessions.get(session_id)
            return session.model_override if session else None

    def set_agent_override(self, session_id: str, value: str | None) -> None:
        """Update the agent associated with a session.

        Unlike model/effort, the agent is stored on the Session itself (it
        determines which adapter handles future turns), so we mutate the
        session record directly. Returns silently if the session is unknown.

        TODO: We're allowing changing the agent, but I'm not sure if that's going to just
              confuse the llm in the middle of a session?  Will need to test.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session or not value:
                return
            if session.agent == value:
                return
            session.agent = value
            self._save()

    def bump_unused_counters(self, session_id: str, referenced: set[str]) -> None:
        """Advance one turn: reset referenced skills, increment the rest.

        A skill is "referenced" this turn if the agent called load_skill() on it
        or the scanner found its name/trigger in the user message or final answer.
        Unreferenced skills get their counter incremented. Callers decide whether
        to drop skills whose counter now exceeds their TTL (we don't know per-skill
        TTL here — that's a frontmatter/config concern).
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session or not session.sticky_skills:
                return
            for name in list(session.sticky_skills):
                if name in referenced:
                    session.sticky_skills[name] = 0
                else:
                    session.sticky_skills[name] += 1
            self._mark_dirty()

    # ── Interactive session management ──

    def get_or_create_interactive(self, user_id: str, agent: str) -> Session:
        """Return the user's primary session, or create a fresh default one."""
        return self.find_default_session(user_id, agent) or self.create_default_session(user_id, agent)

    def default_primary_ids(self, agent: str) -> dict:
        """Return {user_id: session_id} for all primary sessions for this agent."""
        with self._lock:
            return {
                s.user_id: s.id
                for s in self._sessions.values()
                if s.agent == agent
                and s.user_id
                and s.is_primary
                and s.superseded_by is None
                and s.status not in FINISHED_STATUSES
            }

    def _find_named_session_locked(self, user_id: str, agent: str, name: str) -> Optional[Session]:
        """Lock-held variant of find_named_session — caller must hold self._lock."""
        candidates = [
            s
            for s in self._sessions.values()
            if s.user_id == user_id
            and s.agent == agent
            and s.superseded_by is None
            and s.status not in FINISHED_STATUSES
            and s.metadata.get(METADATA_SESSION_NAME) == name
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.last_active)

    def find_named_session(self, user_id: str, agent: str, name: str) -> Optional[Session]:
        """Find the latest non-finished, non-superseded session tagged with metadata.session_name."""
        with self._lock:
            return self._find_named_session_locked(user_id, agent, name)

    def _find_primary_session_locked(self, user_id: str, agent: str) -> Optional[Session]:
        """Lock-held variant of find_primary_session. Caller must hold self._lock."""
        candidates = [
            s
            for s in self._sessions.values()
            if s.user_id == user_id
            and s.agent == agent
            and s.superseded_by is None
            and s.status not in FINISHED_STATUSES
            and s.is_primary
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.last_active)

    def find_primary_session(self, user_id: str, agent: str) -> Optional[Session]:
        """Return the user's primary session for this agent, or None."""
        with self._lock:
            return self._find_primary_session_locked(user_id, agent)

    def find_default_session(self, user_id: str, agent: str) -> Optional[Session]:
        """Canonical lookup for where a default request should land."""
        return self.find_primary_session(user_id, agent)

    def _demote_primaries_locked(
        self, user_id: str, agent: str, *, except_id: Optional[str] = None
    ) -> Optional[Session]:
        """Clear primary flag from all (user, agent) sessions except `except_id`. Returns the last one cleared."""
        cleared: Optional[Session] = None
        for s in self._sessions.values():
            if s.user_id == user_id and s.agent == agent and s.id != except_id and s.is_primary:
                s.metadata.pop(METADATA_PRIMARY_FLAG, None)
                cleared = s
        return cleared

    def create_default_session(self, user_id: str, agent: str, *, title: Optional[str] = None) -> Session:
        """Create a fresh interactive session and mark it primary."""
        with self._lock:
            conv_id = generate_session_id(agent)
            self._demote_primaries_locked(user_id, agent)
            session = Session(
                id=conv_id,
                agent=agent,
                source=SessionSource.INTERACTIVE.value,
                user_id=user_id,
                title=title,
                metadata={METADATA_PRIMARY_FLAG: True},
            )
            self._sessions[conv_id] = session
            self._save()
            return session

    def set_primary_session(self, session_id: str) -> Session:
        """Mark `session_id` as primary, demoting any prior primary for the same (user, agent)."""
        with self._lock:
            if session_id not in self._sessions:
                raise ValueError(f"Session '{session_id}' not found")
            target = self._sessions[session_id]
            if target.status in FINISHED_STATUSES:
                raise ValueError(f"Cannot promote finished session '{session_id}' to primary")
            if target.superseded_by:
                raise ValueError(f"Cannot promote superseded session '{session_id}' to primary")
            self._demote_primaries_locked(target.user_id, target.agent, except_id=target.id)
            target.metadata[METADATA_PRIMARY_FLAG] = True
            self._mark_dirty()
            return target

    def clear_primary_session(self, user_id: str, agent: str) -> Optional[Session]:
        """Remove the primary flag from any session for (user_id, agent). Returns the cleared session, if any."""
        with self._lock:
            cleared = self._demote_primaries_locked(user_id, agent)
            self._mark_dirty()
            return cleared

    def get_or_create_named_session(self, user_id: str, agent: str, name: str) -> Session:
        """Resolve a named-route session for (user_id, agent), creating one if absent.

        The session_name lives in metadata and is preserved across compaction so the
        named route follows the successor session automatically.
        """
        with self._lock:
            existing = self._find_named_session_locked(user_id, agent, name)
            if existing:
                return existing

            conv_id = f"daemon_{agent}_{user_id}_{name}_{uuid4().hex[:6]}"
            session = Session(
                id=conv_id,
                agent=agent,
                source=SessionSource.INTERACTIVE.value,
                user_id=user_id,
                title=f"{name.title()} Session",
                metadata={METADATA_SESSION_NAME: name},
            )
            tokens, msg_count = self._estimate_tokens(conv_id)
            session.cumulative_tokens = tokens
            session.message_count = msg_count
            self._sessions[conv_id] = session
            self._save()
            return session

    def needs_compaction(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if not session:
            return False
        return session.cumulative_tokens >= self.get_session_compaction_threshold(session_id)

    def compact_session(self, session_id: str) -> Session:
        with self._lock:
            old_session = self._sessions.get(session_id)
            if not old_session:
                raise ValueError(f"Session '{session_id}' not found")

            new_id = generate_session_id(old_session.agent)
            new_session = Session(
                id=new_id,
                agent=old_session.agent,
                source=old_session.source,
                user_id=old_session.user_id,
                parent_id=old_session.parent_id,
                metadata={k: v for k, v in old_session.metadata.items() if k in COMPACTION_PRESERVED_METADATA_KEYS},
                scratchpad=old_session.scratchpad,
                title=old_session.title,
                pinned=old_session.pinned,
                pin_position=old_session.pin_position,
                # Carry per-session UI/runtime state forward — compaction is "same
                # conversation" from the user's POV, so suppressions, sticky-skill
                # TTL counters, and effort/model overrides should follow the rotation.
                sticky_skills=dict(old_session.sticky_skills),
                suppressed_skills=list(old_session.suppressed_skills),
                reasoning_effort=old_session.reasoning_effort,
                model_override=old_session.model_override,
                context_limit=old_session.context_limit,
            )
            # Preserve original conversation start so <session_started> in the
            # message context reflects the user's perceived session age, not
            # the compaction moment.
            new_session.created_at = old_session.created_at

            self._sessions[new_id] = new_session

            # Compaction preserves is_primary metadata, so the new session automatically
            # becomes the user's default if the predecessor was. Thread index needs updating.
            thread_id = new_session.metadata.get("thread_id")
            if thread_id:
                self._thread_index[thread_id] = new_id

            # Mark old session as completed and superseded so it stops appearing in
            # the default sidebar list (the new session is the live continuation).
            old_session.status = SessionStatus.COMPLETED.value
            old_session.superseded_by = new_id
            old_session.pinned = False
            old_session.pin_position = None
            # The new session owns the per-session UI/runtime state now; clear
            # the old's so superseded sessions don't carry orphan state.
            old_session.sticky_skills = {}
            old_session.suppressed_skills = []
            old_session.reasoning_effort = None
            old_session.model_override = None
            old_session.compacting = False

            self._save()
        self._evict_progress_cache(session_id)
        return new_session

    def resolve_compacted_successor(self, session_id: str) -> Optional[Session]:
        """Return the post-compaction successor of `session_id`, or None.

        Folds the `superseded_by` lookup and the successor fetch into one lock
        acquisition. Returns None when `session_id` is unknown, has no
        successor, or the successor itself has been pruned.
        """
        with self._lock:
            old = self._sessions.get(session_id)
            if not old or not old.superseded_by:
                return None
            return self._sessions.get(old.superseded_by)

    def update_token_count(self, session_id: str, tokens_used: int) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                if tokens_used > 0:
                    session.cumulative_tokens = tokens_used
                session.message_count += 1
                session.last_active = datetime.now(timezone.utc).isoformat()
                self._mark_dirty()

    def set_cumulative_tokens(self, session_id: str, tokens: int) -> None:
        """Set cumulative_tokens without bumping message_count or last_active.

        Used to seed a fresh post-compaction session with an estimate of its
        carried-over context size, and to sync from prompt_snapshot totals so
        the UI badge matches the inspector. Real exchanges go through
        update_token_count.
        """
        if tokens <= 0:
            return
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.cumulative_tokens != tokens:
                session.cumulative_tokens = tokens
                self._mark_dirty()

    # ── Generic session CRUD ──

    MAX_SCHEDULE_SESSIONS = 20
    MAX_BACKGROUND_SESSIONS = 100

    def create_session(self, session: Session) -> Session:
        with self._lock:
            if session.id in self._sessions:
                raise ValueError(f"Session '{session.id}' already exists")
            self._sessions[session.id] = session

            thread_id = session.metadata.get("thread_id")
            if thread_id:
                self._thread_index[thread_id] = session.id

            if session.source == SessionSource.SCHEDULE.value and session.parent_id:
                self._prune_schedule_sessions(session.parent_id)
            elif session.source in (SessionSource.BACKGROUND.value, SessionSource.SPAWNED.value):
                self._prune_background_sessions(session.agent)

            self._save()
            return session

    def _purge_session_state(self, session_id: str) -> None:
        """Remove a session plus its derived indexes and hot caches.

        Per-session runtime state lives on the Session itself (sticky_skills,
        suppressed_skills, reasoning_effort, model_override, compacting) so it
        drops with the session. What still needs explicit cleanup: the reverse
        lookup indexes (thread_id / channel_id → session_id) and the hot caches
        keyed by session_id. Caller holds `self._lock`; `_cache_lock` is taken
        briefly inside.
        """
        self._sessions.pop(session_id, None)
        for tid, sid in list(self._thread_index.items()):
            if sid == session_id:
                del self._thread_index[tid]
        for key, sid in list(self._channel_index.items()):
            if sid == session_id:
                del self._channel_index[key]
        with self._cache_lock:
            self._progress_cache.pop(session_id, None)
            self._event_count_cache.pop(session_id, None)

    def _prune_schedule_sessions(self, parent_id: str) -> None:
        """Remove oldest completed schedule sessions beyond MAX_SCHEDULE_SESSIONS. Must hold lock."""
        children = [
            s for s in self._sessions.values() if s.source == SessionSource.SCHEDULE.value and s.parent_id == parent_id
        ]
        if len(children) <= self.MAX_SCHEDULE_SESSIONS:
            return
        children.sort(key=lambda s: s.created_at)
        for s in children[: len(children) - self.MAX_SCHEDULE_SESSIONS]:
            if s.status in (SessionStatus.COMPLETED.value, SessionStatus.FAILED.value):
                self._purge_session_state(s.id)

    def _prune_background_sessions(self, agent: str) -> None:
        """Remove oldest completed background/spawned sessions beyond MAX_BACKGROUND_SESSIONS. Must hold lock."""
        children = [
            s
            for s in self._sessions.values()
            if s.agent == agent
            and s.source in (SessionSource.BACKGROUND.value, SessionSource.SPAWNED.value)
            and s.status in FINISHED_STATUSES
        ]
        if len(children) <= self.MAX_BACKGROUND_SESSIONS:
            return
        children.sort(key=lambda s: s.created_at)
        for s in children[: len(children) - self.MAX_BACKGROUND_SESSIONS]:
            self._purge_session_state(s.id)

    def get_session(self, session_id: str) -> Session:
        with self._lock:
            if session_id not in self._sessions:
                raise ValueError(f"Session '{session_id}' not found")
            return self._sessions[session_id]

    def update_session(self, session_id: str, **fields) -> Session:
        with self._lock:
            if session_id not in self._sessions:
                raise ValueError(f"Session '{session_id}' not found")
            session = self._sessions[session_id]
            for key, value in fields.items():
                if key in ("id", "created_at"):
                    continue
                if not hasattr(session, key):
                    raise ValueError(f"Unknown field '{key}'")
                setattr(session, key, value)
            session.last_active = datetime.now(timezone.utc).isoformat()
            self._mark_dirty()
        if session.status in FINISHED_STATUSES:
            self._evict_progress_cache(session_id)
        return session

    def _evict_progress_cache(self, session_id: str) -> None:
        """Drop a session's live-progress entry once it stops appending events.

        Sidebar refreshes only call `session_progress_summary` for sessions in
        live statuses, so finished sessions never re-read the cache; keeping
        them resident grows memory without bound across daemon uptime. The
        event_count entry stays — it's still hit by `session_detail`.
        """
        with self._cache_lock:
            self._progress_cache.pop(session_id, None)

    def _pinned_for_agent(self, agent: str, exclude_id: Optional[str] = None) -> list[Session]:
        """Return pinned sessions for an agent, sorted by current pin_position (None last)."""
        return sorted(
            [s for s in self._sessions.values() if s.agent == agent and s.pinned and s.id != exclude_id],
            key=lambda s: s.pin_position if s.pin_position is not None else 0,
        )

    def set_pin(self, session_id: str, pinned: bool, position: Optional[int] = None) -> Session:
        """Pin or unpin a session. Pinning appends to the end unless position is given;
        unpinning densifies the remaining pinned sessions for the same agent.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise ValueError(f"Session '{session_id}' not found")

            if pinned and session.pinned and position is None:
                return session
            if not pinned and not session.pinned:
                return session

            session.pinned = pinned
            if not pinned:
                session.pin_position = None
                for i, s in enumerate(self._pinned_for_agent(session.agent)):
                    s.pin_position = i
            else:
                others = self._pinned_for_agent(session.agent, exclude_id=session_id)
                insert_at = len(others) if position is None else max(0, min(position, len(others)))
                session.pin_position = insert_at
                for i, s in enumerate(others):
                    s.pin_position = i if i < insert_at else i + 1

            session.last_active = datetime.now(timezone.utc).isoformat()
            self._mark_dirty()
            return session

    def reorder_pins(self, ordered_ids: list[str]) -> list[Session]:
        """Write pin_position 0..N-1 for the given pinned session ids; unknown or
        unpinned ids are silently skipped.
        """
        with self._lock:
            valid = [self._sessions[sid] for sid in ordered_ids if sid in self._sessions and self._sessions[sid].pinned]
            for i, s in enumerate(valid):
                s.pin_position = i
            if valid:
                self._mark_dirty()
            return valid

    def mark_viewed(self, session_id: str, ts: Optional[str] = None) -> Session:
        """Set last_viewed_at on a session. Defaults to now (UTC ISO)."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise ValueError(f"Session '{session_id}' not found")
            session.last_viewed_at = ts or datetime.now(timezone.utc).isoformat()
            self._mark_dirty()
            return session

    def list_sessions(
        self,
        agent: Optional[str] = None,
        source: Optional[str] = None,
        parent_id: Optional[str] = None,
        status: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 0,
        updated_since: Optional[str] = None,
        include_superseded: bool = False,
    ) -> list[Session]:
        _updated_since_dt = _parse_ts(updated_since)
        _epoch = datetime.min.replace(tzinfo=timezone.utc)
        with self._lock:
            results = [
                s
                for s in self._sessions.values()
                if (not agent or s.agent == agent)
                and (not source or s.source == source)
                and (not parent_id or s.parent_id == parent_id)
                and (not status or s.status == status)
                and (not user_id or s.user_id == user_id)
                and (not _updated_since_dt or (_parse_ts(s.last_active) or _epoch) >= _updated_since_dt)
                and (include_superseded or not s.superseded_by)
            ]
            if limit:
                results.sort(key=lambda s: s.last_active or s.created_at, reverse=True)
                results = results[:limit]
            return results

    def list_interactive_by_agent(self, agent: str) -> list[Session]:
        """Return all active interactive sessions for a given agent."""
        with self._lock:
            return [
                s
                for s in self._sessions.values()
                if s.agent == agent
                and s.source == SessionSource.INTERACTIVE.value
                and s.status == SessionStatus.ACTIVE.value
            ]

    # ── Event log: unified with conversation history ──
    #
    # UI events (reactions, prompt_snapshots, etc.) are stored in the same
    # `history/{session_id}.jsonl` file as the conversation events recorded by
    # the agent loop. There is no separate daemon/sessions/ log.

    def _history_path(self, session_id: str) -> Path:
        return self._history_dir / f"{session_id}.jsonl"

    def append_event(self, session_id: str, event: dict) -> None:
        """Append a UI/telemetry event to the session's history JSONL.

        Accepts the legacy flat-dict shape `{type, timestamp, ...rest}` and
        translates it into the per-event Event schema. Creates the file if it
        doesn't exist (without injecting an implicit session_start) so that
        callers like the SSE handler don't have to coordinate file creation
        with the agent loop.
        """
        path = self._history_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        storage = SessionStorage(path)  # bare wrapper; doesn't write anything
        ts = _parse_ts(event.get("timestamp"))
        data = {k: v for k, v in event.items() if k not in ("type", "timestamp")}
        storage.record(event.get("type", "unknown"), ts=ts, **data)

        # Update hot caches incrementally. Skip when the session_id has never
        # been read — the cold-load path will populate everything in one go.
        with self._cache_lock:
            if session_id in self._event_count_cache:
                self._event_count_cache[session_id] += 1
            progress = self._progress_cache.get(session_id)
            if progress is not None:
                _apply_event_to_progress(progress, event)

    def read_events(self, session_id: str) -> list[dict]:
        """Return events as flat dicts for backward compatibility with callers."""
        path = self._history_path(session_id)
        if not path.exists():
            return []
        try:
            storage = SessionStorage.load(path)
        except Exception:
            return []
        return [{"type": e.type, "timestamp": e.ts.isoformat(), **e.data} for e in storage.iter_events()]

    def event_count(self, session_id: str) -> int:
        with self._cache_lock:
            cached = self._event_count_cache.get(session_id)
        if cached is not None:
            return cached
        path = self._history_path(session_id)
        if not path.exists():
            with self._cache_lock:
                self._event_count_cache[session_id] = 0
            return 0
        try:
            storage = SessionStorage.load(path)
            count = sum(1 for _ in storage.iter_events())
        except Exception:
            count = 0
        with self._cache_lock:
            self._event_count_cache[session_id] = count
        return count

    def count_events_by_type(self, session_id: str, event_type: str) -> int:
        path = self._history_path(session_id)
        if not path.exists():
            return 0
        try:
            storage = SessionStorage.load(path)
        except Exception:
            return 0
        return sum(1 for _ in storage.iter_events(types=[event_type]))

    def session_detail(self, session_id: str) -> dict:
        session = self.get_session(session_id)
        result = asdict(session)
        result["event_count"] = self.event_count(session_id)
        result["is_primary"] = session.is_primary
        return result

    def session_events_since(self, session_id: str, since: Optional[str] = None) -> list[dict]:
        """Return events for a session, optionally filtered to those after a timestamp."""
        events = self.read_events(session_id)
        if not since:
            return events
        since_dt = _parse_ts(since)
        if not since_dt:
            return events
        return [
            e for e in events if (_parse_ts(e.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc)) > since_dt
        ]

    def session_progress_summary(self, session_id: str) -> dict:
        """Return a lightweight live-progress summary for a running session.

        Fields are derived entirely from events.jsonl so they stay consistent
        with what the UI would render if it replayed the event log. The first
        call cold-loads the file; subsequent calls hit the in-memory cache,
        which `append_event` keeps current.

        Callers MUST treat the returned dict as read-only — it's the cached
        object itself, shared across calls. Mutating it corrupts the cache.
        The sole production caller hands the result straight to JSONResponse
        and never mutates.
        """
        with self._cache_lock:
            cached = self._progress_cache.get(session_id)
        if cached is not None:
            return cached
        events = self.read_events(session_id)
        progress = _progress_from_events(events)
        with self._cache_lock:
            self._progress_cache[session_id] = progress
            if session_id not in self._event_count_cache:
                self._event_count_cache[session_id] = len(events)
        return progress

    def session_summary(self, session_id: str) -> dict:
        """Return a summary dict for a session including event stats."""
        with self._lock:
            if session_id not in self._sessions:
                raise ValueError(f"Session '{session_id}' not found")
            session = self._sessions[session_id]
            events = self.read_events(session_id)
        tools_used = sorted({e["name"] for e in events if e.get("type") == "tool_call" and "name" in e})

        summary: dict = {
            "id": session.id,
            "agent": session.agent,
            "source": session.source,
            "status": session.status,
            "prompt": session.prompt or "",
            "result": session.result or "",
            "event_count": len(events),
            "tools_used": tools_used,
        }
        if session.error:
            summary["error"] = session.error
        return summary

    # ── Metadata CRUD ──

    def set_metadata(self, session_id: str, key: str, value) -> Session:
        """Set a single metadata key. Raises ValueError for read-only keys."""
        return self.set_metadata_bulk(session_id, {key: value})

    def set_metadata_bulk(self, session_id: str, updates: dict) -> Session:
        """Set multiple metadata keys. Rejects entire batch if any key is read-only.

        Does not bump `last_active`: metadata is housekeeping (status_text,
        topic, task, etc.), not new message activity the user hasn't seen. Bumping
        it here clobbers the post-`mark-viewed` clear because the unread flag is
        derived as `last_active > last_viewed_at`.
        """
        read_only = READ_ONLY_METADATA_KEYS & updates.keys()
        if read_only:
            raise ValueError(f"Cannot set read-only metadata key(s): {', '.join(sorted(read_only))}")
        if "topic" in updates:
            topic = updates["topic"]
            if not isinstance(topic, str):
                raise ValueError("Topic must be a string")
            if len(topic) > TOPIC_MAX_LENGTH:
                raise ValueError(f"Topic must be {TOPIC_MAX_LENGTH} characters or fewer (got {len(topic)})")
        with self._lock:
            if session_id not in self._sessions:
                raise ValueError(f"Session '{session_id}' not found")
            session = self._sessions[session_id]
            session.metadata.update(updates)
            self._mark_dirty()
            return session

    def delete_metadata(self, session_id: str, key: str) -> Session:
        """Delete a metadata key. Raises ValueError for read-only or missing keys.

        See `set_metadata_bulk` for why this doesn't bump `last_active`.
        """
        if key in READ_ONLY_METADATA_KEYS:
            raise ValueError(f"Cannot delete read-only metadata key: {key}")
        with self._lock:
            if session_id not in self._sessions:
                raise ValueError(f"Session '{session_id}' not found")
            session = self._sessions[session_id]
            if key not in session.metadata:
                raise ValueError(f"Key '{key}' not found in metadata")
            del session.metadata[key]
            self._mark_dirty()
            return session

    # ── Channel session index ──

    def get_or_create_channel_session(self, channel_id: str, agent: str, user_id: str) -> Session:
        with self._lock:
            key = (channel_id, agent)
            is_replacement = False
            if key in self._channel_index:
                session_id = self._channel_index[key]
                if session_id in self._sessions:
                    existing = self._sessions[session_id]
                    if existing.status not in FINISHED_STATUSES:
                        return existing
                    is_replacement = True

            conv_id = (
                f"channel_{agent}_{channel_id}_{uuid4().hex[:6]}" if is_replacement else f"channel_{agent}_{channel_id}"
            )
            session = Session(
                id=conv_id,
                agent=agent,
                source=SessionSource.INTERACTIVE.value,
                user_id=user_id,
                metadata={"channel_id": channel_id},
            )
            tokens, msg_count = self._estimate_tokens(conv_id)
            session.cumulative_tokens = tokens
            session.message_count = msg_count
            self._sessions[conv_id] = session
            self._channel_index[key] = conv_id
            self._save()
            return session

    def find_by_channel(self, channel_id: str, agent: str) -> Optional[Session]:
        with self._lock:
            session_id = self._channel_index.get((channel_id, agent))
            if session_id:
                return self._sessions.get(session_id)
        return None

    # ── Thread lookup ──

    def find_by_thread(self, platform_thread_id: str) -> Optional[Session]:
        with self._lock:
            session_id = self._thread_index.get(platform_thread_id)
            if session_id:
                return self._sessions.get(session_id)
        return None

    # ── Persistence ──

    def _load(self):
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            valid_fields = {f.name for f in dataclass_fields(Session)}
            for sid, sdata in data.get("sessions", {}).items():
                # Migrate platform_thread_id -> metadata["thread_id"]
                old_thread_id = sdata.pop("platform_thread_id", None)
                if old_thread_id:
                    meta = sdata.get("metadata") or {}
                    meta.setdefault("thread_id", old_thread_id)
                    sdata["metadata"] = meta
                sdata = {k: v for k, v in sdata.items() if k in valid_fields}
                self._sessions[sid] = Session(**sdata)
            # Rebuild indexes. Legacy stores have no is_primary flag; stamp it on the
            # most-recently-active interactive session per (user, agent) to preserve
            # the user's existing default-routing across the upgrade.
            primary_candidates: dict[tuple[str, str], str] = {}
            already_primary_keys: set[tuple[str, str]] = set()
            for sid, session in self._sessions.items():
                if (
                    session.source == SessionSource.INTERACTIVE.value
                    and session.user_id
                    and session.superseded_by is None
                    and session.status not in FINISHED_STATUSES
                ):
                    key = (session.user_id, session.agent)
                    if session.is_primary:
                        already_primary_keys.add(key)
                    existing_id = primary_candidates.get(key)
                    if not existing_id or session.last_active > self._sessions[existing_id].last_active:
                        primary_candidates[key] = sid
                thread_id = session.metadata.get("thread_id") if session.metadata else None
                if thread_id:
                    self._thread_index[thread_id] = sid
                channel_id = session.metadata.get("channel_id") if session.metadata else None
                if channel_id:
                    self._channel_index[(channel_id, session.agent)] = sid
            for key, sid in primary_candidates.items():
                if key not in already_primary_keys:
                    self._sessions[sid].metadata[METADATA_PRIMARY_FLAG] = True
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.error("Failed to load session store from %s: %s", self._path, e)

    def _mark_dirty(self):
        """Mark store as needing a save. Call flush() to persist."""
        self._dirty = True

    def flush(self):
        """Persist if dirty. Safe to call from outside the lock."""
        with self._lock:
            if self._dirty:
                self._save()
                self._dirty = False

    def _save(self):
        if not self._save_dir_created:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._save_dir_created = True
        data = {
            "sessions": {sid: asdict(s) for sid, s in self._sessions.items()},
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")))
        os.replace(str(tmp), str(self._path))
        self._dirty = False

    def _estimate_tokens(self, session_id: str) -> tuple[int, int]:
        try:
            session_path = get_history_dir() / f"{session_id}.jsonl"
            if not session_path.exists():
                return 0, 0
            storage = SessionStorage.load(session_path)
            last_tokens = 0
            user_input_count = 0
            for event in storage.iter_events():
                if event.type == "user_input":
                    user_input_count += 1
                elif event.type == "model_response":
                    usage = event.data.get("usage") or {}
                    if isinstance(usage, dict):
                        last_tokens = usage.get("total_tokens") or usage.get("input_tokens") or last_tokens
            return last_tokens, user_input_count
        except Exception:
            return 0, 0

    def _recover_stale_sessions(self):
        changed = False
        for session in self._sessions.values():
            if session.status == SessionStatus.RUNNING.value:
                session.status = SessionStatus.FAILED.value
                session.error = "Daemon restarted while session was active"
                session.last_active = datetime.now(timezone.utc).isoformat()
                changed = True
            # A session that was mid-compaction at restart can't still be — the
            # in-memory lock didn't survive. Clear the flag so the UI doesn't
            # show a stuck "compacting…" indicator.
            if session.compacting:
                session.compacting = False
                changed = True
        if changed:
            self._save()

    def _migrate_legacy(self):
        """Migrate from old SessionManager + AgentSessionStore if needed."""
        if self._sessions:
            return  # Already have data, skip migration

        state_dir = self._path.parent
        migrated = False

        # Migrate daemon_sessions/*.json (per-agent directories)
        for agent_dir in state_dir.iterdir():
            sessions_dir = agent_dir / "daemon_sessions" if agent_dir.is_dir() else None
            if not sessions_dir or not sessions_dir.is_dir():
                continue
            agent_name = agent_dir.name
            for path in sessions_dir.glob("*.json"):
                try:
                    data = json.loads(path.read_text())
                    user_id = path.stem.replace("_", ":")
                    conv_id = data.get("conversation_id", "")
                    if not conv_id:
                        continue
                    session = Session(
                        id=conv_id,
                        agent=agent_name,
                        source=SessionSource.INTERACTIVE.value,
                        user_id=user_id,
                        created_at=data.get("created_at", ""),
                        metadata={METADATA_PRIMARY_FLAG: True},
                    )
                    tokens, msg_count = self._estimate_tokens(conv_id)
                    session.cumulative_tokens = tokens
                    session.message_count = msg_count
                    self._sessions[conv_id] = session
                    migrated = True
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("Skipping legacy session file %s: %s", path, e)

        # Migrate sessions.json (AgentSessionStore)
        old_sessions_path = state_dir / "sessions.json"
        if old_sessions_path.exists():
            try:
                data = json.loads(old_sessions_path.read_text())
                for sid, sdata in data.get("sessions", {}).items():
                    session = Session(
                        id=sid,
                        agent=sdata.get("agent", "unknown"),
                        source=SessionSource.BACKGROUND.value,
                        status=sdata.get("state", SessionStatus.COMPLETED.value),
                        prompt=sdata.get("prompt"),
                        error=sdata.get("error"),
                        result=sdata.get("result"),
                        model=sdata.get("model"),
                        agent_file=sdata.get("agent_file"),
                        notify=sdata.get("notify", []),
                        created_at=sdata.get("created_at", ""),
                        last_active=sdata.get("updated_at", ""),
                    )
                    self._sessions[sid] = session
                    migrated = True
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                logger.warning("Failed to migrate legacy sessions.json: %s", e)

        if migrated:
            logger.info("Migrated %d legacy sessions to unified store", len(self._sessions))
            self._save()
