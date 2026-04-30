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
    {"session_complete", "session_error", "session_cancelled", "final_result", "error", "cancelled"}
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
    with a named tool, or persisted tool_invocation."""
    etype = event.get("type")
    if etype == "tool_result":
        return (event.get("tool") or "unknown") != "unknown"
    if etype == "tool_invocation":
        return bool(event.get("name"))
    return False


def _progress_from_events(events: list[dict]) -> dict:
    """Compute a progress summary dict from the raw event list.

    A session/turn-end event clears live progress fields so the sidebar doesn't
    re-render a stale label between turns of an active session.
    """
    turn_count = 0
    tool_count = 0
    status_text = "Starting..."
    last_event_time = None
    for event in events:
        etype = event.get("type")
        last_event_time = event.get("timestamp") or last_event_time
        if etype in _SESSION_END_EVENT_TYPES:
            turn_count = 0
            tool_count = 0
            status_text = ""
            continue
        if etype == "turn_start":
            turn = event.get("turn")
            if isinstance(turn, int) and turn > turn_count:
                turn_count = turn
        elif _is_real_tool_event(event):
            tool_count += 1
        label = _progress_status_text(event)
        if label:
            status_text = label
    return {
        "turn_count": turn_count,
        "tool_count": tool_count,
        "status_text": status_text,
        "last_event_time": last_event_time,
    }


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

        # Index: (user_id, agent) -> session_id for fast interactive lookup
        self._interactive_index: dict[tuple[str, str], str] = {}

        # Index: platform_thread_id -> session_id for fast thread lookup
        self._thread_index: dict[str, str] = {}

        # Index: (channel_id, agent) -> session_id for channel session lookup
        self._channel_index: dict[tuple[str, str], str] = {}

        # Per-session compaction locks: (user_id, agent) -> Event
        # Event is unset while compaction is in progress, set when done.
        self._compaction_events: dict[tuple[str, str], threading.Event] = {}

        # Per-session set of skill names that should not be (re)loaded for the
        # rest of this session's lifetime. In-memory only; resets on daemon restart.
        self._suppressed_skills: dict[str, set[str]] = {}

        # Per-session sticky skills with TTL counters: session_id -> name -> turns_unused.
        # A trigger-matched or load_skill()-loaded skill becomes sticky so it persists
        # across user messages. The counter increments each turn the skill isn't
        # referenced; when it exceeds the skill's TTL the skill is dropped from the set.
        # In-memory only; resets on daemon restart.
        self._sticky_skills: dict[str, dict[str, int]] = {}

        self._reasoning_effort: dict[str, str] = {}
        self._model_overrides: dict[str, str] = {}

        self._load()
        self._migrate_legacy()
        self._recover_stale_sessions()

    # ── Context limit management ──

    def get_context_limit(self, agent: str) -> int:
        return self._context_limits.get(agent, 128000)

    def get_compaction_threshold(self, agent: str) -> int:
        return int(self.get_context_limit(agent) * 0.8)

    def update_context_limit(self, agent: str, limit: int) -> None:
        self._context_limits[agent] = limit

    # ── Compaction locking ──

    def begin_compaction(self, user_id: str, agent: str) -> bool:
        """Try to start compaction. Returns True if this caller should compact.

        If another caller is already compacting this session, returns False.
        """
        with self._lock:
            key = (user_id, agent)
            if key in self._compaction_events:
                return False
            self._compaction_events[key] = threading.Event()
            return True

    def end_compaction(self, user_id: str, agent: str) -> None:
        """Signal that compaction is complete. Wakes all waiters."""
        with self._lock:
            key = (user_id, agent)
            event = self._compaction_events.pop(key, None)
        if event:
            event.set()

    def wait_for_compaction(self, user_id: str, agent: str, timeout: float = 300) -> bool:
        """Block until an in-progress compaction finishes. Returns True if done, False on timeout."""
        with self._lock:
            event = self._compaction_events.get((user_id, agent))
        if event is None:
            return True
        return event.wait(timeout=timeout)

    def is_compacting(self, user_id: str, agent: str) -> bool:
        """Check if a session is currently being compacted."""
        with self._lock:
            return (user_id, agent) in self._compaction_events

    # ── Skill suppression (non-persisted) ──

    def suppress_skill(self, session_id: str, skill_name: str) -> None:
        """Mark a skill as suppressed for the given session.

        AgentPreparer will skip this skill on subsequent turns so it does not
        reload from auto_load_skills or trigger matches. Cleared on daemon
        restart by design.
        """
        with self._lock:
            self._suppressed_skills.setdefault(session_id, set()).add(skill_name)

    def unsuppress_skill(self, session_id: str, skill_name: str) -> None:
        """Remove a skill from the session's suppression set."""
        with self._lock:
            skills = self._suppressed_skills.get(session_id)
            if skills:
                skills.discard(skill_name)
                if not skills:
                    self._suppressed_skills.pop(session_id, None)

    def get_suppressed_skills(self, session_id: str) -> set[str]:
        """Return a copy of the session's suppressed skill names."""
        with self._lock:
            return set(self._suppressed_skills.get(session_id, ()))

    # ── Sticky skills (non-persisted TTL state) ──

    def mark_sticky(self, session_id: str, skill_name: str) -> None:
        """Mark a skill as sticky for the session and reset its unused-turn counter.

        Called when the skill is first trigger-matched or dynamically loaded, and
        again any time it's referenced (so the counter restarts at 0).
        """
        with self._lock:
            self._sticky_skills.setdefault(session_id, {})[skill_name] = 0

    def drop_sticky(self, session_id: str, skill_name: str) -> None:
        """Remove a skill from the sticky set."""
        with self._lock:
            bucket = self._sticky_skills.get(session_id)
            if bucket:
                bucket.pop(skill_name, None)
                if not bucket:
                    self._sticky_skills.pop(session_id, None)

    def get_sticky_skills(self, session_id: str) -> dict[str, int]:
        """Return a copy of the session's sticky skill counters."""
        with self._lock:
            return dict(self._sticky_skills.get(session_id, ()))

    def set_reasoning_effort(self, session_id: str, value: str | None) -> None:
        with self._lock:
            if value:
                self._reasoning_effort[session_id] = value
            else:
                self._reasoning_effort.pop(session_id, None)

    def get_reasoning_effort(self, session_id: str) -> str | None:
        with self._lock:
            return self._reasoning_effort.get(session_id)

    def set_model_override(self, session_id: str, value: str | None) -> None:
        with self._lock:
            if value:
                self._model_overrides[session_id] = value
            else:
                self._model_overrides.pop(session_id, None)

    def get_model_override(self, session_id: str) -> str | None:
        with self._lock:
            return self._model_overrides.get(session_id)

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
            bucket = self._sticky_skills.get(session_id)
            if not bucket:
                return
            for name in list(bucket):
                if name in referenced:
                    bucket[name] = 0
                else:
                    bucket[name] += 1

    # ── Interactive session management ──

    def get_or_create_interactive(self, user_id: str, agent: str) -> Session:
        with self._lock:
            key = (user_id, agent)
            is_replacement = False
            if key in self._interactive_index:
                session_id = self._interactive_index[key]
                if session_id in self._sessions:
                    existing = self._sessions[session_id]
                    if existing.status not in FINISHED_STATUSES:
                        return existing
                    is_replacement = True

            conv_id = f"daemon_{agent}_{user_id}_{uuid4().hex[:6]}" if is_replacement else f"daemon_{agent}_{user_id}"
            session = Session(
                id=conv_id, agent=agent, source=SessionSource.INTERACTIVE.value, user_id=user_id, title="Main Session"
            )
            tokens, msg_count = self._estimate_tokens(conv_id)
            session.cumulative_tokens = tokens
            session.message_count = msg_count
            self._sessions[conv_id] = session
            self._interactive_index[key] = conv_id
            self._save()
            return session

    def default_interactive_ids(self, agent: str) -> dict:
        """Return {user_id: session_id} for all default interactive sessions for this agent."""
        with self._lock:
            return {uid: sid for (uid, ag), sid in self._interactive_index.items() if ag == agent}

    def needs_compaction(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            threshold = self.get_compaction_threshold(session.agent)
            return session.cumulative_tokens >= threshold

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
                metadata={k: v for k, v in old_session.metadata.items() if k in READ_ONLY_METADATA_KEYS},
                scratchpad=old_session.scratchpad,
                title=old_session.title,
                pinned=old_session.pinned,
                pin_position=old_session.pin_position,
            )

            self._sessions[new_id] = new_session

            # Update indexes
            if old_session.user_id and old_session.source == SessionSource.INTERACTIVE.value:
                key = (old_session.user_id, old_session.agent)
                self._interactive_index[key] = new_id
            thread_id = new_session.metadata.get("thread_id")
            if thread_id:
                self._thread_index[thread_id] = new_id

            # Carry skill suppressions forward so a user's "remove this skill" intent
            # persists across compaction, and drop the old entry so the dict doesn't
            # accumulate orphans.
            suppressed = self._suppressed_skills.pop(session_id, None)
            if suppressed:
                self._suppressed_skills[new_id] = suppressed

            # Sticky skill counters carry forward too — compaction is "same conversation"
            # from the user's POV, so an already-loaded sticky skill should keep its TTL.
            sticky = self._sticky_skills.pop(session_id, None)
            if sticky:
                self._sticky_skills[new_id] = sticky

            effort = self._reasoning_effort.pop(session_id, None)
            if effort:
                self._reasoning_effort[new_id] = effort

            model = self._model_overrides.pop(session_id, None)
            if model:
                self._model_overrides[new_id] = model

            # Mark old session as completed and superseded so it stops appearing in
            # the default sidebar list (the new session is the live continuation).
            old_session.status = SessionStatus.COMPLETED.value
            old_session.superseded_by = new_id
            old_session.pinned = False
            old_session.pin_position = None

            self._save()
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
        carried-over context size. Real exchanges go through update_token_count.
        """
        if tokens <= 0:
            return
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
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

            if session.source == SessionSource.INTERACTIVE.value and session.user_id:
                self._interactive_index[(session.user_id, session.agent)] = session.id
            thread_id = session.metadata.get("thread_id")
            if thread_id:
                self._thread_index[thread_id] = session.id

            if session.source == SessionSource.SCHEDULE.value and session.parent_id:
                self._prune_schedule_sessions(session.parent_id)
            elif session.source in (SessionSource.BACKGROUND.value, SessionSource.SPAWNED.value):
                self._prune_background_sessions(session.agent)

            self._save()
            return session

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
                del self._sessions[s.id]

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
            del self._sessions[s.id]

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
            return session

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
        path = self._history_path(session_id)
        if not path.exists():
            return 0
        try:
            storage = SessionStorage.load(path)
        except Exception:
            return 0
        return sum(1 for _ in storage.iter_events())

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
        with what the UI would render if it replayed the event log.
        """
        events = self.read_events(session_id)
        return _progress_from_events(events)

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
        """Set multiple metadata keys. Rejects entire batch if any key is read-only."""
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
            session.last_active = datetime.now(timezone.utc).isoformat()
            self._mark_dirty()
            return session

    def delete_metadata(self, session_id: str, key: str) -> Session:
        """Delete a metadata key. Raises ValueError for read-only or missing keys."""
        if key in READ_ONLY_METADATA_KEYS:
            raise ValueError(f"Cannot delete read-only metadata key: {key}")
        with self._lock:
            if session_id not in self._sessions:
                raise ValueError(f"Session '{session_id}' not found")
            session = self._sessions[session_id]
            if key not in session.metadata:
                raise ValueError(f"Key '{key}' not found in metadata")
            del session.metadata[key]
            session.last_active = datetime.now(timezone.utc).isoformat()
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
            # Rebuild indexes
            for sid, session in self._sessions.items():
                if session.source == SessionSource.INTERACTIVE.value and session.user_id:
                    key = (session.user_id, session.agent)
                    existing_id = self._interactive_index.get(key)
                    if not existing_id or session.last_active > self._sessions[existing_id].last_active:
                        self._interactive_index[key] = sid
                thread_id = session.metadata.get("thread_id") if session.metadata else None
                if thread_id:
                    self._thread_index[thread_id] = sid
                channel_id = session.metadata.get("channel_id") if session.metadata else None
                if channel_id:
                    self._channel_index[(channel_id, session.agent)] = sid
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
                    )
                    tokens, msg_count = self._estimate_tokens(conv_id)
                    session.cumulative_tokens = tokens
                    session.message_count = msg_count
                    self._sessions[conv_id] = session
                    self._interactive_index[(user_id, agent_name)] = conv_id
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
