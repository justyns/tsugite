"""Unified session store.

Single global metadata layer for all session types (interactive, schedule,
webhook, background, spawned). Conversation data stays in JSONL history files.
"""

import json
import logging
import re
import threading
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

from tsugite.core.record_store import SqliteCollectionStorage
from tsugite.core.state import copy_state
from tsugite.history import event_to_ui_dict, generate_session_id, get_history_backend
from tsugite.renderer import parse_iso_utc as _parse_ts
from tsugite_daemon.attention_store import (
    OWNER_SESSION,
    SOURCE_DELIVERY,
    AttentionRecord,
    AttentionStore,
)
from tsugite_daemon.memory import DEFAULT_CONTEXT_LIMIT

logger = logging.getLogger(__name__)


SESSION_ID_LABEL = "session"
"""Readable label baked into generated session ids."""

SESSION_END_EVENT_TYPES = frozenset(
    {"session_complete", "session_error", "session_cancelled", "final_result", "error", "cancelled", "session_end"}
)
"""Event types that mean a run stopped.

Adding one widens the activity feed as well as the progress reset, and the frontend keeps
its own copy in `frontend/src/lib/stores/progress.ts`; mirror changes there.
"""


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
    """True for events that count toward the tool counter - broadcast tool_result
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
    if etype in SESSION_END_EVENT_TYPES:
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
        # Sandbox inheritance policy. Stamped only by spawn code (Session
        # construction); never writable via the agent-facing session_metadata tool,
        # so a sandboxed agent can't tamper with its own / a child's isolation. It's
        # in COMPACTION_PRESERVED via this set, so it also survives compaction.
        "sandbox_override",
        # Marks a placeholder session provisioned to host a /job launched outside
        # any chat. System-stamped at creation; the jobs orchestrator closes such
        # sessions when their job finishes. Read-only so an agent can't tag a real
        # chat as a job host and get it auto-closed.
        "job_host",
        # Dedupe key for a monitor's incident session. System-stamped, so an
        # agent can't hijack another monitor's incidents by writing its key.
        "incident_key",
        # The session's alias. Claimed through set_alias, which enforces uniqueness.
        "session_name",
        # Which platform DM route this session serves. Stamped at creation.
        "dm_route",
    }
)

METADATA_SESSION_NAME = "session_name"
# Where a platform's DMs land, per user. Deliberately not the alias: two people
# DMing one bot need their own sessions, and an alias has a single holder.
METADATA_DM_ROUTE = "dm_route"
METADATA_PRIMARY_FLAG = "is_primary"
METADATA_INCIDENT_KEY = "incident_key"
# Truthy on a placeholder session that exists only to host a /job spawned outside a
# conversation. The jobs orchestrator reconciles these to a terminal status when the
# job finishes so they stop rendering as active/"starting" in the sidebar.
METADATA_JOB_HOST = "job_host"

# Metadata keys preserved across compaction in addition to READ_ONLY ones (which
# already carry session_name, so an alias follows the successor session).
# is_primary makes the user's chosen primary session "follow" compaction. topic/type
# describe the conversation's subject and are carried forward like title, so a compacted
# session keeps its subject instead of resetting to blank until the next turn re-sets it.
# task/pr/notes are user-authored workstream links and freeform notes - durable by
# intent. Anything outside this set is dropped deliberately; in particular status_text
# is transient ("investigating", "idle") and must reset on compaction.
COMPACTION_PRESERVED_METADATA_KEYS = READ_ONLY_METADATA_KEYS | frozenset(
    {METADATA_PRIMARY_FLAG, "topic", "type", "task", "pr", "notes"}
)

# What a branch copies from the session it forks. Routing keys are excluded: a branch
# is a second live session, and both routes resolve to the newest holder, so inheriting
# one would silently move the address to the branch.
BRANCH_INHERITED_METADATA_KEYS = READ_ONLY_METADATA_KEYS - {METADATA_SESSION_NAME, METADATA_DM_ROUTE}

TOPIC_MAX_LENGTH = 160

ALIAS_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

ALIAS_REF_PREFIX = "name:"


def alias_from_ref(ref: str) -> Optional[str]:
    """The alias in a `name:<alias>` reference, or None if `ref` is not one."""
    if ref.startswith(ALIAS_REF_PREFIX):
        return ref[len(ALIAS_REF_PREFIX) :]
    return None


class AliasConflictError(ValueError):
    """Another routable session already holds the alias."""


def validate_alias(alias: object) -> None:
    if not isinstance(alias, str) or not ALIAS_PATTERN.fullmatch(alias):
        raise ValueError(
            f"Invalid alias {alias!r}: must start with a letter or digit and use only "
            "letters, digits, '-' and '_' (max 64 characters)"
        )


class SessionSource(str, Enum):
    INTERACTIVE = "interactive"
    SCHEDULE = "schedule"

    BACKGROUND = "background"
    SPAWNED = "spawned"

    # Origin of an interactive session, so the UI can badge it by where it was
    # created instead of collapsing every chat to 'interactive'.
    WEB = "web"
    DISCORD = "discord"
    # Written as a raw string by tsugite/agent_runner/runner.py (core can't
    # import this plugin enum), so "cli" exists in stored session data.
    CLI = "cli"


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
    # `waiting_on` is the reverse view of this.
    notify_sessions: list[str] = field(default_factory=list)
    # Created as a continuing conversation rather than a one-shot run, so it stays
    # reachable once its first turn finishes. See `accepts_followup`.
    resumable: bool = False

    title: Optional[str] = None

    pinned: bool = False
    pin_position: Optional[int] = None
    # One entry per outstanding needs-ack delivery: {id, source, title, message, timestamp}.
    pending_deliveries: list[dict] = field(default_factory=list)
    # Held until the turn ends; persisted so a daemon death mid-turn does not swallow the card.
    deferred_deliveries: list[dict] = field(default_factory=list)
    last_viewed_at: str = ""
    superseded_by: Optional[str] = None

    # Per-session UI / runtime state. Persisted so values survive daemon restart.
    sticky_skills: dict[str, int] = field(default_factory=dict)
    suppressed_skills: list[str] = field(default_factory=list)
    reasoning_effort: Optional[str] = None
    model_override: Optional[str] = None
    # Per-session working directory override (used by Jobs feature so a worker
    # session runs inside its provisioned git worktree, not the adapter's default).
    workspace_override: Optional[str] = None
    compacting: bool = False
    # Durably True while a turn is executing (set/cleared by the adapter around
    # handle_message). A daemon death mid-turn leaves it set, so the next boot
    # can finalize the orphaned turn (see _recover_stale_sessions).
    turn_in_flight: bool = False
    # Provider-reported context window for this session's model. None until the
    # first turn reports it; consumers fall back to the daemon-wide default via
    # SessionStore.get_session_context_limit. Per-session so a compact-model
    # call (or any other secondary-model side effect) can't clobber the value
    # other sessions are reading.
    context_limit: Optional[int] = None

    @property
    def is_primary(self) -> bool:
        return bool(self.metadata.get(METADATA_PRIMARY_FLAG))

    @property
    def alias(self) -> Optional[str]:
        """This session's routing identity, claimed through SessionStore.set_alias."""
        return self.metadata.get(METADATA_SESSION_NAME)

    @property
    def accepts_followup(self) -> bool:
        """Finished, but still open to another turn. A failed or cancelled run needs
        an explicit restart instead."""
        return self.resumable and self.status == SessionStatus.COMPLETED.value

    @property
    def has_pending_deliveries(self) -> bool:
        return bool(self.pending_deliveries)

    @property
    def pending_delivery_ids(self) -> list[str]:
        return [d["id"] for d in self.pending_deliveries]

    @property
    def has_live_work(self) -> bool:
        """Whether this session currently has work running.

        Two signals, because the two kinds of run are marked differently:
        interactive turns set `turn_in_flight` for the turn's duration, while
        scheduled and background runs are created with status RUNNING and never
        begin a turn at all.

        The durable half of "is this session busy". The HTTP layer's
        `_session_busy` builds on this, adding the live chat task that only it
        can see. Anything that needs to know whether a session is working
        should ask this rather than infer it from progress fields - a progress
        label reports the last event, not what is happening now.
        """
        return self.turn_in_flight or self.status == SessionStatus.RUNNING.value

    def __post_init__(self):
        if not self.id:
            self.id = f"session-{uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.last_active:
            self.last_active = now


PENDING_DELIVERY_MESSAGE_CHARS = 200


def attention_fields(records: list[AttentionRecord]) -> dict:
    """The `needs_attention` + `attention` pair on a session payload."""
    return {"needs_attention": bool(records), "attention": [asdict(r) for r in records]}


def render_pending_deliveries_xml(session: Session) -> str:
    if not session.pending_deliveries:
        return ""

    from tsugite.prompt_xml import El

    deliveries = []
    for item in session.pending_deliveries:
        message = item.get("message") or ""
        if len(message) > PENDING_DELIVERY_MESSAGE_CHARS:
            message = message[:PENDING_DELIVERY_MESSAGE_CHARS] + "…"
        deliveries.append(
            El(
                "delivery",
                [message],
                {
                    "id": item["id"],
                    "source": item.get("source") or "",
                    "title": item.get("title") or None,
                    "at": item.get("timestamp") or None,
                },
                inline=True,
            )
        )
    return El("pending_deliveries", deliveries).render(indent="  ", level=1)


class SessionStore:
    """Global unified session metadata store.

    One instance shared across all agents and adapters.
    Persists write-through to {state_dir}/daemon.db (sessions collection).
    """

    def __init__(
        self,
        store_path: Path,
        default_context_limit: int = DEFAULT_CONTEXT_LIMIT,
    ):
        self._path = store_path  # legacy JSON location; one-time migration source
        self._storage = SqliteCollectionStorage.for_state_file(store_path, "sessions")
        self.attention = AttentionStore(store_path)
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

        # Default context limit for compaction; sessions that report their own
        # window override it via Session.context_limit.
        self._default_context_limit = default_context_limit

        # Index: platform_thread_id -> session_id for fast thread lookup
        self._thread_index: dict[str, str] = {}

        # Index: channel_id -> session_id for channel session lookup
        self._channel_index: dict[str, str] = {}

        # Per-user compaction synchronization. Event is unset while compaction is
        # in progress, set when done. Per-session compacting state is stored on
        # Session itself (Session.compacting); this map only gates concurrent
        # begin_compaction calls.
        self._compaction_events: dict[str, threading.Event] = {}

        # Fired with the session id after each turn ends, outside the lock.
        self._on_turn_end: Optional[Callable[[str], None]] = None

        # Hot caches keyed by session_id, populated lazily on first read and
        # then updated incrementally inside `append_event`. Without these,
        # `session_progress_summary` and `event_count` would re-parse the
        # full .jsonl on every sidebar refresh - at 800+ sessions and
        # multi-MB active session files that's tens of MB of file I/O per
        # SSE-driven update.
        self._progress_cache: dict[str, dict] = {}
        self._event_count_cache: dict[str, int] = {}
        self._cache_lock = threading.Lock()

        # One conditional boot snapshot keeps the db equal to memory after
        # load-time reconciliation (legacy imports, primary stamping, stale
        # recovery) without rewriting the whole table on a no-op boot.
        changed = self._load()
        changed |= self._migrate_legacy()
        changed |= self._recover_stale_sessions()
        if changed:
            self._save()

    # ── Context limit management ──

    def get_context_limit(self) -> int:
        return self._default_context_limit

    def get_compaction_threshold(self) -> int:
        return int(self.get_context_limit() * 0.8)

    def update_context_limit(self, limit: int) -> None:
        self._default_context_limit = limit

    def get_session_context_limit(self, session_id: str) -> int:
        """Return the session's tracked context window, falling back to the
        daemon-wide default when the session hasn't completed a turn yet.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return DEFAULT_CONTEXT_LIMIT
            if session.context_limit is not None:
                return session.context_limit
            return self._default_context_limit

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
            self._persist(session)

    def get_session_compaction_threshold(self, session_id: str) -> int:
        return int(self.get_session_context_limit(session_id) * 0.8)

    # ── Compaction locking ──

    def begin_compaction(self, user_id: str, session_id: str | None = None) -> bool:
        """Try to start compaction. Returns True if this caller should compact.

        If another caller is already compacting this session, returns False.
        Pass session_id so per-session UI state (e.g. the "compacting…" chip)
        is also scoped via Session.compacting.
        """
        with self._lock:
            key = user_id
            if key in self._compaction_events:
                return False
            self._compaction_events[key] = threading.Event()
            if session_id:
                session = self._sessions.get(session_id)
                if session:
                    session.compacting = True
                    self._persist(session)
            return True

    def end_compaction(self, user_id: str, session_id: str | None = None) -> None:
        """Signal that compaction is complete. Wakes all waiters."""
        with self._lock:
            key = user_id
            event = self._compaction_events.pop(key, None)
            if session_id:
                session = self._sessions.get(session_id)
                if session:
                    session.compacting = False
                    self._persist(session)
        if event:
            event.set()

    def wait_for_compaction(self, user_id: str, timeout: float = 300) -> bool:
        """Block until an in-progress compaction finishes. Returns True if done, False on timeout."""
        with self._lock:
            event = self._compaction_events.get(user_id)
        if event is None:
            return True
        return event.wait(timeout=timeout)

    def is_compacting(self, user_id: str, session_id: str | None = None) -> bool:
        """Check if a session is currently being compacted.

        With session_id: per-session answer from Session.compacting.
        Without: per-user lock state.
        """
        with self._lock:
            if session_id is not None:
                session = self._sessions.get(session_id)
                return bool(session and session.compacting)
            return user_id in self._compaction_events

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
                self._persist(session)

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
                self._persist(session)

    def drop_sticky(self, session_id: str, skill_name: str) -> None:
        """Remove a skill from the sticky set."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session and skill_name in session.sticky_skills:
                del session.sticky_skills[skill_name]
                self._persist(session)

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
                self._persist(session)

    def get_reasoning_effort(self, session_id: str) -> str | None:
        with self._lock:
            session = self._sessions.get(session_id)
            return session.reasoning_effort if session else None

    def freeze_session_models_to_current(self, current_model: str | None) -> None:
        """Pin every active, non-superseded session without a `model_override`
        to `current_model`. Used when the daemon default model is about to
        change, so existing sessions stay on whatever model they were resolving
        to instead of silently switching on their next turn.

        No-op when `current_model` is falsy (nothing to pin to).
        """
        if not current_model:
            return
        with self._lock:
            changed = []
            for session in self._sessions.values():
                if session.status in FINISHED_STATUSES:
                    continue
                if session.superseded_by:
                    continue
                if session.model_override:
                    continue
                session.model_override = current_model
                changed.append(session)
            self._persist(*changed)

    def set_model_override(self, session_id: str, value: str | None) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.model_override = value or None
                self._persist(session)

    def get_model_override(self, session_id: str) -> str | None:
        with self._lock:
            session = self._sessions.get(session_id)
            return session.model_override if session else None

    def bump_unused_counters(self, session_id: str, referenced: set[str]) -> None:
        """Advance one turn: reset referenced skills, increment the rest.

        A skill is "referenced" this turn if the agent called load_skill() on it
        or the scanner found its name/trigger in the user message or final answer.
        Unreferenced skills get their counter incremented. Callers decide whether
        to drop skills whose counter now exceeds their TTL (we don't know per-skill
        TTL here - that's a frontmatter/config concern).
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
            self._persist(session)

    # ── Interactive session management ──

    def get_or_create_interactive(self, user_id: str, source: str = SessionSource.INTERACTIVE.value) -> Session:
        """Return the user's primary session, or create a fresh default one.

        `source` stamps only newly created sessions; an existing primary keeps
        the source it was created with.
        """
        return self.find_default_session(user_id) or self.create_default_session(user_id, source=source)

    def _live_sessions_locked(self, user_id: Optional[str] = None):
        """Every session that is still routable: not finished, not superseded,
        optionally scoped to one user. Caller must hold self._lock."""
        for s in self._sessions.values():
            if user_id is not None and s.user_id != user_id:
                continue
            if s.superseded_by is not None or s.status in FINISHED_STATUSES:
                continue
            yield s

    @staticmethod
    def _latest(sessions) -> Optional[Session]:
        return max(sessions, key=lambda s: s.last_active, default=None)

    def default_primary_ids(self) -> dict:
        """Return {user_id: session_id} for all primary sessions."""
        with self._lock:
            return {s.user_id: s.id for s in self._live_sessions_locked() if s.user_id and s.is_primary}

    def _find_named_session_locked(self, name: str) -> Optional[Session]:
        return self._latest(s for s in self._live_sessions_locked() if s.metadata.get(METADATA_SESSION_NAME) == name)

    def find_named_session(self, name: str) -> Optional[Session]:
        """The routable session holding `name`, or None.

        One holder for the whole daemon. One person reaches it as several user_ids
        (web-anonymous, a canonical name, a Discord id, a bare scheduler user), so a
        per-user alias would be invisible to the surfaces that did not set it.
        """
        with self._lock:
            return self._find_named_session_locked(name)

    def set_alias(self, session_id: str, alias: str) -> Session:
        """Claim `alias` as this session's routing identity.

        Raises ValueError for a malformed alias, AliasConflictError for one another
        routable session already holds. Passing a new alias renames, releasing the old
        one in the same lock.
        """
        validate_alias(alias)
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError(f"Session '{session_id}' not found")
            holder = next(
                (s for s in self._live_sessions_locked() if s.alias == alias and s.id != session_id),
                None,
            )
            if holder is not None:
                raise AliasConflictError(f"Alias '{alias}' is already held by session '{holder.id}'")
            session.metadata[METADATA_SESSION_NAME] = alias
            self._persist(session)
            return session

    def clear_alias(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError(f"Session '{session_id}' not found")
            session.metadata.pop(METADATA_SESSION_NAME, None)
            self._persist(session)
            return session

    def find_incident_session(self, user_id: str, incident_key: str) -> Optional[Session]:
        with self._lock:
            return self._latest(
                s for s in self._live_sessions_locked(user_id) if s.metadata.get(METADATA_INCIDENT_KEY) == incident_key
            )

    def _find_primary_session_locked(self, user_id: str) -> Optional[Session]:
        """Lock-held variant of find_primary_session. Caller must hold self._lock."""
        return self._latest(s for s in self._live_sessions_locked(user_id) if s.is_primary)

    def find_primary_session(self, user_id: str) -> Optional[Session]:
        """Return the user's primary session, or None."""
        with self._lock:
            return self._find_primary_session_locked(user_id)

    def find_default_session(self, user_id: str) -> Optional[Session]:
        """Canonical lookup for where a default request should land."""
        return self.find_primary_session(user_id)

    def _demote_primaries_locked(self, user_id: str, *, except_id: Optional[str] = None) -> Optional[Session]:
        """Clear primary flag from all of the user's sessions except `except_id`. Returns the last one cleared.
        Persists every demoted session itself, so callers only persist their own target."""
        cleared: Optional[Session] = None
        for s in self._sessions.values():
            if s.user_id == user_id and s.id != except_id and s.is_primary:
                s.metadata.pop(METADATA_PRIMARY_FLAG, None)
                self._persist(s)
                cleared = s
        return cleared

    def create_default_session(
        self,
        user_id: str,
        *,
        title: Optional[str] = None,
        source: str = SessionSource.INTERACTIVE.value,
    ) -> Session:
        """Create a fresh interactive session and mark it primary."""
        with self._lock:
            conv_id = generate_session_id(SESSION_ID_LABEL)
            self._demote_primaries_locked(user_id)
            session = Session(
                id=conv_id,
                source=source,
                user_id=user_id,
                title=title,
                metadata={METADATA_PRIMARY_FLAG: True},
            )
            self._sessions[conv_id] = session
            self._persist(session)
            return session

    def set_primary_session(self, session_id: str) -> Session:
        """Mark `session_id` as primary, demoting any prior primary for the same user."""
        with self._lock:
            if session_id not in self._sessions:
                raise ValueError(f"Session '{session_id}' not found")
            target = self._sessions[session_id]
            if target.status in FINISHED_STATUSES:
                raise ValueError(f"Cannot promote finished session '{session_id}' to primary")
            if target.superseded_by:
                raise ValueError(f"Cannot promote superseded session '{session_id}' to primary")
            self._demote_primaries_locked(target.user_id, except_id=target.id)
            target.metadata[METADATA_PRIMARY_FLAG] = True
            self._persist(target)
            return target

    def clear_primary_session(self, user_id: str) -> Optional[Session]:
        """Remove the primary flag from any of the user's sessions. Returns the cleared session, if any."""
        with self._lock:
            return self._demote_primaries_locked(user_id)

    def claim_aliased_session(
        self, alias: str, user_id: Optional[str] = None, source: str = SessionSource.INTERACTIVE.value
    ) -> Session:
        """The session holding `alias`, creating one if none does.

        The alias lives in metadata and is preserved across compaction, so it follows
        the successor session automatically.
        """
        validate_alias(alias)
        with self._lock:
            existing = self._find_named_session_locked(alias)
            if existing:
                return existing
            return self._create_routed_locked(key=METADATA_SESSION_NAME, value=alias, user_id=user_id, source=source)

    def get_or_create_dm_session(
        self, user_id: str, route: str, source: str = SessionSource.INTERACTIVE.value
    ) -> Session:
        """The session a platform's DMs from `user_id` land in, creating one if absent."""
        with self._lock:
            existing = self._latest(
                s for s in self._live_sessions_locked(user_id) if s.metadata.get(METADATA_DM_ROUTE) == route
            )
            if existing:
                return existing
            return self._create_routed_locked(key=METADATA_DM_ROUTE, value=route, user_id=user_id, source=source)

    def _create_routed_locked(self, *, key: str, value: str, user_id: Optional[str], source: str) -> Session:
        conv_id = f"daemon_{user_id or 'shared'}_{value}_{uuid4().hex[:6]}"
        session = Session(
            id=conv_id, source=source, user_id=user_id, title=f"{value.title()} Session", metadata={key: value}
        )
        session.cumulative_tokens, session.message_count = self._estimate_tokens(conv_id)
        self._sessions[conv_id] = session
        self._persist(session)
        return session

    def needs_compaction(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if not session:
            return False
        return session.cumulative_tokens >= self.get_session_compaction_threshold(session_id)

    def branch_session(self, session_id: str, at_event_id: int, label: Optional[str] = None) -> Session:
        """Fork ``session_id`` at ``at_event_id`` into an independent branch session.

        The source session is unchanged (not superseded). The branch gets the forked
        history (provider state scrubbed) plus its own sidebar entry mirroring the
        source's user/runtime settings.
        """
        with self._lock:
            source = self._sessions.get(session_id)
            if not source:
                raise ValueError(f"Session '{session_id}' not found")
            src = source.source
            user_id = source.user_id
            title = source.title
            preserved = {k: v for k, v in source.metadata.items() if k in BRANCH_INHERITED_METADATA_KEYS}
            model_override = source.model_override
            reasoning_effort = source.reasoning_effort
            workspace_override = source.workspace_override
            context_limit = source.context_limit

        new_id = get_history_backend().create_branch(session_id, at_event_id=at_event_id)

        branch = Session(
            id=new_id,
            source=src,
            user_id=user_id,
            metadata={**preserved, "branched_from": session_id},
            title=label or (f"Branch of {title}" if title else f"Branch of {session_id}"),
            model_override=model_override,
            reasoning_effort=reasoning_effort,
            workspace_override=workspace_override,
            context_limit=context_limit,
        )
        with self._lock:
            self._sessions[new_id] = branch
            self._persist(branch)
        return branch

    def compact_session(self, session_id: str) -> Session:
        with self._lock:
            old_session = self._sessions.get(session_id)
            if not old_session:
                raise ValueError(f"Session '{session_id}' not found")
            if old_session.superseded_by:
                # Re-compacting a superseded session (a stale tab that missed the
                # rotation) would overwrite superseded_by and fork the chain into
                # two live successors. Callers should resolve the successor first.
                raise ValueError(f"Session '{session_id}' was already compacted into '{old_session.superseded_by}'")

            new_id = generate_session_id(SESSION_ID_LABEL)
            new_session = Session(
                id=new_id,
                source=old_session.source,
                user_id=old_session.user_id,
                parent_id=old_session.parent_id,
                metadata={k: v for k, v in old_session.metadata.items() if k in COMPACTION_PRESERVED_METADATA_KEYS},
                title=old_session.title,
                pinned=old_session.pinned,
                pin_position=old_session.pin_position,
                # Carry per-session UI/runtime state forward - compaction is "same
                # conversation" from the user's POV, so suppressions, sticky-skill
                # TTL counters, and effort/model overrides should follow the rotation.
                sticky_skills=dict(old_session.sticky_skills),
                suppressed_skills=list(old_session.suppressed_skills),
                reasoning_effort=old_session.reasoning_effort,
                model_override=old_session.model_override,
                workspace_override=old_session.workspace_override,
                context_limit=old_session.context_limit,
                pending_deliveries=list(old_session.pending_deliveries),
                deferred_deliveries=list(old_session.deferred_deliveries),
                notify_sessions=list(old_session.notify_sessions),
            )
            # Preserve original conversation start so <session_started> in the
            # message context reflects the user's perceived session age, not
            # the compaction moment.
            new_session.created_at = old_session.created_at

            self._sessions[new_id] = new_session
            copy_state(session_id, new_id)

            # Compaction preserves is_primary metadata, so the new session automatically
            # becomes the user's default if the predecessor was. The named-route lookup
            # indexes must follow the successor too, otherwise the next thread/channel
            # message finds the now-completed predecessor and forks a fresh empty session,
            # abandoning the compacted history.
            thread_id = new_session.metadata.get("thread_id")
            if thread_id:
                self._thread_index[thread_id] = new_id
            channel_id = new_session.metadata.get("channel_id")
            if channel_id:
                self._channel_index[channel_id] = new_id

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
            old_session.pending_deliveries = []
            # The cards moved to the successor, so the obligations they carry move too.
            for record in self.attention.clear_owner(session_id, source=SOURCE_DELIVERY):
                self.attention.open(
                    owner_kind=OWNER_SESSION,
                    owner_id=new_id,
                    source=record.source,
                    ref_id=record.ref_id,
                    kind=record.kind,
                )
            old_session.notify_sessions = []
            old_session.deferred_deliveries = []
            # The successor owns any in-flight turn now; a stale marker on the
            # superseded session would trigger a spurious boot-time repair.
            old_session.turn_in_flight = False

            self._persist(old_session, new_session)
        self._evict_progress_cache(session_id)
        return new_session

    def _live_end_locked(self, session: Session) -> Optional[Session]:
        """Tail of `session`'s compaction chain, or None if it dead-ends."""
        seen = set()
        while session.superseded_by and session.id not in seen:
            seen.add(session.id)
            session = self._sessions.get(session.superseded_by)
            if session is None:
                return None
        return session

    def resolve_compacted_successor(self, session_id: str) -> Optional[Session]:
        """Return the LIVE end of `session_id`'s compaction chain, or None.

        Walks `superseded_by` links to the tail: a tab that missed SSE updates
        can hold a session that was compacted more than once, and stopping one
        hop in lands on another superseded/completed session (which the chat
        endpoint then rejects). Returns None when `session_id` is unknown, has
        no successor, or the chain dead-ends on a pruned session.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session or not session.superseded_by:
                return None
            return self._live_end_locked(session)

    def resolve_live(self, session_id: str) -> Optional[Session]:
        """The live session `session_id` refers to: itself when it was never
        compacted, otherwise the tail of its chain. None when the id is unknown
        or the chain dead-ends on a pruned session."""
        with self._lock:
            session = self._sessions.get(session_id)
            return self._live_end_locked(session) if session else None

    def update_token_count(self, session_id: str, tokens_used: int) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                if tokens_used > 0:
                    session.cumulative_tokens = tokens_used
                session.message_count += 1
                session.last_active = datetime.now(timezone.utc).isoformat()
                self._persist(session)

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
                self._persist(session)

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
                self._prune_background_sessions()

            self._persist(session)
            return session

    def delete_session(self, session_id: str) -> bool:
        """Remove a session. Returns whether there was one to remove."""
        with self._lock:
            if session_id not in self._sessions:
                return False
            self._purge_session_state(session_id)
            return True

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
        self._storage.delete(session_id)
        self.attention.clear_owner(session_id)
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

    def _prune_background_sessions(self) -> None:
        """Remove least-recently-active finished background/spawned sessions beyond
        MAX_BACKGROUND_SESSIONS. Job workers churn the shared cap, so a session still
        being replied to outlives an older idle one. Must hold lock."""
        children = [
            s
            for s in self._sessions.values()
            if s.source in (SessionSource.BACKGROUND.value, SessionSource.SPAWNED.value)
            and s.status in FINISHED_STATUSES
        ]
        if len(children) <= self.MAX_BACKGROUND_SESSIONS:
            return
        children.sort(key=lambda s: s.last_active)
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
            self._persist(session)
        if session.status in FINISHED_STATUSES:
            self._evict_progress_cache(session_id)
        return session

    def _evict_progress_cache(self, session_id: str) -> None:
        """Drop a session's live-progress entry once it stops appending events.

        Sidebar refreshes only call `session_progress_summary` for sessions in
        live statuses, so finished sessions never re-read the cache; keeping
        them resident grows memory without bound across daemon uptime. The
        event_count entry stays - it's still hit by `session_detail`.
        """
        with self._cache_lock:
            self._progress_cache.pop(session_id, None)

    def _pinned_sessions(self, exclude_id: Optional[str] = None) -> list[Session]:
        """Return pinned sessions, sorted by current pin_position (None last)."""
        return sorted(
            [s for s in self._sessions.values() if s.pinned and s.id != exclude_id],
            key=lambda s: s.pin_position if s.pin_position is not None else 0,
        )

    def set_pin(self, session_id: str, pinned: bool, position: Optional[int] = None) -> Session:
        """Pin or unpin a session. Pinning appends to the end unless position is given;
        unpinning densifies the remaining pinned sessions.
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
                for i, s in enumerate(self._pinned_sessions()):
                    s.pin_position = i
            else:
                others = self._pinned_sessions(exclude_id=session_id)
                insert_at = len(others) if position is None else max(0, min(position, len(others)))
                session.pin_position = insert_at
                for i, s in enumerate(others):
                    s.pin_position = i if i < insert_at else i + 1

            session.last_active = datetime.now(timezone.utc).isoformat()
            # Pin/unpin repositions sibling pins too - persist the whole pin set.
            self._persist(session, *self._pinned_sessions(exclude_id=session_id))
            return session

    def reorder_pins(self, ordered_ids: list[str]) -> list[Session]:
        """Write pin_position 0..N-1 for the given pinned session ids; unknown or
        unpinned ids are silently skipped.
        """
        with self._lock:
            valid = [self._sessions[sid] for sid in ordered_ids if sid in self._sessions and self._sessions[sid].pinned]
            for i, s in enumerate(valid):
                s.pin_position = i
            self._persist(*valid)
            return valid

    def mark_viewed(self, session_id: str, ts: Optional[str] = None) -> Session:
        """Set last_viewed_at on a session. Defaults to now (UTC ISO)."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise ValueError(f"Session '{session_id}' not found")
            session.last_viewed_at = ts or datetime.now(timezone.utc).isoformat()
            self._persist(session)
            return session

    def hold_delivery(self, session_id: str, event: dict) -> bool:
        """Hold a delivery when a turn is in flight, reporting whether it was held.

        The check and the hold share the lock: deciding outside it lets a turn end
        in between, leaving the card behind a flush that has already run.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session or not session.turn_in_flight:
                return False
            session.deferred_deliveries.append(event)
            self._persist(session)
            return True

    def take_deferred_deliveries(self, session_id: str) -> list[dict]:
        """In arrival order."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session or not session.deferred_deliveries:
                return []
            held = session.deferred_deliveries
            session.deferred_deliveries = []
            self._persist(session)
            return held

    def sessions_holding_deliveries(self) -> list[str]:
        with self._lock:
            return [s.id for s in self._sessions.values() if s.deferred_deliveries]

    def record_delivery(self, session_id: str, event: dict, *, needs_ack: bool) -> Session:
        """Append a delivery event and bump last_active, which is what `unread` derives from."""
        self.append_event(session_id, event)
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise ValueError(f"Session '{session_id}' not found")
            if needs_ack:
                session.pending_deliveries.append(
                    {
                        "id": event["delivery_id"],
                        "source": event.get("source") or "",
                        "title": event.get("title"),
                        "message": event.get("message") or "",
                        "timestamp": event.get("timestamp") or "",
                    }
                )
                self.attention.open(
                    owner_kind=OWNER_SESSION,
                    owner_id=session_id,
                    source=SOURCE_DELIVERY,
                    ref_id=event["delivery_id"],
                    kind="needs_ack",
                )
            session.last_active = datetime.now(timezone.utc).isoformat()
            self._persist(session)
        return session

    def clear_attention(self, session_id: str, delivery_id: Optional[str] = None) -> Session:
        """Discharge one obligation, or every one when `delivery_id` is None.

        Does not bump last_active: that would re-mark the session unread the
        moment it is answered.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise ValueError(f"Session '{session_id}' not found")
            if delivery_id is None:
                session.pending_deliveries = []
                self.attention.clear_owner(session_id, source=SOURCE_DELIVERY)
            else:
                session.pending_deliveries = [d for d in session.pending_deliveries if d.get("id") != delivery_id]
                self.attention.clear_ref(SOURCE_DELIVERY, delivery_id)
            self._persist(session)
            return session

    def add_notify_session(self, session_id: str, target_id: str) -> Session:
        """Register `target_id` as a session to notify when `session_id` finishes.

        Bookkeeping, not conversation activity, so it leaves last_active alone.
        Rejects a target belonging to someone else: the notification starts a turn
        in that session, carrying this one's title and result into it.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise ValueError(f"Session '{session_id}' not found")
            target = self._sessions.get(target_id)
            if not target:
                raise ValueError(f"Notify target '{target_id}' not found")
            # A background worker carries no user_id and inherits its spawner's
            # audience, so only two attributed sessions can disagree.
            if session.user_id and target.user_id and target.user_id != session.user_id:
                raise ValueError(f"Notify target '{target_id}' belongs to another user")
            if target_id not in session.notify_sessions:
                session.notify_sessions.append(target_id)
                self._persist(session)
            return session

    def open_attention_by_owner(self) -> dict[str, list[AttentionRecord]]:
        """Grouped so a session list resolves every row in one pass."""
        grouped: dict[str, list[AttentionRecord]] = {}
        for record in self.attention.open_records():
            grouped.setdefault(record.owner_id, []).append(record)
        return grouped

    def waiting_on_map(self) -> dict[str, list[str]]:
        """Map a session id to the unfinished sessions that will notify it.

        Derived on read: a notifier drops out the moment it finishes.
        """
        waiting: dict[str, list[str]] = {}
        with self._lock:
            for session in self._sessions.values():
                if session.status in FINISHED_STATUSES:
                    continue
                for target_id in session.notify_sessions:
                    if target_id != session.id:
                        waiting.setdefault(target_id, []).append(session.id)
        return waiting

    def list_sessions(
        self,
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
                if (not source or s.source == source)
                and (not parent_id or s.parent_id == parent_id)
                and (not status or s.status == status)
                and (not user_id or s.user_id == user_id)
                and (not _updated_since_dt or (_parse_ts(s.last_active) or _epoch) >= _updated_since_dt)
                and (include_superseded or not s.superseded_by)
            ]
            if limit:
                results.sort(key=lambda s: s.last_active or s.created_at, reverse=True)
                # Pins and open obligations must outlive the recency window; the
                # limit keeps bounding the rest of the tail.
                head, tail = results[:limit], results[limit:]
                if tail:
                    head_ids = {s.id for s in head}
                    waiting = {r.owner_id for r in self.attention.open_records()}
                    head.extend(s for s in tail if (s.pinned or s.id in waiting) and s.id not in head_ids)
                results = head
            return results

    def search_sessions(self, q: str, limit: int = 50) -> list[Session]:
        """Case-insensitive substring search over ALL sessions - title, id,
        prompt, and string metadata values (topic etc.) - ignoring the recency
        window that bounds list_sessions. A hit on a superseded session
        resolves to its live successor so a compacted conversation stays
        findable by its pre-compaction content."""
        needle = q.strip().lower()
        if not needle:
            return []
        with self._lock:
            hits = []
            for s in self._sessions.values():
                meta_text = " ".join(str(v) for v in (s.metadata or {}).values() if isinstance(v, (str, int, float)))
                text = " ".join(filter(None, [s.title, s.id, s.prompt, meta_text])).lower()
                if needle in text:
                    hits.append(s)
            resolved: dict[str, Session] = {}
            for s in hits:
                seen: set[str] = set()
                while s.superseded_by and s.superseded_by in self._sessions and s.id not in seen:
                    seen.add(s.id)
                    s = self._sessions[s.superseded_by]
                resolved[s.id] = s
            results = list(resolved.values())
            results.sort(key=lambda s: s.last_active or s.created_at, reverse=True)
            return results[:limit]

    def list_interactive(self) -> list[Session]:
        """Return all active interactive sessions."""
        with self._lock:
            return [
                s
                for s in self._sessions.values()
                if s.source == SessionSource.INTERACTIVE.value and s.status == SessionStatus.ACTIVE.value
            ]

    # ── Event log: unified with conversation history ──
    #
    # UI events (prompt_snapshots, hook executions, etc.) are recorded into the
    # same history session as the conversation events, through the history backend.

    def append_event(self, session_id: str, event: dict) -> None:
        """Append a UI/telemetry event to the session's history JSONL.

        Accepts the legacy flat-dict shape `{type, timestamp, ...rest}` and
        translates it into the per-event Event schema. Creates the file if it
        doesn't exist (without injecting an implicit session_start) so that
        callers like the SSE handler don't have to coordinate file creation
        with the agent loop.
        """
        backend = get_history_backend()
        ts = _parse_ts(event.get("timestamp"))
        data = {k: v for k, v in event.items() if k not in ("type", "timestamp")}
        backend.ensure_session(session_id).record(event.get("type", "unknown"), ts=ts, **data)

        # Update hot caches incrementally. Skip when the session_id has never
        # been read - the cold-load path will populate everything in one go.
        with self._cache_lock:
            if session_id in self._event_count_cache:
                self._event_count_cache[session_id] += 1
            progress = self._progress_cache.get(session_id)
            if progress is not None:
                _apply_event_to_progress(progress, event)

    def read_events(self, session_id: str) -> list[dict]:
        """Return events as flat UI dicts (type/timestamp + data) for callers."""
        backend = get_history_backend()
        if not backend.exists(session_id):
            return []
        try:
            session = backend.load(session_id)
        except Exception:
            return []
        return [event_to_ui_dict(e) for e in session.iter_events()]

    def read_events_page(
        self,
        session_id: str,
        *,
        after_id: Optional[int] = None,
        before_id: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> dict:
        """Windowed / delta read of a session's UI events for the chat surface.

        - ``after_id``: forward delta - only events with ``id`` greater than it
          (the incremental resync path). A forward catch-up, so ``has_more`` is
          always False.
        - ``limit`` alone: the newest ``limit`` events (the tail window on open).
        - ``before_id`` + ``limit``: the newest ``limit`` events with ``id`` below
          the cursor (the "load earlier" page).
        - none of the above: every event.

        Returns ``{"events": [...chronological...], "has_more": bool,
        "oldest_id": int | None}``. ``has_more`` is True when older events exist
        before the window; ``oldest_id`` is the window's smallest id (the cursor
        for the next earlier page), or None when the backend assigns no ids (the
        deprecated jsonl backend) or the window is empty. Only the returned window
        is normalized to a UI dict, so a fat log doesn't pay the model-response
        backfill parse for events the client will never render.
        """
        backend = get_history_backend()
        if not backend.exists(session_id):
            return {"events": [], "has_more": False, "oldest_id": None}
        try:
            session = backend.load(session_id)
            # Each Session windows in its own layer: SQLite bounds it in the query, the
            # jsonl backend slices its materialized log. A negative limit can't be a SQL
            # LIMIT, but the HTTP endpoint only ever passes None or a non-negative int.
            events, has_more = session.read_events_window(after_id=after_id, before_id=before_id, limit=limit)
        except Exception:
            return {"events": [], "has_more": False, "oldest_id": None}

        return {
            "events": [event_to_ui_dict(e) for e in events],
            "has_more": has_more,
            "oldest_id": events[0].id if events else None,
        }

    def event_count(self, session_id: str) -> int:
        with self._cache_lock:
            cached = self._event_count_cache.get(session_id)
        if cached is not None:
            return cached
        count = get_history_backend().count_events(session_id)
        with self._cache_lock:
            self._event_count_cache[session_id] = count
        return count

    def count_events_by_type(self, session_id: str, event_type: str) -> int:
        return get_history_backend().count_events(session_id, type=event_type)

    def session_detail(self, session_id: str) -> dict:
        session = self.get_session(session_id)
        result = asdict(session)
        result["event_count"] = self.event_count(session_id)
        result["is_primary"] = session.is_primary
        result["alias"] = session.alias
        result.update(attention_fields(self.attention.open_records(session_id)))
        # Same key, same shape as the session-list row: ids, not whole cards.
        result["pending_deliveries"] = session.pending_delivery_ids
        # `context_limit` (the raw dataclass field) is None until the first turn
        # reports a provider window, so a fresh session would get no meter. Expose
        # the RESOLVED limit alongside it (falls back to the agent default) so the
        # web UI can paint `0 / <default>` from session open. Additive: the raw
        # field keeps its "unset == None" meaning.
        result["context_limit_resolved"] = self.get_session_context_limit(session_id)
        return result

    def session_progress_summary(self, session_id: str) -> dict:
        """Return a lightweight live-progress summary for a running session.

        Fields are derived entirely from events.jsonl so they stay consistent
        with what the UI would render if it replayed the event log. The first
        call cold-loads the file; subsequent calls hit the in-memory cache,
        which `append_event` keeps current.

        `status_text` is the last status-bearing event, which is the session's
        *current* status only while the log ends on a terminator. A log can end
        mid-turn with nothing running - compaction retains a slice of turns and
        deliberately drops their `session_end` markers, and a crash or truncated
        write leaves the same shape. Callers rendering the label must therefore
        gate it on `has_live_work` (or the HTTP layer's broader `_session_busy`),
        or an idle session displays a stale "Waiting on LLM..." forever. This
        stays a pure derivation from events; liveness is not an event.

        Callers MUST treat the returned dict as read-only - it's the cached
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

    # ── Metadata CRUD ──

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
            self._persist(session)
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
            self._persist(session)
            return session

    # ── Channel session index ──

    def get_or_create_channel_session(
        self, channel_id: str, user_id: str, source: str = SessionSource.INTERACTIVE.value
    ) -> Session:
        with self._lock:
            is_replacement = False
            if channel_id in self._channel_index:
                session_id = self._channel_index[channel_id]
                if session_id in self._sessions:
                    existing = self._sessions[session_id]
                    if existing.status not in FINISHED_STATUSES:
                        return existing
                    is_replacement = True

            conv_id = f"channel_{channel_id}_{uuid4().hex[:6]}" if is_replacement else f"channel_{channel_id}"
            session = Session(
                id=conv_id,
                source=source,
                user_id=user_id,
                metadata={"channel_id": channel_id},
            )
            tokens, msg_count = self._estimate_tokens(conv_id)
            session.cumulative_tokens = tokens
            session.message_count = msg_count
            self._sessions[conv_id] = session
            self._channel_index[channel_id] = conv_id
            self._persist(session)
            return session

    # ── Thread lookup ──

    def find_by_thread(self, platform_thread_id: str) -> Optional[Session]:
        with self._lock:
            session_id = self._thread_index.get(platform_thread_id)
            if session_id:
                return self._sessions.get(session_id)
        return None

    # ── Persistence ──

    def _load(self) -> bool:
        """Load sessions from storage (or legacy import). Returns True when the
        in-memory set differs from what storage holds (import or stamping)."""
        entries, migrating = self._storage.load_or_migrate(self._path, "sessions")
        if not entries:
            return migrating
        valid_fields = {f.name for f in dataclass_fields(Session)}
        for sdata in entries:
            try:
                # Migrate platform_thread_id -> metadata["thread_id"]
                old_thread_id = sdata.pop("platform_thread_id", None)
                if old_thread_id:
                    meta = sdata.get("metadata") or {}
                    meta.setdefault("thread_id", old_thread_id)
                    sdata["metadata"] = meta
                # Discord DMs routed on the alias before dm_route existed, so each one
                # squats a daemon-wide identity. Only the DM path built the id
                # `daemon_<user>_<route>_<hex>`, which is what separates those from a
                # Discord chat the user aliased by hand.
                meta = sdata.get("metadata") or {}
                route = meta.get(METADATA_SESSION_NAME)
                if sdata.get("source") == SessionSource.DISCORD.value and route:
                    if str(sdata.get("id", "")).startswith(f"daemon_{sdata.get('user_id')}_{route}_"):
                        meta[METADATA_DM_ROUTE] = meta.pop(METADATA_SESSION_NAME)
                        sdata["metadata"] = meta
                sdata = {k: v for k, v in sdata.items() if k in valid_fields}
                session = Session(**sdata)
            except (TypeError, KeyError) as e:
                logger.error("Skipping malformed session record %s: %s", sdata.get("id"), e)
                continue
            self._sessions[session.id] = session

        # Rebuild indexes. Legacy stores have no is_primary flag; stamp it on the
        # most-recently-active interactive session per user to preserve
        # the user's existing default-routing across the upgrade.
        primary_candidates: dict[str, str] = {}
        already_primary_keys: set[str] = set()
        for sid, session in self._sessions.items():
            if (
                session.source == SessionSource.INTERACTIVE.value
                and session.user_id
                and session.superseded_by is None
                and session.status not in FINISHED_STATUSES
            ):
                key = session.user_id
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
                self._channel_index[channel_id] = sid
        stamped = False
        for key, sid in primary_candidates.items():
            if key not in already_primary_keys:
                self._sessions[sid].metadata[METADATA_PRIMARY_FLAG] = True
                stamped = True
        return migrating or stamped

    def _persist(self, *sessions: Session) -> None:
        """Write-through the given sessions' rows. Caller holds self._lock."""
        for session in sessions:
            self._storage.upsert(session.id, asdict(session))

    # ── Turn lifecycle ──

    def set_turn_end_listener(self, callback: Optional[Callable[[str], None]]) -> None:
        self._on_turn_end = callback

    def begin_turn(self, session_id: str) -> None:
        """Durably mark a turn in flight so a daemon death mid-turn can be
        finalized at the next boot."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session and not session.turn_in_flight:
                session.turn_in_flight = True
                self._persist(session)

    def end_turn(self, session_id: Optional[str], *, notify_listeners: bool = True) -> None:
        """Clear the in-flight marker. Tolerates None/unknown ids (a turn that
        failed before session routing has nothing to clear).

        `notify_listeners=False` is for compaction's marker handoff, where the
        turn moves to the successor rather than ending.
        """
        if not session_id:
            return
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.turn_in_flight:
                session.turn_in_flight = False
                self._persist(session)
        if notify_listeners and self._on_turn_end:
            self._on_turn_end(session_id)

    def _save(self):
        """Snapshot the whole in-memory store to daemon.db in one transaction.
        Boot-time only (load reconciliation); runtime mutations use _persist."""
        self._storage.replace_all({sid: asdict(s) for sid, s in self._sessions.items()})

    def _estimate_tokens(self, session_id: str) -> tuple[int, int]:
        try:
            backend = get_history_backend()
            if not backend.exists(session_id):
                return 0, 0
            last_tokens = 0
            user_input_count = 0
            for event in backend.load(session_id).iter_events():
                if event.type == "user_input":
                    user_input_count += 1
                elif event.type == "model_response":
                    usage = event.data.get("usage") or {}
                    if isinstance(usage, dict):
                        last_tokens = usage.get("total_tokens") or usage.get("input_tokens") or last_tokens
            return last_tokens, user_input_count
        except Exception:
            return 0, 0

    def _recover_stale_sessions(self) -> bool:
        changed = False
        for session in self._sessions.values():
            was_running = session.status == SessionStatus.RUNNING.value
            if was_running:
                session.status = SessionStatus.FAILED.value
                session.error = "Daemon restarted while session was active"
                session.last_active = datetime.now(timezone.utc).isoformat()
                changed = True
            # A turn that was executing when the previous daemon died left its
            # history mid-turn (no terminal event): the progress label would
            # stay live forever and the UI would hide the turn's replay as
            # "still streaming". Finalize it in the history.
            if session.turn_in_flight or was_running:
                self._finalize_interrupted_turn(session.id)
            if session.turn_in_flight:
                session.turn_in_flight = False
                changed = True
            # A session that was mid-compaction at restart can't still be - the
            # in-memory lock didn't survive. Clear the flag so the UI doesn't
            # show a stuck "compacting…" indicator.
            if session.compacting:
                session.compacting = False
                changed = True
        return changed

    def _finalize_interrupted_turn(self, session_id: str) -> None:
        """Append a visible explanation + terminal event to an orphaned turn's
        history. Best-effort: boot recovery must never fail the daemon."""
        try:
            backend = get_history_backend()
            if not backend.exists(session_id):
                return
            now = datetime.now(timezone.utc).isoformat()
            self.append_event(
                session_id,
                {"type": "info", "message": "Turn interrupted: the daemon restarted mid-turn.", "timestamp": now},
            )
            self.append_event(
                session_id,
                {"type": "session_error", "error": "daemon restarted mid-turn", "timestamp": now},
            )
        except Exception as e:
            logger.warning("Could not finalize interrupted turn for session '%s': %s", session_id, e)

    def _migrate_legacy(self) -> bool:
        """Migrate from old SessionManager + AgentSessionStore if needed.
        Returns True when anything was imported (the boot snapshot persists it)."""
        if self._sessions:
            return False  # Already have data, skip migration

        state_dir = self._path.parent
        migrated = False

        # Migrate daemon_sessions/*.json (per-agent directories)
        for agent_dir in state_dir.iterdir():
            sessions_dir = agent_dir / "daemon_sessions" if agent_dir.is_dir() else None
            if not sessions_dir or not sessions_dir.is_dir():
                continue
            for path in sessions_dir.glob("*.json"):
                try:
                    data = json.loads(path.read_text())
                    user_id = path.stem.replace("_", ":")
                    conv_id = data.get("conversation_id", "")
                    if not conv_id:
                        continue
                    session = Session(
                        id=conv_id,
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
        return migrated


def create_interactive_session(
    session_store,
    user_id: str,
    title=None,
    event_bus=None,
    metadata=None,
    source: str = SessionSource.INTERACTIVE.value,
) -> str:
    """Provision a fresh interactive session and broadcast its creation.

    Single implementation behind both the HTTP "new chat" endpoint and the
    Jobs-tab host-session path, so session provisioning can't drift between them.
    Returns the new session id.
    """
    session_id = generate_session_id(SESSION_ID_LABEL)
    session_store.create_session(
        Session(
            id=session_id,
            source=source,
            user_id=user_id,
            title=title or None,
            metadata=metadata or {},
        )
    )
    if event_bus is not None:
        try:
            event_bus.emit("session_update", {"action": "created", "id": session_id})
        except Exception:
            logger.debug("session_update emit failed for new session '%s'", session_id)
    return session_id
