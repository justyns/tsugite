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

from tsugite.history import SessionStorage, Turn, generate_session_id, get_history_dir

logger = logging.getLogger(__name__)


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

    # Platform thread mapping
    platform_thread_id: Optional[str] = None

    title: Optional[str] = None

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

    def __init__(self, store_path: Path, context_limits: Optional[dict[str, int]] = None):
        self._path = store_path
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._events_dir_created = False
        self._dirty = False
        self._save_dir_created = False

        # Per-agent context limits for compaction
        self._context_limits: dict[str, int] = context_limits or {}

        # Index: (user_id, agent) -> session_id for fast interactive lookup
        self._interactive_index: dict[tuple[str, str], str] = {}

        # Index: platform_thread_id -> session_id for fast thread lookup
        self._thread_index: dict[str, str] = {}

        # Per-session compaction locks: (user_id, agent) -> Event
        # Event is unset while compaction is in progress, set when done.
        self._compaction_events: dict[tuple[str, str], threading.Event] = {}

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

    # ── Interactive session management (replaces SessionManager) ──

    def get_or_create_interactive(self, user_id: str, agent: str) -> Session:
        with self._lock:
            key = (user_id, agent)
            if key in self._interactive_index:
                session_id = self._interactive_index[key]
                if session_id in self._sessions:
                    return self._sessions[session_id]

            # Create new interactive session
            conv_id = f"daemon_{agent}_{user_id}"
            session = Session(
                id=conv_id,
                agent=agent,
                source=SessionSource.INTERACTIVE.value,
                user_id=user_id,
            )

            # Estimate tokens from existing history
            tokens, msg_count = self._estimate_tokens(conv_id)
            session.cumulative_tokens = tokens
            session.message_count = msg_count

            self._sessions[conv_id] = session
            self._interactive_index[key] = conv_id
            self._save()
            return session

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
                platform_thread_id=old_session.platform_thread_id,
            )

            self._sessions[new_id] = new_session

            # Update interactive index if this was an interactive session
            if old_session.user_id and old_session.source == SessionSource.INTERACTIVE.value:
                key = (old_session.user_id, old_session.agent)
                self._interactive_index[key] = new_id

            # Mark old session as completed
            old_session.status = SessionStatus.COMPLETED.value

            self._save()
            return new_session

    def update_token_count(self, session_id: str, tokens_used: int) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                if tokens_used > 0:
                    session.cumulative_tokens = tokens_used
                session.message_count += 1
                session.last_active = datetime.now(timezone.utc).isoformat()
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
            if session.platform_thread_id:
                self._thread_index[session.platform_thread_id] = session.id

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
        terminal = (SessionStatus.COMPLETED.value, SessionStatus.FAILED.value, SessionStatus.CANCELLED.value)
        children = [
            s
            for s in self._sessions.values()
            if s.agent == agent
            and s.source in (SessionSource.BACKGROUND.value, SessionSource.SPAWNED.value)
            and s.status in terminal
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
            self._save()
            return session

    def list_sessions(
        self,
        agent: Optional[str] = None,
        source: Optional[str] = None,
        parent_id: Optional[str] = None,
        status: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 0,
    ) -> list[Session]:
        with self._lock:
            results = [
                s
                for s in self._sessions.values()
                if (not agent or s.agent == agent)
                and (not source or s.source == source)
                and (not parent_id or s.parent_id == parent_id)
                and (not status or s.status == status)
                and (not user_id or s.user_id == user_id)
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

    # ── Event log (per-session JSONL) ──

    def _events_dir(self) -> Path:
        return self._path.parent / "sessions"

    def _event_log_path(self, session_id: str) -> Path:
        return self._events_dir() / f"{session_id}.jsonl"

    def append_event(self, session_id: str, event: dict) -> None:
        if not self._events_dir_created:
            self._events_dir().mkdir(parents=True, exist_ok=True)
            self._events_dir_created = True
        path = self._event_log_path(session_id)
        with open(path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def read_events(self, session_id: str) -> list[dict]:
        path = self._event_log_path(session_id)
        if not path.exists():
            return []
        events = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events

    def event_count(self, session_id: str) -> int:
        path = self._event_log_path(session_id)
        if not path.exists():
            return 0
        count = 0
        with open(path) as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def session_detail(self, session_id: str) -> dict:
        session = self.get_session(session_id)
        result = asdict(session)
        result["event_count"] = self.event_count(session_id)
        return result

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
                sdata = {k: v for k, v in sdata.items() if k in valid_fields}
                self._sessions[sid] = Session(**sdata)
            # Rebuild indexes
            for sid, session in self._sessions.items():
                if session.source == SessionSource.INTERACTIVE.value and session.user_id:
                    key = (session.user_id, session.agent)
                    existing_id = self._interactive_index.get(key)
                    if not existing_id or session.last_active > self._sessions[existing_id].last_active:
                        self._interactive_index[key] = sid
                if session.platform_thread_id:
                    self._thread_index[session.platform_thread_id] = sid
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
            turns = [r for r in storage.load_records() if isinstance(r, Turn)]
            last_tokens = turns[-1].tokens if turns and turns[-1].tokens else 0
            return last_tokens, storage.turn_count
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
