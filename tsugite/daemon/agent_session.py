"""Data models and persistence for async agent sessions with review gates."""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class SessionState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_FOR_REVIEW = "waiting_for_review"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class ReviewDecision(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DECLINED = "declined"


@dataclass
class ReviewGate:
    id: str
    session_id: str
    title: str
    description: str = ""
    context: dict = field(default_factory=dict)
    decision: str = ReviewDecision.PENDING.value
    reviewer_comment: str = ""
    created_at: str = ""
    resolved_at: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = f"review-{uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class AgentSession:
    id: str
    agent: str
    prompt: str
    state: str = SessionState.PENDING.value
    created_at: str = ""
    updated_at: str = ""
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[str] = None
    current_review_id: Optional[str] = None
    model: Optional[str] = None
    agent_file: Optional[str] = None
    notify: list[str] = field(default_factory=list)
    sandbox: bool = False
    allow_domains: list[str] = field(default_factory=list)
    no_network: bool = False

    def __post_init__(self):
        if not self.id:
            self.id = f"session-{uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


class AgentSessionStore:
    """JSON persistence for agent sessions and review gates.

    Sessions stored in {state_dir}/sessions.json.
    Per-session event logs in {state_dir}/sessions/{id}.jsonl.
    """

    def __init__(self, sessions_path: Path):
        self._path = sessions_path
        self._sessions: dict[str, AgentSession] = {}
        self._reviews: dict[str, ReviewGate] = {}
        self._load()
        self._recover_stale_sessions()

    def _recover_stale_sessions(self):
        """Mark sessions in running/waiting states as interrupted (daemon restart recovery)."""
        changed = False
        stale_states = {SessionState.RUNNING.value, SessionState.WAITING_FOR_REVIEW.value}
        for session in self._sessions.values():
            if session.state in stale_states:
                session.state = SessionState.INTERRUPTED.value
                session.error = "Daemon restarted while session was active"
                session.updated_at = datetime.now(timezone.utc).isoformat()
                changed = True
        for review in self._reviews.values():
            if review.decision == ReviewDecision.PENDING.value:
                parent = self._sessions.get(review.session_id)
                if parent and parent.state == SessionState.INTERRUPTED.value:
                    review.decision = ReviewDecision.DECLINED.value
                    review.reviewer_comment = "Auto-declined: daemon restarted"
                    review.resolved_at = datetime.now(timezone.utc).isoformat()
                    changed = True
        if changed:
            self._save()

    # Session CRUD

    def create_session(self, session: AgentSession) -> AgentSession:
        if session.id in self._sessions:
            raise ValueError(f"Session '{session.id}' already exists")
        self._sessions[session.id] = session
        self._save()
        return session

    def get_session(self, session_id: str) -> AgentSession:
        if session_id not in self._sessions:
            raise ValueError(f"Session '{session_id}' not found")
        return self._sessions[session_id]

    def update_session(self, session_id: str, **fields) -> AgentSession:
        session = self.get_session(session_id)
        for key, value in fields.items():
            if key in ("id", "created_at"):
                continue
            if not hasattr(session, key):
                raise ValueError(f"Unknown field '{key}'")
            setattr(session, key, value)
        session.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return session

    def list_sessions(self, state: Optional[str] = None) -> list[AgentSession]:
        sessions = list(self._sessions.values())
        if state:
            sessions = [s for s in sessions if s.state == state]
        return sessions

    # Review CRUD

    def create_review(self, review: ReviewGate) -> ReviewGate:
        if review.id in self._reviews:
            raise ValueError(f"Review '{review.id}' already exists")
        self._reviews[review.id] = review
        # Update parent session
        if review.session_id in self._sessions:
            self._sessions[review.session_id].current_review_id = review.id
            self._sessions[review.session_id].state = SessionState.WAITING_FOR_REVIEW.value
            self._sessions[review.session_id].updated_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return review

    def get_review(self, review_id: str) -> ReviewGate:
        if review_id not in self._reviews:
            raise ValueError(f"Review '{review_id}' not found")
        return self._reviews[review_id]

    def resolve_review(self, review_id: str, decision: str, comment: str = "") -> ReviewGate:
        review = self.get_review(review_id)
        if review.decision != ReviewDecision.PENDING.value:
            raise ValueError(f"Review '{review_id}' already resolved")
        review.decision = decision
        review.reviewer_comment = comment
        review.resolved_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return review

    def list_reviews(self, status: Optional[str] = None, session_id: Optional[str] = None) -> list[ReviewGate]:
        reviews = list(self._reviews.values())
        if status:
            reviews = [r for r in reviews if r.decision == status]
        if session_id:
            reviews = [r for r in reviews if r.session_id == session_id]
        return reviews

    # Event log (per-session JSONL)

    def _events_dir(self) -> Path:
        return self._path.parent / "sessions"

    def _event_log_path(self, session_id: str) -> Path:
        return self._events_dir() / f"{session_id}.jsonl"

    def append_event(self, session_id: str, event: dict) -> None:
        events_dir = self._events_dir()
        events_dir.mkdir(parents=True, exist_ok=True)
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
        return sum(1 for line in path.read_text().splitlines() if line.strip())

    # Persistence

    def _load(self):
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for sid, sdata in data.get("sessions", {}).items():
                self._sessions[sid] = AgentSession(**sdata)
            for rid, rdata in data.get("reviews", {}).items():
                self._reviews[rid] = ReviewGate(**rdata)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.error("Failed to load sessions from %s: %s", self._path, e)

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "sessions": {sid: asdict(s) for sid, s in self._sessions.items()},
            "reviews": {rid: asdict(r) for rid, r in self._reviews.items()},
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(str(tmp), str(self._path))
