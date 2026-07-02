"""Persistent Job store backed by a JSON file."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import uuid4

from tsugite.core.record_store import JsonRecordStore, now_iso

logger = logging.getLogger(__name__)


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    VERIFYING = "verifying"
    DONE = "done"
    STUCK = "stuck"
    CANCELLED = "cancelled"
    ERRORED = "errored"


_TERMINAL_STATES = frozenset(
    {JobState.DONE.value, JobState.STUCK.value, JobState.CANCELLED.value, JobState.ERRORED.value}
)

_VALID_TRANSITIONS: dict[str, frozenset[str]] = {
    JobState.QUEUED.value: frozenset({JobState.RUNNING.value, JobState.CANCELLED.value, JobState.ERRORED.value}),
    JobState.RUNNING.value: frozenset(
        {JobState.VERIFYING.value, JobState.STUCK.value, JobState.CANCELLED.value, JobState.ERRORED.value}
    ),
    # VERIFYING → RUNNING is the verifier-rejected retry path; the timer covers
    # this window so a hung verifier can still hit STUCK.
    JobState.VERIFYING.value: frozenset(
        {
            JobState.DONE.value,
            JobState.RUNNING.value,
            JobState.STUCK.value,
            JobState.CANCELLED.value,
            JobState.ERRORED.value,
        }
    ),
    JobState.DONE.value: frozenset(),
    # STUCK/ERRORED are parked, not sinks: retry-with-hint resurrects either to
    # RUNNING, mark-done overrides STUCK to DONE, and cancel/dismiss gives up on
    # either -> CANCELLED (distinct from mark-done, which records a false success).
    JobState.STUCK.value: frozenset({JobState.RUNNING.value, JobState.DONE.value, JobState.CANCELLED.value}),
    JobState.CANCELLED.value: frozenset(),
    JobState.ERRORED.value: frozenset({JobState.RUNNING.value, JobState.CANCELLED.value}),
}


class JobStateTransitionError(ValueError):
    """Raised when a Job state change violates the state machine."""


def _coerce_ac_list(items) -> list[str]:
    """Coerce an AC list to plain strings, dropping empties.

    Tolerates the legacy `{text, kind}` dict shape persisted by older daemon
    versions; takes the `.text` value and discards `kind`.
    """
    if not items:
        return []
    out: list[str] = []
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
        else:
            text = str(item).strip()
        if text:
            out.append(text)
    return out


@dataclass
class Job:
    id: str
    parent_session_id: str
    prompt: str
    state: str = JobState.QUEUED.value
    worker_session_id: Optional[str] = None
    # Plain list of criterion strings the verifier grades the worker against.
    # Pre-existing persisted records may carry a legacy `{text, kind}` dict shape;
    # the coercion in __post_init__ flattens to strings on load.
    acceptance_criteria: list[str] = field(default_factory=list)
    repo: Optional[str] = None
    model: Optional[str] = None
    agent: Optional[str] = None
    timeout_minutes: int = 30
    verify_attempts: int = 0
    spawned_by: str = "user-slash"
    created_at: str = ""
    updated_at: str = ""
    resolved_at: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    # Most recent verifier session id - set by orchestrator when verifier spawns
    # so a hung verifier can be cancelled from _on_timeout.
    verifier_session_id: Optional[str] = None
    # Absolute path of the provisioned git worktree (set when --repo was given);
    # workers run inside this directory. Orchestrator prunes it on DONE/CANCELLED
    # and keeps it on STUCK/ERRORED so the user can inspect what the worker did.
    worktree_path: Optional[str] = None
    # Workspace anchor for non-repo jobs: the parent session's workspace root at
    # spawn time. Worker, verifier, predicate evaluation, and retries all resolve
    # file artifacts against this directory - without it the verifier can't
    # inspect what the worker wrote. None for repo jobs (worktree_path wins) and
    # for legacy records predating the field.
    workspace_path: Optional[str] = None
    # When to fire the wake-up message after a terminal transition: one of
    # "done", "stuck", "errored", "terminal" (any terminal state), or "never".
    # Default is None so __post_init__ can normalise to "never"; the JobStore
    # loader migrates legacy `notify=True` records to "terminal" before
    # construction.
    notify_when: Optional[str] = None
    # Maximum verifier rounds before the Job goes stuck. Defaults to 3 to match
    # the pre-feature constant; overridable per-job via /job --max-attempts or
    # the spawn_job() tool.
    max_attempts: int = 3
    # Append-only history of each worker round + its verifier, so retried jobs
    # don't orphan their earlier session ids. Each entry:
    #   {"index": int, "kind": "initial"|"retry"|"hint",
    #    "worker_session_id": str, "verifier_session_id": str | None,
    #    "verifier_pass": bool | None}
    # `worker_session_id` and `verifier_session_id` on Job point at the LATEST
    # entry; earlier entries are navigable via this list.
    attempts: list[dict] = field(default_factory=list)
    # Per-criterion verifier verdicts, one entry per AC per attempt. Each entry:
    #   {ac_index, ac_text, pass, reason, attempt}
    # Accumulates across retry attempts (the orchestrator replaces only the current
    # attempt's entries when the verifier responds). None when no verifier has run.
    ac_results: Optional[list] = None
    # Inherited sandbox policy (a SandboxSettings-shaped dict) stamped when a
    # sandboxed agent spawns the job, so worker + verifier sessions stay sandboxed.
    sandbox_override: Optional[dict] = None

    def __post_init__(self):
        if not self.id:
            self.id = f"job-{uuid4().hex[:8]}"
        now = now_iso()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now
        # Coerce to plain strings - tolerates legacy dict shape from older saves.
        self.acceptance_criteria = _coerce_ac_list(self.acceptance_criteria)
        if not self.notify_when:
            self.notify_when = "never"

    def to_payload(self) -> dict:
        """Canonical serialiser for tile/list payloads. Used by both the SSE
        `job_status` event (orchestrator._emit_job_event) and `/api/jobs` so
        the two surfaces never drift on field shape.
        """
        return {
            "job_id": self.id,
            "parent_session_id": self.parent_session_id,
            "worker_session_id": self.worker_session_id,
            "verifier_session_id": self.verifier_session_id,
            "state": self.state,
            "prompt": (self.prompt or "")[:200],
            "verify_attempts": self.verify_attempts,
            "max_attempts": self.max_attempts,
            "notify_when": self.notify_when,
            "error": self.error,
            "attempts": list(self.attempts or []),
            "acceptance_criteria": list(self.acceptance_criteria or []),
            "ac_results": list(self.ac_results or []),
            "result": self.result,
            "agent": self.agent,
            "model": self.model,
            "repo": self.repo,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "resolved_at": self.resolved_at,
            "spawned_by": self.spawned_by,
        }


class JobStore(JsonRecordStore):
    """JSON-backed persistent store for Job records.

    State transitions are guarded by `_VALID_TRANSITIONS`; resolved jobs are
    pruned beyond `max_terminal_jobs` so jobs.json stays bounded.
    """

    record_cls = Job
    collection_key = "jobs"
    record_label = "job"
    valid_transitions = _VALID_TRANSITIONS
    terminal_states = _TERMINAL_STATES
    transition_error_cls = JobStateTransitionError

    def __init__(self, path: Path, max_terminal_jobs: int = 200):
        # Retention cap for resolved jobs - without it jobs.json grows forever
        # and every save re-serializes the entire history.
        self._max_terminal_jobs = max_terminal_jobs
        super().__init__(path)

    def list_all(self) -> list[Job]:
        """Return every Job record, newest-first by updated_at."""
        return sorted(super().list_all(), key=lambda j: j.updated_at or "", reverse=True)

    def _coerce_update_value(self, key: str, value):
        if key == "state":
            # State changes normally go through update_state(); this loose path
            # exists for load-time migrations and test fixtures. The value must
            # still be a real JobState - a typo would silently land an
            # unrecoverable state on disk.
            valid = {s.value for s in JobState}
            if value not in valid:
                raise ValueError(f"Invalid Job state value: {value!r}; expected one of {sorted(valid)}")
        elif key == "acceptance_criteria":
            return _coerce_ac_list(value)
        return value

    def _on_state_updated(self, job: Job) -> None:
        if job.state in _TERMINAL_STATES:
            self._prune_terminal_overflow()

    def _prune_terminal_overflow(self) -> None:
        """Drop the oldest resolved jobs beyond the retention cap. Caller holds the lock."""
        terminal = [j for j in self._records.values() if j.state in _TERMINAL_STATES]
        overflow = len(terminal) - self._max_terminal_jobs
        if overflow <= 0:
            return
        terminal.sort(key=lambda j: j.resolved_at or j.updated_at or "")
        for victim in terminal[:overflow]:
            del self._records[victim.id]

    def _load_entry(self, entry: dict) -> Optional[Job]:
        # Migrate legacy state values that no longer exist in JobState.
        # LOOPING was a transient state in the v1 design that was removed; any
        # pre-upgrade Job stored as 'looping' is functionally a retry-running.
        if entry.get("state") == "looping":
            entry["state"] = JobState.RUNNING.value
        # Migrate legacy `notify: bool` records: True → notify_when="terminal"
        # iff the new field wasn't set. Drop the legacy key either way so the
        # Job dataclass (which no longer carries it) accepts the dict.
        legacy_notify = entry.pop("notify", None)
        if legacy_notify and not entry.get("notify_when"):
            entry["notify_when"] = "terminal"
        return super()._load_entry(entry)

    def _after_load(self) -> bool:
        self._prune_terminal_overflow()
        return False
