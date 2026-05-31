"""Persistent Job store backed by a JSON file."""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import uuid4

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
    JobState.STUCK.value: frozenset(),
    JobState.CANCELLED.value: frozenset(),
    JobState.ERRORED.value: frozenset(),
}


class JobStateTransitionError(ValueError):
    """Raised when a Job state change violates the state machine."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_VALID_AC_KINDS = frozenset({"ui", "test", "cmd", "llm"})


def normalize_ac(ac) -> dict:
    """Normalise a single acceptance-criterion entry to the dict shape.

    Accepts a plain string (legacy) → {text, kind:"llm"}, or a dict with
    `text` and optional `kind`. Unknown kinds fall back to "llm" so a typo
    on disk doesn't crash the orchestrator.
    """
    if isinstance(ac, dict):
        text = str(ac.get("text", "")).strip()
        kind = ac.get("kind") or "llm"
        if kind not in _VALID_AC_KINDS:
            kind = "llm"
        return {"text": text, "kind": kind}
    return {"text": str(ac).strip(), "kind": "llm"}


def normalize_acs(items) -> list[dict]:
    """Normalise an AC list to dicts, dropping empty entries."""
    if not items:
        return []
    out: list[dict] = []
    for item in items:
        entry = normalize_ac(item)
        if entry["text"]:
            out.append(entry)
    return out


@dataclass
class Job:
    id: str
    parent_session_id: str
    prompt: str
    state: str = JobState.QUEUED.value
    worker_session_id: Optional[str] = None
    # Stored as list[dict] ({text, kind}); legacy list[str] payloads are coerced
    # on construction and reload. See normalize_acs / normalize_ac.
    acceptance_criteria: list = field(default_factory=list)
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
    # Most recent verifier session id — set by orchestrator when verifier spawns
    # so a hung verifier can be cancelled from _on_timeout.
    verifier_session_id: Optional[str] = None
    # Absolute path of the provisioned git worktree (set when --repo was given);
    # workers run inside this directory. Orchestrator prunes it on DONE/CANCELLED
    # and keeps it on STUCK/ERRORED so the user can inspect what the worker did.
    worktree_path: Optional[str] = None
    # If True, the orchestrator posts a one-line wake-up message into the parent
    # session on terminal transition so the parent agent learns the Job finished
    # and can react. Defaults: /job slash → False (human-driven, tile is enough);
    # spawn_job() agent tool → True (autonomous composition).
    #
    # Legacy field — see `notify_when` for the granular replacement. `notify=True`
    # is normalised to `notify_when="terminal"` on construction so old persisted
    # jobs and old callers behave identically.
    notify: bool = False
    # When to fire the wake-up message: one of "done", "stuck", "errored",
    # "terminal" (any terminal state), or "never" (default). Replaces the binary
    # `notify` field for finer control from the new-job modal / slash command.
    notify_when: str = "never"
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

    def __post_init__(self):
        if not self.id:
            self.id = f"job-{uuid4().hex[:8]}"
        now = _now_iso()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now
        # Always normalise AC to the dict shape so downstream code (verifier
        # prompt, slash command preview, tile event payload) sees one shape.
        self.acceptance_criteria = normalize_acs(self.acceptance_criteria)
        # Legacy `notify=True` → notify_when="terminal" if notify_when wasn't set
        # explicitly. Preserves behaviour for callers / persisted jobs that
        # predate notify_when.
        if self.notify and self.notify_when in ("", "never", None):
            self.notify_when = "terminal"
        if not self.notify_when:
            self.notify_when = "never"
        if self.max_attempts is None or self.max_attempts <= 0:
            self.max_attempts = 3


class JobStore:
    """JSON-backed persistent store for Job records.

    Mirrors `WebhookStore`'s shape: in-memory dict + atomic tmpfile-swap saves.
    State transitions are guarded by `_VALID_TRANSITIONS` to prevent invalid moves.
    """

    def __init__(self, path: Path):
        self._path = path
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._load()

    def add(self, job: Job) -> Job:
        with self._lock:
            if job.id in self._jobs:
                raise ValueError(f"Job already exists: {job.id}")
            self._jobs[job.id] = job
            self._save()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list_active(self) -> list[Job]:
        return [j for j in self._jobs.values() if j.state not in _TERMINAL_STATES]

    def list_for_parent(self, parent_session_id: str) -> list[Job]:
        return [j for j in self._jobs.values() if j.parent_session_id == parent_session_id]

    def list_all(self) -> list[Job]:
        """Return every Job record, newest-first by updated_at."""
        return sorted(self._jobs.values(), key=lambda j: j.updated_at or "", reverse=True)

    def update_state(self, job_id: str, new_state: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Unknown job: {job_id}")
            allowed = _VALID_TRANSITIONS.get(job.state, frozenset())
            if new_state not in allowed:
                raise JobStateTransitionError(
                    f"Invalid Job state transition: {job.state} -> {new_state} (job {job_id})"
                )
            job.state = new_state
            job.updated_at = _now_iso()
            if new_state in _TERMINAL_STATES and not job.resolved_at:
                job.resolved_at = job.updated_at
            self._save()
            return job

    def update(self, job_id: str, **fields) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Unknown job: {job_id}")
            if "state" in fields:
                # Some orchestrator escape hatches (mark_done_manual, retry_with_hint)
                # set state via update() to bypass _VALID_TRANSITIONS. That's by design,
                # but the value must still be a real JobState — typos would silently
                # land an unrecoverable state on disk.
                valid = {s.value for s in JobState}
                if fields["state"] not in valid:
                    raise ValueError(f"Invalid Job state value: {fields['state']!r}; expected one of {sorted(valid)}")
            for key, value in fields.items():
                if not hasattr(job, key):
                    raise ValueError(f"Unknown Job field: {key}")
                if key == "acceptance_criteria":
                    value = normalize_acs(value)
                setattr(job, key, value)
            job.updated_at = _now_iso()
            self._save()
            return job

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load jobs from %s: %s", self._path, e)
            return
        for entry in data.get("jobs", []):
            # Migrate legacy state values that no longer exist in JobState.
            # LOOPING was a transient state in the v1 design that was removed; any
            # pre-upgrade Job stored as 'looping' is functionally a retry-running.
            if entry.get("state") == "looping":
                entry["state"] = JobState.RUNNING.value
            try:
                job = Job(**entry)
                self._jobs[job.id] = job
            except TypeError as e:
                logger.error("Skipping malformed Job record: %s (%s)", entry.get("id"), e)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"jobs": [asdict(j) for j in self._jobs.values()]}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(str(tmp), str(self._path))
