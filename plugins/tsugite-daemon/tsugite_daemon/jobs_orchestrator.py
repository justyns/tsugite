"""Jobs orchestrator - bridges Job state machine with SessionRunner lifecycle.

Subscribes to SessionRunner's notify_callback to detect worker/verifier
completions, transitions Job state, spawns verifier rounds, and broadcasts
tile updates to the parent chat.

Note on timeouts: `Job.timeout_minutes` is a per-phase budget, not a whole-job
wall clock. The timer is re-armed at the start of each phase (worker run, each
verifier round, each retry worker), so a Job that loops through several attempts
can run longer than `timeout_minutes` in total - each individual phase is what's
bounded.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable, Optional
from xml.sax.saxutils import quoteattr

from tsugite_daemon.attention_store import SOURCE_JOB
from tsugite_daemon.job_predicates import _evaluate_predicate, _resolve_predicate_cwd, partition_acs
from tsugite_daemon.job_prompts import (
    _build_barrier_message,
    _build_followup_prompt,
    _build_hint_prompt,
    _build_notify_message,
    _build_verifier_prompt,
    build_worker_prompt,
)
from tsugite_daemon.job_store import _TERMINAL_STATES, Job, JobState, JobStateTransitionError, JobStore
from tsugite_daemon.job_verdicts import (
    _extract_failed_acs,
    _is_infra_failure,
    _parse_verifier_output,
    _sanitize_output_excerpt,
)
from tsugite_daemon.job_worktrees import _provision_worktree, _prune_worktree
from tsugite_daemon.session_store import FINISHED_STATUSES, METADATA_JOB_HOST, Session, SessionSource, SessionStatus

logger = logging.getLogger(__name__)

VERIFIER_AGENT = "job_verifier"
WORKER_AGENT = "job_worker"

# Terminal Job state -> the status its placeholder host session is reconciled to.
# ERRORED/STUCK map to FAILED (ERROR isn't a FINISHED_STATUSES, so it would still
# render as live); anything unmapped falls back to COMPLETED.
_JOB_TO_SESSION_STATUS = {
    JobState.DONE: SessionStatus.COMPLETED,
    JobState.CANCELLED: SessionStatus.CANCELLED,
    JobState.ERRORED: SessionStatus.FAILED,
    JobState.STUCK: SessionStatus.FAILED,
}

# Allowed notify_when values; anything else is coerced to "never" at intake.
_VALID_NOTIFY_WHEN = frozenset({"done", "stuck", "errored", "terminal", "all_done", "never"})


# A job in one of these is parked on the user: it cannot progress until someone acts.
_PARKED_STATES = frozenset({JobState.AWAITING_INPUT.value, JobState.STUCK.value, JobState.ERRORED.value})


def _with_sandbox(job: Job, metadata: dict) -> dict:
    """Carry the job's inherited sandbox policy onto a spawned session's metadata
    so worker/verifier runs stay sandboxed (resolved at the adapter chokepoint)."""
    if job.sandbox_override:
        return {**metadata, "sandbox_override": job.sandbox_override}
    return metadata


def _job_workspace(job: Job) -> Optional[str]:
    """Directory holding the job's file artifacts: the provisioned worktree for
    repo jobs, else the persisted parent-workspace anchor for non-repo jobs.
    Every spawn/eval site (worker, verifier, retries, predicates) resolves
    against this so they all see the same files."""
    return job.worktree_path or job.workspace_path


class JobsOrchestrator:
    """Drives Job state in response to worker/verifier session completions."""

    def __init__(self, job_store: JobStore, session_runner, event_bus=None, terminal_store=None):
        self._jobs = job_store
        self._runner = session_runner
        self._event_bus = event_bus
        # Optional: when wired, the orchestrator includes worker_terminal_id in
        # the job_status payload so the frontend tile can mount the embedded
        # xterm without a separate /api/terminals probe per tile.
        self._terminal_store = terminal_store
        self._timeout_handles: dict[str, asyncio.TimerHandle] = {}
        # Strong refs to in-flight notify tasks; asyncio only weak-refs scheduled
        # tasks, so without this a notify could be GC'd mid-send.
        self._notify_tasks: set[asyncio.Task] = set()
        # Strong refs to in-flight background tasks (e.g. offloaded worktree
        # prunes). Same GC rationale as _notify_tasks; also lets tests drain them.
        self._bg_tasks: set[asyncio.Task] = set()
        # Per-job locks serializing out-of-band STUCK transitions (retry / mark-done)
        # so two concurrent tile clicks can't both pass the stuck guard and double-act.
        self._job_action_locks: dict[str, asyncio.Lock] = {}
        # Pluggable job executors, keyed by Job.executor name. "agent" is implicit
        # (the built-in SessionRunner path); plugins register others here. See the
        # executor contract on register_executor.
        self._executors: dict = {}

    def register_executor(self, name: str, executor) -> None:
        """Register a non-agent job executor under `name` (matched against
        Job.executor). Called by adapter plugins at load time.

        Executor contract:

            async def start(self, job, followup: str | None) -> None
                Kick off the work for `job`. `followup` is None on the initial
                attempt; on a retry it is the prompt the agent path would have
                spawned a fresh worker with (failed-AC / hint guidance), which the
                executor should feed into its live session. The executor reports
                the outcome back via orchestrator.complete_worker / fail_worker.

            async def cancel(self, job) -> None
                Tear down the executor's child (e.g. kill the PTY). Called on
                EVERY terminal finalize - on done/cancelled BEFORE the worktree is
                pruned (the child holds the cwd open); on stuck/errored the child
                is reaped too (or it leaks), but the executor should keep any
                state a later retry needs to resume the conversation. Best-effort.
        """
        self._executors[name] = executor

    def all_jobs(self) -> list[Job]:
        """Every job, newest first."""
        return self._jobs.list_all()

    def get_job(self, job_id: str) -> Optional[Job]:
        """Read a Job by id. The public accessor for executor plugins, so they
        don't depend on the private JobStore layout."""
        return self._jobs.get(job_id)

    def attach_worker_terminal(self, job_id: str, terminal_id: str) -> None:
        """Stamp a job's worker_terminal_id so the web tile embeds that live
        terminal. Executor plugins call this instead of touching the JobStore."""
        self._jobs.update(job_id, worker_terminal_id=terminal_id)

    @property
    def executor_names(self) -> list[str]:
        """Registered non-agent executor names, for UI feature-detection.

        "agent" is implicit and excluded; the web new-job modal shows its
        executor dropdown only when this is non-empty.
        """
        return sorted(self._executors)

    def attach(self) -> None:
        """Register for session-completion notifications. Idempotent (the
        runner's listener registry dedups), so test fixtures / hot-reload paths
        can call it repeatedly without double-dispatching."""
        self._runner.add_completion_listener(self.on_session_complete)

    def shutdown(self) -> None:
        """Cancel pending timers and in-flight background tasks at daemon shutdown
        so they don't die with 'Task was destroyed but it is pending'."""
        for handle in self._timeout_handles.values():
            handle.cancel()
        self._timeout_handles.clear()
        for task in list(self._notify_tasks) + list(self._bg_tasks):
            task.cancel()

    def render_context_xml(self, session_id: str, recent_limit: int = 3) -> str:
        """Jobs context block for <message_context>; "" when the session has no jobs."""
        return render_jobs_context_xml(self.jobs_for_parent(session_id), recent_limit)

    def recover_orphaned_jobs(self) -> int:
        """Mark jobs left active by a previous daemon process as errored.

        Timeout timers live only in memory and worker/verifier sessions don't
        survive a restart (the session store flips RUNNING sessions to FAILED
        without going through the notify callback), so an active job on disk at
        startup can never progress - without this it renders as 'running'
        forever. ERRORED is retryable from the UI. Deliberately skips the
        parent-notify path: waking parent agents with LLM turns at boot is not
        what anyone wants.
        """
        recovered = 0
        for job in self._jobs.list_active():
            try:
                self._jobs.update_state(job.id, JobState.ERRORED.value)
                self._jobs.update(job.id, error="daemon restarted while job was active; retry to spawn a fresh worker")
            except (JobStateTransitionError, KeyError) as e:
                logger.warning("Could not recover orphaned job '%s': %s", job.id, e)
                continue
            self._emit_job_event(self._jobs.get(job.id))
            recovered += 1
        if recovered:
            logger.info("Recovered %d orphaned job(s) from previous daemon run", recovered)
        return recovered

    async def create_and_start_job(
        self,
        *,
        parent_session_id: str,
        prompt: str,
        acceptance_criteria: Optional[list] = None,
        repo: Optional[str] = None,
        model: Optional[str] = None,
        verifier_model: Optional[str] = None,
        model_ladder: Optional[list] = None,
        agent: Optional[str] = None,
        timeout_minutes: int = 30,
        spawned_by: str = "user-slash",
        max_attempts: Optional[int] = None,
        notify_when: Optional[str] = None,
        sandbox_override: Optional[dict] = None,
        executor: str = "agent",
        effort: Optional[str] = None,
        delegation_files: Optional[list] = None,
    ) -> tuple[Job, Optional[Session]]:
        """Create a Job record + spawn the worker session in one step.

        Used by both the /job slash command and the spawn_job() agent tool.
        Returns (job, started_worker_session). For a non-agent executor the second
        element is None (no worker Session is spawned - the executor runs the work
        and reports back via complete_worker/fail_worker). The orchestrator's
        register_worker is still called so the timeout is scheduled and the tile
        event fires.

        executor: which registered executor produces the work; "agent" (default)
            uses the built-in SessionRunner path. An unknown name raises ValueError
            before any Job record is created.
        notify_when: one of "done", "stuck", "errored", "terminal", "all_done",
            "never" (default). "all_done" replaces the per-job wake-up with a single
            aggregate one, fired when the parent's last active job finalizes.
        max_attempts: verifier-loop cap. Defaults to 3 when omitted.
        model_ladder: ordered "cheap first" model list; `model` is set to the
            first rung and qualifying failures escalate to the next rung.
        """
        worker_agent_file = agent or WORKER_AGENT

        # Reject an unknown executor before persisting anything - a Job pinned to a
        # missing executor would spawn nothing and strand in QUEUED.
        if executor != "agent" and executor not in self._executors:
            raise ValueError(f"Unknown job executor: {executor!r} (registered: {sorted(self._executors)})")

        # If the spawner didn't supply an inherited policy (the /job slash command
        # path, vs the spawn_job tool from a sandboxed agent), fall back to the
        # parent agent's own sandbox config so jobs are sandboxed whenever that
        # agent is - the worker/verifier sessions and predicate evaluation all key
        # off job.sandbox_override.
        if sandbox_override is None:
            sandbox_override = self._resolve_parent_sandbox_override(parent_session_id)

        job_kwargs: dict = {}
        if model_ladder:
            ladder = [str(m).strip() for m in model_ladder if str(m).strip()]
            if not ladder:
                raise ValueError("model_ladder must contain at least one non-empty model name")
            job_kwargs["model_ladder"] = ladder
            job_kwargs["ladder_index"] = 0
            model = ladder[0]
        if max_attempts is not None:
            job_kwargs["max_attempts"] = max_attempts
        if notify_when:
            if notify_when not in _VALID_NOTIFY_WHEN:
                logger.warning("Unknown notify_when '%s'; coercing to 'never'", notify_when)
                notify_when = "never"
            job_kwargs["notify_when"] = notify_when

        # The parent's workspace root: relative --repo paths resolve against it,
        # and non-repo jobs persist it as their workspace anchor (see
        # Job.workspace_path) so every later phase resolves the same directory,
        # even after a daemon restart when the parent session may be gone.
        workspace_root = self._resolve_workspace_root(parent_session_id)

        job = self._jobs.add(
            Job(
                id="",
                parent_session_id=parent_session_id,
                prompt=prompt,
                acceptance_criteria=acceptance_criteria or [],
                repo=repo,
                model=model,
                verifier_model=verifier_model,
                effort=effort,
                agent=worker_agent_file,
                timeout_minutes=timeout_minutes,
                spawned_by=spawned_by,
                sandbox_override=sandbox_override,
                workspace_path=str(workspace_root) if workspace_root is not None and not repo else None,
                executor=executor,
                delegation_files=delegation_files or [],
                **job_kwargs,
            )
        )
        logger.info(
            "Job '%s' created: source=%s parent_session=%s executor=%s repo=%s prompt=%r",
            job.id,
            spawned_by,
            parent_session_id,
            executor,
            repo or "-",
            (prompt or "")[:80],
        )

        # Provision a fresh git worktree if --repo was given so the worker has
        # an isolated working tree (no clashes with the parent shell or other jobs).
        if repo:
            try:
                # Offload the blocking `git worktree add` so a slow clone/checkout
                # can't stall the daemon's single event loop (the retry/prune paths
                # already do this).
                worktree_path = await asyncio.to_thread(_provision_worktree, repo, job.id, workspace_root)
                job = self._jobs.update(job.id, worktree_path=worktree_path)
            except Exception as e:
                logger.exception("Failed to provision worktree for job '%s': %s", job.id, e)
                self._finalize(job, JobState.ERRORED, error=f"worktree provisioning failed: {e}")
                raise

        worker_prompt = build_worker_prompt(prompt, acceptance_criteria or [], repo)
        try:
            started = await self._spawn_worker(job, prompt=worker_prompt, workspace=_job_workspace(job))
        except Exception as e:
            # Don't leave the Job persisted in QUEUED; mark it ERRORED so it
            # doesn't accumulate as a zombie in jobs.json.
            self._finalize(job, JobState.ERRORED, error=f"worker spawn failed: {e}")
            raise
        self.register_worker(job.id, started.id if started else None, timeout_minutes=timeout_minutes)
        return job, started

    async def _spawn_worker(
        self,
        job: Job,
        *,
        prompt: str,
        workspace: Optional[str],
        extra_metadata: Optional[dict] = None,
        followup: Optional[str] = None,
    ) -> Optional[Session]:
        """Produce the work for `job` - the single spawn path for initial,
        verifier-rejected retry, hint-retry, and escalation workers.

        Agent jobs build + start a worker Session and return it. Non-agent jobs
        dispatch to the registered executor (which reports back via
        complete_worker/fail_worker) and return None. `followup` is None on the
        initial spawn and the retry/hint/escalation prompt otherwise; the executor
        feeds it into its live session instead of respawning. Raises on spawn
        failure; callers decide whether that finalizes the Job or surfaces to the user."""
        if job.executor != "agent":
            executor = self._executors.get(job.executor)
            if executor is None:
                raise ValueError(f"No executor registered for '{job.executor}'")
            await executor.start(job, followup=followup)
            return None
        # Carry any delegated files onto the worker session's metadata; the runner
        # materializes them into the first-turn attachments once the model is known
        # (same key it reads for spawn_agent-style delegation).
        worker_meta: dict = {"job_id": job.id, **(extra_metadata or {})}
        if job.delegation_files:
            worker_meta["delegation_files"] = list(job.delegation_files)
        session = Session(
            id="",
            source=SessionSource.SPAWNED.value,
            prompt=prompt,
            # Sidebar label; without it the UI falls back to the raw session id.
            # Also skips the LLM auto-title pass on completion.
            title=f"{job.id} · worker",
            agent_file=job.agent or WORKER_AGENT,
            model=job.model,
            workspace_override=workspace,
            metadata=_with_sandbox(job, worker_meta),
        )
        return self._runner.start_session(session)

    def _activate_worker(
        self, job_id: str, worker_session_id: str, *, kind: str, timeout_minutes: int, clear_error: bool = False
    ) -> None:
        """Common bookkeeping after a retry/hint worker spawn: attempt entry,
        RUNNING transition, tile event, fresh phase timer."""
        fields: dict = {"worker_session_id": worker_session_id}
        if clear_error:
            # barrier_notified too, or a resurrected job's one-job batch could never close.
            fields.update(
                resolved_at=None, error=None, error_detail=None, pending_question=None, barrier_notified=False
            )
        self._jobs.update(job_id, **fields)
        self._append_attempt(job_id, kind=kind, worker_session_id=worker_session_id)
        try:
            self._jobs.update_state(job_id, JobState.RUNNING.value)
        except JobStateTransitionError as e:
            logger.warning("Cannot transition job '%s' to running after %s worker spawn: %s", job_id, kind, e)
            return
        self._emit_job_event(self._jobs.get(job_id))
        self._schedule_timeout(job_id, timeout_minutes)

    def _append_attempt(self, job_id: str, *, kind: str, worker_session_id: str) -> None:
        """Append a new worker-attempt entry to Job.attempts. Called by every
        spawn path (initial, verifier-rejected retry, hint-retry)."""
        job = self._jobs.get(job_id)
        if job is None:
            return
        attempts = list(job.attempts or [])
        attempts.append(
            {
                "index": len(attempts),
                "kind": kind,
                "worker_session_id": worker_session_id,
                "verifier_session_id": None,
                "verifier_pass": None,
                # Which model this attempt ran on - makes a ladder walk (and any
                # manual retry-on-model) auditable in the UI/history.
                "model": job.model,
            }
        )
        self._jobs.update(job_id, attempts=attempts)

    def _update_latest_attempt(self, job_id: str, **patch) -> None:
        """Patch the latest attempt entry in Job.attempts (verifier session id, verdict)."""
        job = self._jobs.get(job_id)
        if job is None or not job.attempts:
            return
        attempts = list(job.attempts)
        attempts[-1] = {**attempts[-1], **patch}
        self._jobs.update(job_id, attempts=attempts)

    def _record_ac_results(
        self,
        job_id: str,
        raw_ac_results,
        attempt_num: int,
        *,
        ac_index_map: Optional[list[int]] = None,
        merge_with_existing_attempt: bool = False,
    ) -> None:
        """Append per-criterion verdicts for one verifier round to Job.ac_results.

        Replaces any prior entries tagged with the same attempt_num (defensive in
        case a verifier completion is delivered twice). The list itself grows by
        one batch per attempt so the UI can render historical verdicts.

        Args:
            ac_index_map: optional list mapping each raw entry's positional index
                to its original AC index in the Job's full AC list. Used by the
                mixed-mode predicate+prose path: the verifier sees only prose
                ACs at positions 0..M, but their ac_index_map values point to
                the original positions (e.g. [1, 3]).
            merge_with_existing_attempt: when True, do NOT wipe existing entries
                from the same attempt_num. Used after pre-recording predicate
                results: the verifier completion then adds prose results without
                clobbering the already-stored predicate ones.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return
        if merge_with_existing_attempt:
            prior = list(job.ac_results or [])
        else:
            prior = [e for e in (job.ac_results or []) if e.get("attempt") != attempt_num]
        new_entries: list[dict] = []
        if isinstance(raw_ac_results, list):
            for i, item in enumerate(raw_ac_results):
                ac_idx = ac_index_map[i] if ac_index_map is not None and i < len(ac_index_map) else i
                if isinstance(item, dict):
                    new_entries.append(
                        {
                            "ac_index": ac_idx,
                            "ac_text": item.get("ac_text", ""),
                            "pass": bool(item.get("pass")),
                            "reason": item.get("reason"),
                            "attempt": attempt_num,
                        }
                    )
                else:
                    # Malformed verifier output (string, None, etc) - keep a
                    # placeholder so the UI still shows something.
                    new_entries.append(
                        {
                            "ac_index": ac_idx,
                            "ac_text": "(malformed verifier output)",
                            "pass": False,
                            "reason": repr(item),
                            "attempt": attempt_num,
                        }
                    )
        self._jobs.update(job_id, ac_results=prior + new_entries)

    def register_worker(self, job_id: str, worker_session_id: Optional[str], timeout_minutes: int) -> None:
        """Record the worker session id and schedule the wall-clock timeout.

        worker_session_id is None for non-agent executor jobs (no Session exists);
        the attempt entry and tile still get recorded so the job progresses."""
        # A synchronous startup failure (e.g. the cc executor's _fail during
        # start()) already finalized the Job to ERRORED while it was QUEUED. Don't
        # resurrect it: ERRORED -> RUNNING is a valid (retry) transition, so an
        # unconditional flip here would re-zombie the just-errored job.
        job = self._jobs.get(job_id)
        if job is not None and job.state != JobState.QUEUED.value:
            logger.info(
                "Job '%s' left QUEUED (now '%s') before register_worker - skipping RUNNING flip", job_id, job.state
            )
            return
        self._jobs.update(job_id, worker_session_id=worker_session_id)
        # First call to register_worker for this Job - record the initial attempt.
        # Retry / hint paths bypass register_worker and call _append_attempt directly.
        if job is not None and not (job.attempts or []):
            self._append_attempt(job_id, kind="initial", worker_session_id=worker_session_id)
        # The guard above guarantees the Job is QUEUED here, and QUEUED -> RUNNING
        # is always valid, so no transition-error handling is needed.
        self._jobs.update_state(job_id, JobState.RUNNING.value)
        self._emit_job_event(self._jobs.get(job_id))
        self._schedule_timeout(job_id, timeout_minutes)

    # ── Tile actions (called from HTTP /api/jobs/<id>/{cancel,mark-done,retry}) ──

    def _job_lock(self, job_id: str) -> asyncio.Lock:
        """Get-or-create the per-job action lock. Race-free on the single event
        loop: there is no await between the dict lookup and the assignment, so two
        coroutines can't both create a competing lock for the same job."""
        lock = self._job_action_locks.get(job_id)
        if lock is None:
            lock = asyncio.Lock()
            self._job_action_locks[job_id] = lock
        return lock

    async def cancel_job(self, job_id: str, reason: str = "cancelled by user") -> Job:
        """User-initiated cancel/dismiss from the tile.

        No-op only when the Job is already resolved (DONE/CANCELLED). STUCK and
        ERRORED are *parked*, not resolved: cancel is the "give up / dismiss" action
        for them, distinct from mark-done (which records a false success). Their
        sessions are already terminal, so the loop below skips them.
        """
        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(f"Unknown job: {job_id}")
        if job.state in (JobState.DONE.value, JobState.CANCELLED.value):
            return job
        for sid in (job.worker_session_id, job.verifier_session_id):
            if sid and not self._session_already_terminal(sid):
                try:
                    self._runner.cancel_session(sid)
                except Exception:
                    logger.exception("cancel_job: failed to cancel session '%s'", sid)
        self._finalize(job, JobState.CANCELLED, error=reason)
        return self._jobs.get(job_id)

    async def pause_worker(self, job_id: str, question: str) -> None:
        """Executor-facing pause: the worker is blocked on supervisor input it
        cannot obtain itself (e.g. cc's CCDRIVER_NEED_INPUT marker). Parks the
        job in AWAITING_INPUT - a within-attempt pause: no verify attempt is
        consumed, the live worker (PTY) stays up, and the phase timer keeps
        running so an unanswered question still times out to STUCK - then wakes
        the parent session so its agent can answer via respond_to_job (or
        escalate to the human). No-op when the Job is not RUNNING: a pause must
        never revive a resolved/parked job."""
        job = self._jobs.get(job_id)
        if job is None:
            logger.warning("pause_worker for unknown job '%s'", job_id)
            return
        if job.state != JobState.RUNNING.value:
            logger.warning("pause_worker for job '%s' in state '%s' (expected RUNNING) - ignoring", job_id, job.state)
            return
        # The RUNNING guard above makes this transition always valid.
        self._jobs.update_state(job_id, JobState.AWAITING_INPUT.value)
        self._jobs.update(job_id, pending_question=question)
        fresh = self._jobs.get(job_id)
        self._emit_job_event(fresh)
        self._schedule_reply(
            fresh,
            f"Job {fresh.id} needs input: {question}\n"
            f"Answer with respond_to_job('{fresh.id}', <answer>) to resume the worker. "
            f"Only escalate to the user (ask_user) if you cannot answer this yourself.",
            source="job_needs_input",
        )

    async def respond_to_job(self, job_id: str, message: str) -> Job:
        """Deliver supervisor input to an executor job's live worker: answer an
        AWAITING_INPUT question (resumes the attempt) or steer a RUNNING worker
        mid-flight. The executor feeds the message into its live session - for
        cc that's typed into the PTY - without consuming a verification attempt.

        Agent jobs are rejected: their workers are autonomous sessions with no
        steering channel (a durable comment thread is a separate feature).
        Parked jobs are rejected too - retry-with-hint is the resurrect path.
        """
        if not (message or "").strip():
            raise ValueError("message is required")
        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(f"Unknown job: {job_id}")
        if job.executor == "agent":
            raise ValueError("respond_to_job only supports executor jobs; agent workers have no steering channel")
        if job.state not in (JobState.AWAITING_INPUT.value, JobState.RUNNING.value):
            raise ValueError(f"job '{job_id}' is {job.state}, not running - use retry for parked jobs")
        executor = self._executors.get(job.executor)
        if executor is None:
            raise ValueError(f"executor '{job.executor}' is not loaded")
        # Deliver first: resuming clears the pending question and re-arms the timer,
        # so an answer that never lands would leave the job RUNNING with nobody
        # driving it.
        await executor.start(job, followup=message)
        if job.state == JobState.AWAITING_INPUT.value:
            await self.resume_worker(job_id)
        return self._jobs.get(job_id)

    async def resume_worker(self, job_id: str) -> Job:
        """Return an AWAITING_INPUT job to RUNNING with a fresh phase timer and
        a cleared question. Called when the pause ends: an answer arrives via
        respond_to_job, or a human answered directly in the live TUI (the hook
        route sees a Stop while paused)."""
        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(f"Unknown job: {job_id}")
        self._jobs.update_state(job_id, JobState.RUNNING.value)
        self._jobs.update(job_id, pending_question=None)
        job = self._jobs.get(job_id)
        self._schedule_timeout(job_id, job.timeout_minutes)
        self._emit_job_event(job)
        return job

    async def mark_done_manual(self, job_id: str, reason: str = "marked done by user") -> Job:
        """Override a STUCK Job to DONE. Audit trail goes into result.manual_done_reason
        AND result.stuck_error_at_override (the verifier's prior diagnostic)."""
        # Share retry_with_hint's per-job lock so a concurrent retry can't interleave
        # with this STUCK override.
        async with self._job_lock(job_id):
            job = self._jobs.get(job_id)
            if job is None:
                raise ValueError(f"Unknown job: {job_id}")
            if job.state != JobState.STUCK.value:
                raise ValueError(f"mark_done_manual only valid on stuck jobs (job '{job_id}' is {job.state})")
            # Defensive: STUCK should have no pending timer (set in _finalize), but cancel
            # to be safe in case a future code path leaves one behind.
            self._cancel_timeout(job_id)
            result = dict(job.result or {})
            result["manual_done_reason"] = reason
            # Preserve the verifier's diagnostic so we don't lose the audit trail of why
            # the job was stuck in the first place.
            if job.error:
                result["stuck_error_at_override"] = job.error
            self._jobs.update_state(job_id, JobState.DONE.value)
            self._jobs.update(job_id, result=result, error=None, resolved_at=_iso_now())
            self.close_batch_barrier(job.parent_session_id)
            # Clean exit: stop a non-agent executor's child BEFORE pruning its cwd.
            await self._cancel_executor(self._jobs.get(job_id))
            if job.worktree_path:
                await asyncio.to_thread(_prune_worktree, job.worktree_path)
                self._jobs.update(job_id, worktree_path=None)
            self._emit_job_event(self._jobs.get(job_id))
            return self._jobs.get(job_id)

    async def retry_with_hint(
        self,
        job_id: str,
        hint: str,
        *,
        reset_counter: bool = False,
        fresh_workspace: bool = False,
        model: Optional[str] = None,
        verifier_model: Optional[str] = None,
    ) -> Job:
        """Give a STUCK or ERRORED Job one more shot, with the user's hint as the worker prompt.

        Args:
            job_id: STUCK/ERRORED job to resurrect.
            hint: Free-text guidance for the new worker. Optional when `model`
                is supplied - retrying purely to switch models (e.g. after a
                usage-limit death) shouldn't force the user to invent a hint.
            reset_counter: Zero out `verify_attempts` so the retry gets a full new
                budget of verifier rounds (the UI exposes this as "reset to 1").
                Defaults False to preserve the historical no-infinite-loops guard.
            fresh_workspace: When the job has a `repo` worktree, prune the existing
                tree and recreate it from HEAD before spawning. No-op when the Job
                was created without `repo`.
            model: Run this and subsequent attempts on a different model
                (persisted onto the Job).
            verifier_model: Same, for the verifier round.
        """
        if not (hint or "").strip() and not model:
            raise ValueError("hint or model is required - an unchanged retry would just repeat the same failure")
        # Serialize the whole check-then-act region per job: the stuck guard and the
        # stuck → running flip straddle the fresh_workspace provisioning await, so two
        # concurrent retries would otherwise both pass the guard and double-spawn.
        async with self._job_lock(job_id):
            # Re-read inside the lock - a concurrent retry / mark-done may have already
            # moved this job out of STUCK while we waited for the lock. A stale read
            # here is exactly what lets the second caller double-spawn.
            job = self._jobs.get(job_id)
            if job is None:
                raise ValueError(f"Unknown job: {job_id}")
            if job.state not in (JobState.STUCK.value, JobState.ERRORED.value):
                raise ValueError(f"retry_with_hint only valid on stuck/errored jobs (job '{job_id}' is {job.state})")

            worktree_path = job.worktree_path
            if fresh_workspace and job.repo:
                try:
                    if worktree_path and Path(worktree_path).exists():
                        await asyncio.to_thread(_prune_worktree, worktree_path)
                    workspace_root = self._resolve_workspace_root(job.parent_session_id)
                    worktree_path = await asyncio.to_thread(_provision_worktree, job.repo, job.id, workspace_root)
                    job = self._jobs.update(job_id, worktree_path=worktree_path)
                except Exception as e:
                    logger.exception("retry_with_hint: fresh_workspace failed for job '%s': %s", job_id, e)
                    raise ValueError(f"failed to recreate worktree: {e}") from e
            else:
                anchor = _job_workspace(job)
                if anchor and not Path(anchor).is_dir():
                    # Worktree/workspace was hand-deleted between STUCK and retry -
                    # refuse rather than spawn a worker into a missing directory.
                    raise ValueError(
                        f"retry_with_hint: workspace at '{anchor}' no longer exists; cannot resume in a missing directory"
                    )

            if model or verifier_model:
                fields = {}
                if model:
                    fields["model"] = model
                if verifier_model:
                    fields["verifier_model"] = verifier_model
                self._jobs.update(job_id, **fields)
                job = self._jobs.get(job_id)

            if reset_counter:
                self._jobs.update(job_id, verify_attempts=0)
                job = self._jobs.get(job_id)

            # fail_worker() acts only on a live job, so leave the parked state before
            # the executor starts, or its startup failure is dropped and the job is
            # activated on top of a worker that never ran.
            activate_first = job.executor != "agent"
            if activate_first:
                self._activate_worker(job_id, None, kind="hint", timeout_minutes=job.timeout_minutes, clear_error=True)

            try:
                hint_prompt = _build_hint_prompt(job, hint)
                started = await self._spawn_worker(
                    job,
                    prompt=hint_prompt,
                    workspace=_job_workspace(job),
                    extra_metadata={"hint_attempt": True},
                    followup=hint_prompt,
                )
            except Exception as e:
                logger.exception("retry_with_hint: failed to spawn worker for job '%s': %s", job_id, e)
                if activate_first:
                    # The optimistic RUNNING above also cleared the old error, so
                    # without this the job sits live with no diagnostic, refusing
                    # further retries until its phase timeout fires.
                    self._finalize(job, JobState.ERRORED, error=f"retry worker spawn failed: {e}")
                raise ValueError(f"failed to spawn retry worker: {e}") from e
            if not activate_first:
                self._activate_worker(
                    job_id,
                    started.id if started else None,
                    kind="hint",
                    timeout_minutes=job.timeout_minutes,
                    clear_error=True,
                )
            return self._jobs.get(job_id)

    async def on_session_complete(self, session: Session, result_str: str) -> None:
        job_id = (session.metadata or {}).get("job_id")
        if not job_id:
            return
        job = self._jobs.get(job_id)
        if job is None:
            logger.warning("Session '%s' references unknown job '%s'", session.id, job_id)
            return

        is_verifier = bool((session.metadata or {}).get("verifier_for"))
        session_failed = session.status in (SessionStatus.FAILED.value, SessionStatus.CANCELLED.value)

        if is_verifier:
            # A verifier speaks only for the phase it was spawned in: once the job
            # has moved on, a duplicate or superseded callback must not advance or
            # finalize the attempt that replaced it (RUNNING -> ERRORED is legal).
            if job.state != JobState.VERIFYING.value:
                logger.warning(
                    "Verifier callback for job '%s' in state '%s' (expected VERIFYING) - ignoring",
                    job.id,
                    job.state,
                )
            elif session_failed:
                # Verifier crashed or got cancelled - infrastructure failure, NOT a
                # verdict against the worker. Don't burn a verify attempt.
                reason = session.error or result_str or f"verifier session ended with status '{session.status}'"
                self._finalize(job, JobState.ERRORED, error=f"verifier infra failure: {reason}")
            else:
                await self._handle_verifier_complete(job, result_str)
        elif session_failed:
            await self._handle_worker_failed(job, session, result_str)
        else:
            await self._handle_worker_complete(job, session, result_str)

    async def _handle_worker_complete(self, job: Job, worker: Session, result_str: str) -> None:
        """Agent-path worker completion (from on_session_complete). Delegates to
        the shared verify flow keyed on the worker session id."""
        await self._run_verification(job, worker_id=worker.id, result_str=result_str)

    async def complete_worker(self, job_id: str, summary: str) -> None:
        """Executor-facing completion: a non-agent executor finished an attempt and
        reports its summary. Routes into the SAME verify/done/retry flow the agent
        path uses. No-op when the Job is not RUNNING (already terminal/cancelled)."""
        job = self._jobs.get(job_id)
        if job is None:
            logger.warning("complete_worker for unknown job '%s'", job_id)
            return
        await self._run_verification(job, worker_id=None, result_str=summary)

    async def _run_verification(self, job: Job, *, worker_id: Optional[str], result_str: str) -> None:
        # Guard: if the Job was already advanced out of RUNNING by a concurrent
        # path (e.g. user cancellation, external state mutation), don't overwrite
        # the result and don't attempt the VERIFYING transition. A late notify for
        # a cancelled job otherwise lands a contradictory worker summary on the tile.
        if job.state != JobState.RUNNING.value:
            logger.warning(
                "Worker completion for job '%s' in state '%s' (expected RUNNING) - ignoring",
                job.id,
                job.state,
            )
            return
        self._cancel_timeout(job.id)
        # Persist worker output before spawning the verifier so the verifier-pass
        # path can echo it back into job.result.
        self._jobs.update(job.id, result={"summary": result_str})
        try:
            self._jobs.update_state(job.id, JobState.VERIFYING.value)
        except JobStateTransitionError as e:
            logger.warning("Cannot transition job '%s' to verifying: %s", job.id, e)
            self._emit_job_event(self._jobs.get(job.id))
            return
        self._emit_job_event(self._jobs.get(job.id))

        if not job.acceptance_criteria:
            self._finalize(job, JobState.DONE, result={"summary": result_str})
            return

        # Partition into predicates (mechanically decided) vs prose (sent to LLM).
        # Predicates evaluate locally first; their results are pre-recorded onto
        # Job.ac_results before any verifier spawn so the UI sees verdicts even
        # if the verifier never runs.
        predicates, prose_entries = partition_acs(job.acceptance_criteria)
        attempt_num = job.verify_attempts + 1
        predicate_results: list[dict] = []
        if predicates:
            cwd = _resolve_predicate_cwd(job)

            def _eval_all() -> list[dict]:
                return [
                    _evaluate_predicate(
                        p["predicate"],
                        cwd=cwd,
                        ac_index=p["ac_index"],
                        ac_text=p["ac_text"],
                        attempt=attempt_num,
                        sandbox_override=job.sandbox_override,
                    )
                    for p in predicates
                ]

            # Predicates shell out via subprocess.run - offload off the daemon's
            # single event loop so a slow/hung command can't stall all sessions,
            # SSE, and timers.
            predicate_results = await asyncio.to_thread(_eval_all)
            # The await above is a yield point: the per-phase timeout can fire
            # mid-eval and finalize this Job to STUCK. Re-read state and bail
            # before recording verdicts / spawning a verifier - otherwise we'd
            # spend a verifier LLM call and arm a timer on a terminal Job.
            current = self._jobs.get(job.id)
            if current is None or current.state != JobState.VERIFYING.value:
                logger.warning(
                    "Job '%s' left VERIFYING (now '%s') during predicate eval - skipping verifier spawn",
                    job.id,
                    current.state if current else "deleted",
                )
                return
            # Pre-record predicate verdicts. Even on predicate failure the UI
            # should be able to render them; on the prose-spawn path the
            # verifier completion will merge against these.
            self._record_predicate_results(job.id, predicate_results, attempt_num)
            failed_predicates = [r for r in predicate_results if not r["pass"]]
            if failed_predicates:
                # Short-circuit BEFORE spawning the verifier - there's no point
                # spending tokens on prose ACs when a predicate already failed.
                self._update_latest_attempt(job.id, verifier_pass=False)
                await self._handle_verifier_failure(job, failed_acs=failed_predicates)
                return

        prose_acs = [e["ac_text"] for e in prose_entries]
        if not prose_acs:
            # All-predicate job, all predicates passed → straight to DONE.
            self._update_latest_attempt(job.id, verifier_pass=True)
            fresh = self._jobs.get(job.id)
            result = dict(fresh.result or {})
            result["ac_results"] = predicate_results
            self._finalize(job, JobState.DONE, result=result)
            return

        # Mixed (predicates passed + prose remaining) or pure-prose path -
        # spawn the verifier with prose ACs only.

        # Arm a fresh timer covering the verifier round; a hung verifier would
        # otherwise leave the Job pinned in VERIFYING forever.
        self._schedule_timeout(job.id, job.timeout_minutes)

        verifier_prompt = _build_verifier_prompt(job, worker_output=result_str, prose_acs=prose_acs)
        verifier_session = Session(
            id="",
            source=SessionSource.SPAWNED.value,
            prompt=verifier_prompt,
            title=f"{job.id} · verifier",
            agent_file=VERIFIER_AGENT,
            # Verifier uses its own model override when set, else inherits the
            # job's model (same override as the worker) for an agent job, else the
            # workspace default. A non-agent (e.g. cc) job's `model` is the driven
            # tool's own model string (a claude CLI alias like "sonnet"), NOT a
            # tsugite provider:model - so the verifier must never inherit it.
            model=job.verifier_model or (job.model if job.executor == "agent" else None),
            # Same directory as the worker - the verifier inspects the files the
            # worker wrote (and `git diff` for repo jobs), which only exist there.
            workspace_override=_job_workspace(job),
            # verifier_for must stay truthy so on_session_complete detects this as a
            # verifier (not a worker) completion. Executor jobs have no worker
            # session id, so fall back to the job id as the marker.
            metadata=_with_sandbox(job, {"job_id": job.id, "verifier_for": worker_id or job.id}),
        )
        try:
            started_verifier = self._runner.start_session(verifier_session)
        except Exception as e:
            logger.exception("Failed to spawn verifier for job '%s': %s", job.id, e)
            self._finalize(job, JobState.ERRORED, error=f"verifier spawn failed: {e}")
            return
        # Store verifier session id so _on_timeout can cancel a hung verifier.
        self._jobs.update(job.id, verifier_session_id=started_verifier.id)
        self._update_latest_attempt(job.id, verifier_session_id=started_verifier.id)

    def _snapshot_attempt_results(self, job_id: str, attempt_num: int) -> list[dict]:
        """This attempt's ac_results with the attempt tag stripped - the audit
        copy stored on job.result for DONE/STUCK."""
        job = self._jobs.get(job_id)
        return [
            {k: v for k, v in entry.items() if k != "attempt"}
            for entry in ((job.ac_results if job else None) or [])
            if entry.get("attempt") == attempt_num
        ]

    def _record_predicate_results(self, job_id: str, predicate_results: list[dict], attempt_num: int) -> None:
        """Append predicate-evaluated ac_results onto Job.ac_results.

        Predicate results already carry their original ac_index, attempt, etc -
        we just need to wipe any prior entries from the same attempt (defensive
        against a duplicate worker-complete) and append.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return
        prior = [e for e in (job.ac_results or []) if e.get("attempt") != attempt_num]
        self._jobs.update(job_id, ac_results=prior + list(predicate_results))

    def _next_ladder_model(self, job: Job) -> Optional[str]:
        ladder = job.model_ladder or []
        nxt = job.ladder_index + 1
        return ladder[nxt] if nxt < len(ladder) else None

    async def _escalate(self, job: Job, *, prompt: str, reason: str, followup: Optional[str] = None) -> bool:
        """Advance the Job to its next ladder rung and spawn a worker there with
        a fresh verifier budget. Returns False when there is no next rung (the
        caller then finalizes normally). Works from both RUNNING (worker infra
        death) and VERIFYING (budget exhausted) states."""
        next_model = self._next_ladder_model(job)
        if not next_model:
            return False
        self._cancel_timeout(job.id)
        self._jobs.update(
            job.id,
            model=next_model,
            ladder_index=job.ladder_index + 1,
            verify_attempts=0,
            resolved_at=None,
            error=None,
            error_detail=None,
        )
        job = self._jobs.get(job.id)
        logger.info("Job '%s' escalating to model '%s' (%s)", job.id, next_model, reason)
        # Go live BEFORE the executor starts, for the same reason the retry paths do:
        # fail_worker() acts only on a live job, so a startup failure raised from
        # `verifying` would be dropped and the job then marked running behind a
        # worker that never ran.
        if job.state != JobState.RUNNING.value:
            try:
                self._jobs.update_state(job.id, JobState.RUNNING.value)
            except JobStateTransitionError as e:
                logger.warning("Cannot transition job '%s' to running before escalation: %s", job.id, e)
        try:
            started = await self._spawn_worker(
                job,
                prompt=prompt,
                workspace=_job_workspace(job),
                extra_metadata={"escalation": True},
                followup=followup,
            )
        except Exception as e:
            logger.exception("Failed to spawn escalation worker for job '%s': %s", job.id, e)
            self._finalize(job, JobState.ERRORED, error=f"escalation worker spawn failed: {e}")
            return True  # handled (terminally)
        started_id = started.id if started else None
        self._jobs.update(job.id, worker_session_id=started_id)
        self._append_attempt(job.id, kind="escalation", worker_session_id=started_id)
        self._emit_job_event(self._jobs.get(job.id))
        self._schedule_timeout(job.id, job.timeout_minutes)
        return True

    async def _handle_worker_failed(self, job: Job, worker: Session, result_str: str) -> None:
        """Agent-path worker failure (from on_session_complete). Derives the reason
        + cancellation intent from the session, then routes into the shared path."""
        if worker.status == SessionStatus.CANCELLED.value:
            reason = worker.error or result_str or "worker session was cancelled"
            cancelled = True
        else:
            reason = worker.error or result_str or f"worker session ended with status '{worker.status}'"
            cancelled = False
        await self._process_worker_failure(job, reason=reason, cancelled=cancelled)

    async def fail_worker(self, job_id: str, error: str, detail: Optional[str] = None) -> None:
        """Executor-facing failure: a non-agent executor's attempt failed (process
        died, missing binary, etc). Routes into the SAME errored/retry path the
        agent path uses. No-op once the Job is already verifying or finalized
        (see _process_worker_failure). `detail` is an optional multi-line
        diagnostic (e.g. the worker PTY's output tail) persisted onto
        job.error_detail so the failure is diagnosable from the tile."""
        job = self._jobs.get(job_id)
        if job is None:
            logger.warning("fail_worker for unknown job '%s'", job_id)
            return
        await self._process_worker_failure(job, reason=error, cancelled=False, detail=detail)

    async def _process_worker_failure(
        self, job: Job, *, reason: str, cancelled: bool = False, detail: Optional[str] = None
    ) -> None:
        # Any other state means a late or duplicate notify for an already-finalized
        # Job (a timeout marked it stuck, then the cancelled worker's own notify
        # lands here); acting on it would overwrite the real outcome.
        if job.state not in (JobState.RUNNING.value, JobState.QUEUED.value, JobState.AWAITING_INPUT.value):
            logger.warning(
                "Worker failure for job '%s' in state '%s' (expected running/queued/awaiting_input) - ignoring",
                job.id,
                job.state,
            )
            return
        self._cancel_timeout(job.id)
        # User-initiated cancellation vs genuine failure - terminal state reflects intent.
        if cancelled:
            self._finalize(job, JobState.CANCELLED, error=reason)
            return
        if _is_infra_failure(reason):
            # Quota/rate-limit death is not a quality failure: escalate to the
            # next ladder rung when one exists, and label it either way so the
            # user knows waiting or switching models (not fixing the task) is
            # the remedy. Never consumes a verifier attempt.
            fresh_prompt = build_worker_prompt(job.prompt, job.acceptance_criteria or [], job.repo)
            if await self._escalate(job, prompt=fresh_prompt, reason=f"worker infra failure: {reason}", followup=None):
                return
            self._finalize(
                job,
                JobState.ERRORED,
                error=f"infrastructure/quota failure (not a quality failure): {reason}",
                error_detail=detail,
            )
            return
        self._finalize(job, JobState.ERRORED, error=reason, error_detail=detail)

    async def _handle_verifier_complete(self, job: Job, result_str: str) -> None:
        # Attempt counter is 1-indexed (matches the UX of "attempt #N"). Captured
        # before _handle_verifier_failure bumps job.verify_attempts.
        attempt_num = job.verify_attempts + 1

        # Recompute predicate / prose partition so the verifier's positional
        # ac_results can be mapped back to original AC indices. A Job with no
        # predicates gets a plain 1-to-1 mapping (None map + no merge).
        predicates, prose_entries = partition_acs(job.acceptance_criteria)
        if predicates:
            prose_ac_indices: Optional[list[int]] = [e["ac_index"] for e in prose_entries]
            merge_with_existing_attempt = True
        else:
            prose_ac_indices = None
            merge_with_existing_attempt = False

        parsed = _parse_verifier_output(result_str)
        if parsed is None:
            self._update_latest_attempt(job.id, verifier_pass=False)
            # Synthetic one-entry batch so the UI sees *something* for this
            # attempt; the excerpt makes the formatting failure diagnosable from
            # the tile instead of requiring a dig through the verifier session.
            reason = (
                "verifier output unparseable (no JSON verdict object found); "
                f"excerpt: {_sanitize_output_excerpt(result_str)}"
            )
            synthetic = [{"ac_text": "(verifier output)", "pass": False, "reason": reason}]
            self._record_ac_results(
                job.id,
                synthetic,
                attempt_num,
                merge_with_existing_attempt=merge_with_existing_attempt,
            )
            return await self._handle_verifier_failure(job, failed_acs=synthetic)
        # Strict True: a string like "false" / "no" is truthy in Python but
        # explicitly NOT a pass. Treat anything other than literal True as failure.
        ac_results_raw = parsed.get("ac_results", [])
        # Coverage guard: a verifier that grades only a subset of the prose ACs
        # must not yield DONE for the ungraded remainder - ungraded is not
        # verified. Pad the missing tail (positional, matching ac_index_map) as
        # failed so the UI shows it and the retry prompt names it.
        if isinstance(ac_results_raw, list) and len(ac_results_raw) < len(prose_entries):
            ac_results_raw = list(ac_results_raw) + [
                {"ac_text": e["ac_text"], "pass": False, "reason": "verifier did not grade this criterion"}
                for e in prose_entries[len(ac_results_raw) :]
            ]
        self._record_ac_results(
            job.id,
            ac_results_raw,
            attempt_num,
            ac_index_map=prose_ac_indices,
            merge_with_existing_attempt=merge_with_existing_attempt,
        )
        # The per-criterion verdicts are the ground truth; `overall_pass` is a
        # derived summary the model can omit or get wrong. Derive it when absent
        # (an all-pass round must not be retried into STUCK on a missing key)
        # and veto a contradictory True when any entry explicitly fails - both
        # directions fail closed. An explicit non-True value (False, "false",
        # etc.) is still honoured as failure.
        overall_pass = parsed.get("overall_pass")
        if isinstance(ac_results_raw, list) and ac_results_raw and overall_pass in (None, True):
            overall_pass = all(isinstance(r, dict) and r.get("pass") is True for r in ac_results_raw)
        if overall_pass is True:
            self._update_latest_attempt(job.id, verifier_pass=True)
            fresh = self._jobs.get(job.id)
            result = dict(fresh.result or {})
            # Snapshot the full ac_results (predicate + verifier merged) for the
            # job.result audit trail. In the prose-only path this is identical
            # to ac_results_raw; in the mixed path it includes predicate verdicts.
            result["ac_results"] = self._snapshot_attempt_results(job.id, attempt_num) or ac_results_raw
            self._finalize(job, JobState.DONE, result=result)
            return
        self._update_latest_attempt(job.id, verifier_pass=False)
        await self._handle_verifier_failure(
            job,
            failed_acs=_extract_failed_acs(ac_results_raw),
        )

    async def _handle_verifier_failure(self, job: Job, failed_acs: list[dict]) -> None:
        # Empty failed_acs gives the retry worker zero signal and the STUCK error
        # zero diagnostic content - synthesize a placeholder so users see something.
        if not failed_acs:
            failed_acs = [
                {
                    "ac_text": "(verifier reported failure with no specific criteria)",
                    "pass": False,
                    "reason": "verifier set overall_pass=false but did not list which AC failed; review the AC list verbatim",
                }
            ]
        followup_prompt = _build_followup_prompt(job, failed_acs)
        new_attempts = job.verify_attempts + 1
        if new_attempts >= job.max_attempts:
            # Ladder: an exhausted verifier budget on this rung escalates to the
            # next model with a fresh budget instead of going STUCK. The failed
            # ACs ride along so the stronger model knows what to fix.
            if await self._escalate(
                job, prompt=followup_prompt, reason="verifier budget exhausted", followup=followup_prompt
            ):
                return
            error_lines = ["Verifier failed after max attempts:"]
            for ac in failed_acs:
                error_lines.append(f"- {ac.get('ac_text', '?')}: {ac.get('reason', '?')}")
            # STUCK preserves the structured ac_results alongside the worker
            # summary so UI/API consumers can render per-AC verdicts (mirrors DONE).
            # Snapshot the full per-attempt ac_results when available so mixed-mode
            # jobs preserve the predicate-pass verdicts alongside the verifier failures.
            fresh = self._jobs.get(job.id)
            snapshot = self._snapshot_attempt_results(job.id, job.verify_attempts + 1)
            result = dict(fresh.result or {})
            result["ac_results"] = snapshot or failed_acs
            self._jobs.update(job.id, verify_attempts=new_attempts, result=result)
            self._finalize(job, JobState.STUCK, error="\n".join(error_lines))
            return

        # Retry: bump counter, spawn new worker, transition verifying → running.
        self._jobs.update(job.id, verify_attempts=new_attempts)
        # fail_worker() acts only on a live job, so leave `verifying` before the
        # executor starts, or its startup failure is dropped and the job is
        # activated on top of a worker that never ran.
        activate_first = job.executor != "agent"
        if activate_first:
            self._activate_worker(job.id, None, kind="retry", timeout_minutes=job.timeout_minutes)
        try:
            started = await self._spawn_worker(
                job,
                prompt=followup_prompt,
                workspace=_job_workspace(job),
                # loop_attempt mirrors verify_attempts (1-indexed retry count).
                extra_metadata={"loop_attempt": new_attempts},
                followup=followup_prompt,
            )
        except Exception as e:
            logger.exception("Failed to spawn retry worker for job '%s': %s", job.id, e)
            self._finalize(job, JobState.ERRORED, error=f"retry worker spawn failed: {e}")
            return
        if not activate_first:
            self._activate_worker(
                job.id, started.id if started else None, kind="retry", timeout_minutes=job.timeout_minutes
            )

    def _finalize(self, job: Job, terminal: JobState, **fields) -> None:
        """Cancel timer, write per-terminal fields, transition, emit tile event.

        Worktrees are pruned on DONE/CANCELLED (clean exit, no inspection value),
        kept on STUCK/ERRORED so the user can see what the worker did wrong.
        Also wakes the parent when `notify_when` matches, and closes the batch
        barrier when this was the parent's last active job.
        """
        self._cancel_timeout(job.id)
        # Transition FIRST: if the Job is already terminal this raises and we
        # bail before touching error/result - a rejected finalize must not
        # clobber the existing diagnostic on disk.
        try:
            self._jobs.update_state(job.id, terminal.value)
        except JobStateTransitionError as e:
            logger.warning("Cannot mark job '%s' %s: %s", job.id, terminal.value, e)
            return
        if fields:
            self._jobs.update(job.id, **fields)
        fresh = self._jobs.get(job.id)
        # Reap the executor's child (e.g. a PTY) on every terminal finalize, or a
        # parked job leaks a live claude. Prune the worktree only on a clean exit;
        # STUCK/ERRORED keep it for inspection and for retry-with-hint to --resume into.
        if fresh:
            prune = terminal in (JobState.DONE, JobState.CANCELLED)
            # worktree_path is cleared by the teardown once the tree is actually
            # gone. Clearing it here would drop the only pointer to a directory a
            # failed prune left behind.
            self._schedule_executor_teardown(fresh, prune_worktree=prune)
        self._emit_job_event(fresh)
        self._reconcile_host_session(job, terminal)
        if fresh and _should_notify(fresh, terminal):
            self._schedule_notify(fresh)
        self.close_batch_barrier(job.parent_session_id)

    async def _cancel_executor(self, job: Job) -> None:
        """Best-effort teardown of a non-agent executor's child. No-op for agent
        jobs and for jobs whose executor plugin isn't loaded (e.g. after restart)."""
        if job is None or job.executor == "agent":
            return
        executor = self._executors.get(job.executor)
        if executor is None:
            return
        try:
            await executor.cancel(job)
        except Exception:
            logger.exception("Executor cancel failed for job '%s'", job.id)

    def _schedule_executor_teardown(self, job: Job, *, prune_worktree: bool = True) -> None:
        """Cancel a non-agent executor's child THEN (on a clean exit) prune the
        worktree, in that order and off the event loop - the child holds the cwd
        open. Parked finalizes (STUCK/ERRORED) pass prune_worktree=False: the child
        is still reaped but the worktree survives for inspection. For an agent job
        (or a job whose executor is gone) this collapses to the plain worktree
        prune, matching the historical behaviour exactly."""
        worktree_path = job.worktree_path if prune_worktree else None
        if job.executor == "agent":
            if worktree_path:
                self._prune_worktree_bg(job.id, worktree_path)
            return

        async def _teardown() -> None:
            await self._cancel_executor(job)
            if worktree_path:
                await self._prune_and_clear(job.id, worktree_path)

        if not self._spawn_bg(_teardown):
            # No loop (sync CLI/test path): the async executor cancel can't run,
            # but the worktree still needs pruning.
            if worktree_path:
                self._prune_and_clear_sync(job.id, worktree_path)

    def _schedule_notify(self, job: Job) -> None:
        """Post a one-line wake-up message into the parent session so its agent
        learns the Job finished. Best-effort: errors are logged, not raised."""
        self._schedule_reply(job, _build_notify_message(job), source="job_complete")

    def close_batch_barrier(self, parent_session_id: str) -> None:
        """Wake the parent once, when the last of its active jobs has finalized.

        Call this in the same synchronous region as the transition that produced
        the terminal state; the remaining-active count means nothing if another
        job can move in between. Only notify_when="all_done" jobs are summarised,
        and a parent still mid-turn may spawn more of the batch, so the barrier
        waits for the turn to end.
        """
        jobs = self.jobs_for_parent(parent_session_id)
        if any(j.state not in self._jobs.terminal_states for j in jobs):
            return
        if self._parent_turn_in_flight(parent_session_id):
            return
        batch = [j for j in jobs if j.notify_when == "all_done" and not j.barrier_notified]
        if not batch:
            return
        batch.sort(key=lambda j: j.created_at or "")

        def mark_batch_notified() -> None:
            for member in batch:
                self._jobs.update(member.id, barrier_notified=True)

        self._schedule_reply(
            batch[-1],
            _build_barrier_message(batch),
            source="jobs_all_complete",
            metadata={"kind": "jobs_barrier"},
            on_delivered=mark_batch_notified,
        )

    def _parent_turn_in_flight(self, parent_session_id: str) -> bool:
        parent = self._get_parent_session(parent_session_id)
        return bool(parent and parent.turn_in_flight)

    def _schedule_reply(
        self,
        job: Job,
        message: str,
        *,
        source: str,
        metadata: Optional[dict] = None,
        on_delivered: Optional[Callable[[], None]] = None,
    ) -> None:
        """Queue a wake-up for the parent.

        `on_delivered` runs once the parent has actually taken the turn, so a
        caller can record the wake-up only when it landed. Best-effort: errors are
        logged, not raised.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("No running loop; skipping %s wake-up for job '%s'", source, job.id)
            return
        metadata = metadata or {"job_id": job.id, "kind": "job_notify"}

        async def _send():
            try:
                reply = await self._runner.reply_to_session(
                    job.parent_session_id,
                    message,
                    source=source,
                    metadata=metadata,
                )
            except Exception:
                logger.exception("Failed to wake parent of job '%s' (%s)", job.id, source)
                return
            if reply is None:
                logger.warning(
                    "Dropped the %s wake-up for job '%s': parent session '%s' has finished",
                    source,
                    job.id,
                    job.parent_session_id,
                )
                return
            if on_delivered:
                on_delivered()

        task = loop.create_task(_send())
        self._notify_tasks.add(task)
        task.add_done_callback(self._notify_tasks.discard)

    def _spawn_bg(self, make_coro) -> bool:
        """Run `make_coro()` as a tracked background task, so it can't be garbage
        collected mid-flight. Returns False when there is no running loop (sync
        CLI/test context) and the caller must do the work inline instead."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        task = loop.create_task(make_coro())
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return True

    def _prune_and_clear_sync(self, job_id: str, worktree_path: str) -> None:
        """Prune, then drop the path only once the tree is confirmed gone. A failed
        prune keeps it, so the leftover directory stays visible for a later cleanup."""
        if not _prune_worktree(worktree_path):
            return
        try:
            self._jobs.update(job_id, worktree_path=None)
        except KeyError:
            # The job aged out of the store while the prune ran; nothing to clear.
            pass

    async def _prune_and_clear(self, job_id: str, worktree_path: str) -> None:
        """`_prune_and_clear_sync` off the event loop.

        `git worktree remove` shells out and can stall on a large tree or a busy
        `.git` lock; running it inline on the daemon's single loop would freeze
        every other session, SSE stream, and timer until it returns.
        """
        await asyncio.to_thread(self._prune_and_clear_sync, job_id, worktree_path)

    def _prune_worktree_bg(self, job_id: str, worktree_path: str) -> None:
        """Prune a worktree off the loop, falling back to an inline prune when no
        loop is running (sync CLI/test context) - there's nothing to block there."""
        if not self._spawn_bg(lambda: self._prune_and_clear(job_id, worktree_path)):
            self._prune_and_clear_sync(job_id, worktree_path)

    def _schedule_timeout(self, job_id: str, timeout_minutes: int) -> None:
        self._cancel_timeout(job_id)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("No running loop; skipping timeout schedule for job '%s'", job_id)
            return
        handle = loop.call_later(max(timeout_minutes, 1) * 60, self._on_timeout, job_id)
        self._timeout_handles[job_id] = handle

    def _cancel_timeout(self, job_id: str) -> None:
        handle = self._timeout_handles.pop(job_id, None)
        if handle:
            handle.cancel()

    def _on_timeout(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None or job.state in _TERMINAL_STATES:
            return
        # Cancel whichever sessions are still live. Both can leak: the worker if
        # the timeout fires during the RUNNING phase, the verifier if it hangs
        # during VERIFYING after the worker has already completed.
        for sid in (job.worker_session_id, job.verifier_session_id):
            if sid and not self._session_already_terminal(sid):
                try:
                    self._runner.cancel_session(sid)
                except Exception:
                    logger.exception("Timeout: failed to cancel session '%s'", sid)
        self._finalize(job, JobState.STUCK, error=f"timeout after {job.timeout_minutes} minutes")

    def _session_already_terminal(self, session_id: str) -> bool:
        """Race guard: if the session already reached a terminal status, don't
        overwrite it via cancel_session - the session won. Tolerates both the
        FakeStore-style None-on-miss contract and real SessionStore's
        ValueError-on-miss; treats either as 'session is gone, nothing to cancel'."""
        store = getattr(self._runner, "store", None)
        if store is None or not hasattr(store, "get_session"):
            return False
        try:
            session = store.get_session(session_id)
        except (ValueError, KeyError):
            return True
        if session is None:
            return False
        return session.status in (
            SessionStatus.COMPLETED.value,
            SessionStatus.FAILED.value,
            SessionStatus.CANCELLED.value,
        )

    def jobs_for_parent(self, session_id: str) -> list[Job]:
        """Every job belonging to `session_id`'s conversation. A job keeps the id
        its parent had when it was spawned and compaction rotates that chat onto a
        successor, so both sides of the match resolve through the chain."""
        live = self._runner.live_id
        target = live(session_id)
        return [j for j in self._jobs.list_all() if live(j.parent_session_id) == target]

    def _get_parent_session(self, parent_session_id: str) -> Optional[Session]:
        """Fetch a session, tolerating both the FakeStore None-on-miss contract and a
        real SessionStore raising ValueError/KeyError. Returns None if the runner has
        no store or the session is gone."""
        store = getattr(self._runner, "store", None)
        if store is None or not hasattr(store, "get_session"):
            return None
        try:
            return store.get_session(self._runner.live_id(parent_session_id))
        except (ValueError, KeyError):
            return None

    def _reconcile_host_session(self, job: Job, terminal: JobState) -> bool:
        """Close a Job's placeholder host session when the Job reaches a terminal state.

        A /job launched outside any chat is given a dedicated placeholder session that is
        the job's parent_session_id and is tagged METADATA_JOB_HOST; it otherwise sits
        `active` forever and renders as "starting" in the sidebar. When the Job finishes,
        transition that placeholder to a matching terminal status. Only touches sessions
        carrying the job_host marker, so a /job typed inside a real conversation never
        closes the user's chat. Returns True if it reconciled a session.
        """
        parent = self._get_parent_session(job.parent_session_id)
        if parent is None or not parent.metadata.get(METADATA_JOB_HOST) or parent.status in FINISHED_STATUSES:
            return False
        store = getattr(self._runner, "store", None)
        if not hasattr(store, "update_session"):
            return False
        target = _JOB_TO_SESSION_STATUS[terminal].value
        try:
            store.update_session(parent.id, status=target)
        except Exception:
            logger.exception("Failed to reconcile host session '%s' for job '%s'", parent.id, job.id)
            return False
        if self._event_bus:
            try:
                self._event_bus.emit("session_update", {"action": "updated", "id": parent.id})
            except Exception:
                logger.debug("session_update emit failed for host session '%s'", parent.id)
        return True

    def reconcile_orphaned_host_sessions(self) -> int:
        """On daemon load, close any lingering job-host placeholder session whose Job is
        already terminal. Runs after recover_orphaned_jobs so jobs it just errored out
        also get their host session reconciled. Placeholders created before this feature
        shipped lack the job_host marker and are left alone."""
        reconciled = sum(
            self._reconcile_host_session(job, JobState(job.state))
            for job in self._jobs.list_all()
            if job.state in _TERMINAL_STATES
        )
        if reconciled:
            logger.info("Reconciled %d orphaned job-host session(s) from previous daemon run", reconciled)
        return reconciled

    def _runtime(self):
        """The daemon's runtime defaults, or None when the runner has no adapter."""
        return getattr(self._runner, "runtime", None)

    def _resolve_parent_sandbox_override(self, parent_session_id: str) -> Optional[dict]:
        """Resolve the daemon's sandbox config as an override dict, or None when
        sandboxing is off. Lets /job-created jobs inherit the sandbox even though
        there's no running-agent context."""
        sb = getattr(self._runtime(), "sandbox", None)
        if sb is None or not sb.enabled:
            return None
        return sb.model_dump(mode="json")

    def _resolve_workspace_root(self, parent_session_id: str) -> Optional[Path]:
        """Workspace root a relative --repo is interpreted against: the parent
        session's workspace_override, else the daemon's configured workspace_dir.

        Returns None when nothing is resolvable, so a relative path falls back to
        the daemon CWD. Tolerates a real SessionStore raising ValueError on a
        missing session.
        """
        parent = self._get_parent_session(parent_session_id)
        if parent is not None and getattr(parent, "workspace_override", None):
            return Path(parent.workspace_override)
        workspace_dir = getattr(self._runtime(), "workspace_dir", None)
        return Path(workspace_dir) if workspace_dir is not None else None

    def _emit_job_event(self, job: Optional[Job]) -> None:
        if job is None:
            return
        self._sync_attention(job)
        payload = job.to_payload()
        # payload already carries worker_terminal_id from the Job field, which a
        # non-agent executor (PTY-driven) stamps directly. Only fall back to the
        # terminal_store lookup for agent jobs that spawn a PTY without stamping
        # the field - and skip it entirely when the field is already set.
        if not payload.get("worker_terminal_id") and self._terminal_store and job.worker_session_id:
            try:
                terms = self._terminal_store.list_for_parent(job.worker_session_id)
                if terms:
                    payload["worker_terminal_id"] = terms[0].id
            except Exception:
                logger.debug("Worker-terminal lookup failed for job '%s'", job.id)
        # Persist into parent session JSONL so a page reload re-renders the tile.
        try:
            self._runner.store.append_event(
                self._runner.live_id(job.parent_session_id), {"type": "job_status", **payload}
            )
        except Exception:
            logger.exception("Failed to persist job_status event for job '%s'", job.id)
        if self._event_bus:
            try:
                self._event_bus.emit("job_update", payload)
            except Exception:
                logger.exception("Failed to broadcast job_update for job '%s'", job.id)

    def reconcile_attention(self) -> None:
        """Match every job's record to its state at daemon start.

        A parked job emits nothing until someone acts on it, so a record opened or
        left over by a previous process is only reachable from here.
        """
        for job in self._jobs.list_all():
            self._sync_attention(job)

    def _sync_attention(self, job: Job) -> None:
        """Match the parent session's record for this job to the job's state."""
        if job.state in _PARKED_STATES:
            self._runner.open_attention(
                job.parent_session_id,
                source=SOURCE_JOB,
                ref_id=job.id,
                kind=job.state,
            )
        else:
            self._runner.clear_attention_ref(SOURCE_JOB, job.id)


def _should_notify(job: Job, terminal: JobState) -> bool:
    """Decide whether to wake the parent based on Job.notify_when.

    Recognised values: "done", "stuck", "errored", "terminal" (any terminal state),
    "never" (no-op), and "all_done" (no per-job wake-up; the job is reported by the
    batch barrier instead). Anything else is treated as "never" (defensive - a typo
    on disk shouldn't spam the parent agent).
    """
    # Job.__post_init__ already maps legacy `notify=True` → notify_when="terminal"
    # and normalises a missing value to "never", so this is a plain attribute read.
    notify_when = job.notify_when
    state = terminal.value if isinstance(terminal, JobState) else terminal
    if notify_when in ("never", "all_done"):
        return False
    if notify_when == "terminal":
        return state in _TERMINAL_STATES
    return notify_when == state


def render_jobs_context_xml(jobs: list[Job], recent_limit: int = 3) -> str:
    """Render an XML block describing `jobs`, suitable for inclusion in
    `<message_context>` so the LLM is aware of what's running and what just
    finished without dumping full worker output into the chat.

    Returns "" when there are no jobs (so the surrounding template can omit the
    section entirely without empty whitespace).
    """
    if not jobs:
        return ""

    active_states = {JobState.QUEUED.value, JobState.RUNNING.value, JobState.VERIFYING.value}

    active = [j for j in jobs if j.state in active_states]
    recent = [j for j in jobs if j.state in _TERMINAL_STATES]
    recent.sort(key=lambda j: j.resolved_at or j.updated_at or "", reverse=True)
    recent = recent[: max(0, int(recent_limit))]

    if not active and not recent:
        return ""

    def _attrs(job: Job, is_active: bool) -> str:
        prompt_short = (job.prompt or "")[:80]
        if len(job.prompt or "") > 80:
            prompt_short += "…"
        ts_key = "created_at" if is_active else "resolved_at"
        ts_val = job.created_at if is_active else (job.resolved_at or job.updated_at)
        parts = [
            f"id={quoteattr(job.id)}",
            f"state={quoteattr(job.state)}",
            f"prompt={quoteattr(prompt_short)}",
            f"{ts_key}={quoteattr(ts_val or '')}",
        ]
        if job.worker_session_id:
            parts.append(f"worker_session_id={quoteattr(job.worker_session_id)}")
        if job.verifier_session_id:
            parts.append(f"verifier_session_id={quoteattr(job.verifier_session_id)}")
        if job.verify_attempts:
            parts.append(f'verify_attempts="{job.verify_attempts}"')
        if job.error and job.state in (JobState.STUCK.value, JobState.ERRORED.value):
            parts.append(f"error={quoteattr(job.error.splitlines()[0][:200])}")
        return " ".join(parts)

    lines = ["  <jobs>"]
    if active:
        lines.append("    <active>")
        for j in active:
            lines.append(f"      <job {_attrs(j, is_active=True)} />")
        lines.append("    </active>")
    if recent:
        lines.append("    <recent>")
        for j in recent:
            lines.append(f"      <job {_attrs(j, is_active=False)} />")
        lines.append("    </recent>")
    lines.append("  </jobs>")
    return "\n".join(lines)


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
