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
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from tsugite_daemon.job_store import Job, JobState, JobStateTransitionError, JobStore
from tsugite_daemon.session_store import Session, SessionSource, SessionStatus

logger = logging.getLogger(__name__)

MAX_VERIFY_ATTEMPTS = 3
VERIFIER_AGENT = "job_verifier"
WORKER_AGENT = "job_worker"

# Allowed notify_when values; anything else is coerced to "never" at intake.
_VALID_NOTIFY_WHEN = frozenset({"done", "stuck", "errored", "terminal", "never"})

# Hard cap per predicate. Subprocess hang shouldn't strand a Job; the per-phase
# Job.timeout_minutes timer covers the verification phase too, so even if all
# predicates take the full 30s the phase timeout will reap any longer hang.
_PREDICATE_TIMEOUT_SECONDS = 30
# Stderr snippet length surfaced in the failure reason.
_PREDICATE_STDERR_TRUNCATE = 100


def _with_sandbox(job: Job, metadata: dict) -> dict:
    """Carry the job's inherited sandbox policy onto a spawned session's metadata
    so worker/verifier runs stay sandboxed (resolved at the adapter chokepoint)."""
    if job.sandbox_override:
        return {**metadata, "sandbox_override": job.sandbox_override}
    return metadata


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
        return render_jobs_context_xml(self._jobs, session_id, recent_limit)

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
        agent: Optional[str] = None,
        timeout_minutes: int = 30,
        spawned_by: str = "user-slash",
        max_attempts: Optional[int] = None,
        notify_when: Optional[str] = None,
        sandbox_override: Optional[dict] = None,
    ) -> tuple[Job, Session]:
        """Create a Job record + spawn the worker session in one step.

        Used by both the /job slash command and the spawn_job() agent tool.
        Returns (job, started_worker_session). The orchestrator's register_worker
        is called automatically so the timeout is scheduled and the tile event fires.

        notify_when: one of "done", "stuck", "errored", "terminal", "never" (default).
        max_attempts: verifier-loop cap. Defaults to 3 when omitted.
        """
        worker_agent_file = agent or WORKER_AGENT

        # If the spawner didn't supply an inherited policy (the /job slash command
        # path, vs the spawn_job tool from a sandboxed agent), fall back to the
        # parent agent's own sandbox config so jobs are sandboxed whenever that
        # agent is - the worker/verifier sessions and predicate evaluation all key
        # off job.sandbox_override.
        if sandbox_override is None:
            sandbox_override = self._resolve_parent_sandbox_override(parent_session_id)

        job_kwargs: dict = {}
        if max_attempts is not None:
            job_kwargs["max_attempts"] = max_attempts
        if notify_when:
            if notify_when not in _VALID_NOTIFY_WHEN:
                logger.warning("Unknown notify_when '%s'; coercing to 'never'", notify_when)
                notify_when = "never"
            job_kwargs["notify_when"] = notify_when

        job = self._jobs.add(
            Job(
                id="",
                parent_session_id=parent_session_id,
                prompt=prompt,
                acceptance_criteria=acceptance_criteria or [],
                repo=repo,
                model=model,
                agent=worker_agent_file,
                timeout_minutes=timeout_minutes,
                spawned_by=spawned_by,
                sandbox_override=sandbox_override,
                **job_kwargs,
            )
        )

        # Provision a fresh git worktree if --repo was given so the worker has
        # an isolated working tree (no clashes with the parent shell or other jobs).
        worktree_path: Optional[str] = None
        if repo:
            try:
                # Offload the blocking `git worktree add` so a slow clone/checkout
                # can't stall the daemon's single event loop (the retry/prune paths
                # already do this). A relative --repo resolves against the parent
                # session's workspace, not the daemon CWD.
                workspace_root = self._resolve_workspace_root(parent_session_id)
                worktree_path = await asyncio.to_thread(_provision_worktree, repo, job.id, workspace_root)
                self._jobs.update(job.id, worktree_path=worktree_path)
            except Exception as e:
                logger.exception("Failed to provision worktree for job '%s': %s", job.id, e)
                self._finalize(job, JobState.ERRORED, error=f"worktree provisioning failed: {e}")
                raise

        worker_prompt = build_worker_prompt(prompt, acceptance_criteria or [], repo)
        try:
            started = self._spawn_worker(job, prompt=worker_prompt, workspace=worktree_path)
        except Exception as e:
            # Don't leave the Job persisted in QUEUED; mark it ERRORED so it
            # doesn't accumulate as a zombie in jobs.json.
            self._finalize(job, JobState.ERRORED, error=f"worker spawn failed: {e}")
            raise
        self.register_worker(job.id, started.id, timeout_minutes=timeout_minutes)
        return job, started

    def _spawn_worker(
        self, job: Job, *, prompt: str, workspace: Optional[str], extra_metadata: Optional[dict] = None
    ) -> Session:
        """Build + start a worker session for `job` - the single spawn path for
        initial, verifier-rejected retry, and hint-retry workers. Raises on spawn
        failure; callers decide whether that finalizes the Job or surfaces to the user."""
        session = Session(
            id="",
            agent=self._resolve_adapter_key(job.parent_session_id),
            source=SessionSource.SPAWNED.value,
            prompt=prompt,
            agent_file=job.agent or WORKER_AGENT,
            model=job.model,
            workspace_override=workspace,
            metadata=_with_sandbox(job, {"job_id": job.id, **(extra_metadata or {})}),
        )
        return self._runner.start_session(session)

    def _activate_worker(
        self, job_id: str, worker_session_id: str, *, kind: str, timeout_minutes: int, clear_error: bool = False
    ) -> None:
        """Common bookkeeping after a retry/hint worker spawn: attempt entry,
        RUNNING transition, tile event, fresh phase timer."""
        fields: dict = {"worker_session_id": worker_session_id}
        if clear_error:
            fields.update(resolved_at=None, error=None)
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

    def register_worker(self, job_id: str, worker_session_id: str, timeout_minutes: int) -> None:
        """Record the worker session id and schedule the wall-clock timeout."""
        self._jobs.update(job_id, worker_session_id=worker_session_id)
        # First call to register_worker for this Job - record the initial attempt.
        # Retry / hint paths bypass register_worker and call _append_attempt directly.
        job = self._jobs.get(job_id)
        if job is not None and not (job.attempts or []):
            self._append_attempt(job_id, kind="initial", worker_session_id=worker_session_id)
        try:
            self._jobs.update_state(job_id, JobState.RUNNING.value)
        except JobStateTransitionError:
            # If we're being called again on a retry, the Job is already in RUNNING.
            pass
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
        """User-initiated cancel from the tile. No-op on terminal Jobs."""
        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(f"Unknown job: {job_id}")
        if job.state in _TERMINAL_JOB_STATES:
            return job
        for sid in (job.worker_session_id, job.verifier_session_id):
            if sid and not self._session_already_terminal(sid):
                try:
                    self._runner.cancel_session(sid)
                except Exception:
                    logger.exception("cancel_job: failed to cancel session '%s'", sid)
        self._finalize(job, JobState.CANCELLED, error=reason)
        return self._jobs.get(job_id)

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
    ) -> Job:
        """Give a STUCK or ERRORED Job one more shot, with the user's hint as the worker prompt.

        Args:
            job_id: STUCK/ERRORED job to resurrect.
            hint: Free-text guidance for the new worker.
            reset_counter: Zero out `verify_attempts` so the retry gets a full new
                budget of verifier rounds (the UI exposes this as "reset to 1").
                Defaults False to preserve the historical no-infinite-loops guard.
            fresh_workspace: When the job has a `repo` worktree, prune the existing
                tree and recreate it from HEAD before spawning. No-op when the Job
                was created without `repo`.
        """
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
                    self._jobs.update(job_id, worktree_path=worktree_path)
                except Exception as e:
                    logger.exception("retry_with_hint: fresh_workspace failed for job '%s': %s", job_id, e)
                    raise ValueError(f"failed to recreate worktree: {e}") from e
            elif worktree_path and not Path(worktree_path).is_dir():
                # Worktree was hand-deleted between STUCK and retry - refuse rather
                # than spawn a worker into a missing directory.
                raise ValueError(
                    f"retry_with_hint: worktree at '{worktree_path}' no longer exists; cannot resume in a missing directory"
                )

            if reset_counter:
                self._jobs.update(job_id, verify_attempts=0)
                job = self._jobs.get(job_id)

            try:
                started = self._spawn_worker(
                    job,
                    prompt=_build_hint_prompt(job, hint),
                    workspace=worktree_path,
                    extra_metadata={"hint_attempt": True},
                )
            except Exception as e:
                logger.exception("retry_with_hint: failed to spawn worker for job '%s': %s", job_id, e)
                raise ValueError(f"failed to spawn retry worker: {e}") from e
            self._activate_worker(
                job_id, started.id, kind="hint", timeout_minutes=job.timeout_minutes, clear_error=True
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

        if is_verifier and session_failed:
            # Verifier crashed or got cancelled - infrastructure failure, NOT a
            # verdict against the worker. Don't burn a verify attempt.
            reason = session.error or result_str or f"verifier session ended with status '{session.status}'"
            self._finalize(job, JobState.ERRORED, error=f"verifier infra failure: {reason}")
        elif is_verifier:
            await self._handle_verifier_complete(job, session, result_str)
        elif session_failed:
            await self._handle_worker_failed(job, session, result_str)
        else:
            await self._handle_worker_complete(job, session, result_str)

    async def _handle_worker_complete(self, job: Job, worker: Session, result_str: str) -> None:
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
        verifier_adapter_key = self._resolve_adapter_key(job.parent_session_id)
        verifier_session = Session(
            id="",
            agent=verifier_adapter_key,
            source=SessionSource.SPAWNED.value,
            prompt=verifier_prompt,
            agent_file=VERIFIER_AGENT,
            # Same worktree as the worker - the verifier inspects `git diff` /
            # files, which only exist there for repo jobs.
            workspace_override=job.worktree_path,
            metadata=_with_sandbox(job, {"job_id": job.id, "verifier_for": worker.id}),
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

    async def _handle_worker_failed(self, job: Job, worker: Session, result_str: str) -> None:
        # Guard: mirrors _handle_worker_complete. A late failure/cancel notify for
        # a Job already finalized (e.g. _on_timeout marked it STUCK then cancelled
        # the worker, whose CANCELLED notify lands here) must not touch the record.
        if job.state != JobState.RUNNING.value:
            logger.warning(
                "Worker failure for job '%s' in state '%s' (expected RUNNING) - ignoring",
                job.id,
                job.state,
            )
            return
        self._cancel_timeout(job.id)
        # User-initiated cancellation vs genuine failure - terminal state reflects intent.
        if worker.status == SessionStatus.CANCELLED.value:
            reason = worker.error or result_str or "worker session was cancelled"
            self._finalize(job, JobState.CANCELLED, error=reason)
            return
        reason = worker.error or result_str or f"worker session ended with status '{worker.status}'"
        self._finalize(job, JobState.ERRORED, error=reason)

    async def _handle_verifier_complete(self, job: Job, verifier: Session, result_str: str) -> None:
        # State guard: a duplicate or out-of-order verifier-complete for a Job
        # that's no longer in VERIFYING must NOT advance the state again.
        # Without this, a stale notify could spawn a second concurrent retry worker.
        if job.state != JobState.VERIFYING.value:
            logger.warning(
                "Verifier completion for job '%s' in state '%s' (expected VERIFYING) - ignoring",
                job.id,
                job.state,
            )
            return

        # Attempt counter is 1-indexed (matches the UX of "attempt #N"). Captured
        # before _handle_verifier_failure bumps job.verify_attempts.
        attempt_num = job.verify_attempts + 1

        # Recompute predicate / prose partition so the verifier's positional
        # ac_results can be mapped back to original AC indices. When the Job has
        # NO predicates this collapses to the legacy 1-to-1 mapping (None map +
        # no merge), preserving backward compatibility for prose-only Jobs.
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
            # Synthetic one-entry batch so the UI sees *something* for this attempt.
            synthetic = [{"ac_text": "(verifier output)", "pass": False, "reason": "verifier output unparseable"}]
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
        self._record_ac_results(
            job.id,
            ac_results_raw,
            attempt_num,
            ac_index_map=prose_ac_indices,
            merge_with_existing_attempt=merge_with_existing_attempt,
        )
        # `response_format: json_object` forces valid JSON, not a complete schema -
        # a verifier can list all-pass criteria yet omit the `overall_pass` summary
        # key. When it's absent, derive the verdict from the criteria so an all-pass
        # round isn't wrongly retried into STUCK. An explicit non-True value (False,
        # "false", etc.) is still honoured as failure above.
        overall_pass = parsed.get("overall_pass")
        if overall_pass is None and isinstance(ac_results_raw, list) and ac_results_raw:
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
        new_attempts = job.verify_attempts + 1
        if new_attempts >= job.max_attempts:
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
        try:
            started = self._spawn_worker(
                job,
                prompt=_build_followup_prompt(job, failed_acs),
                workspace=job.worktree_path,
                # loop_attempt mirrors verify_attempts (1-indexed retry count).
                extra_metadata={"loop_attempt": new_attempts},
            )
        except Exception as e:
            logger.exception("Failed to spawn retry worker for job '%s': %s", job.id, e)
            self._finalize(job, JobState.ERRORED, error=f"retry worker spawn failed: {e}")
            return
        self._activate_worker(job.id, started.id, kind="retry", timeout_minutes=job.timeout_minutes)

    def _finalize(self, job: Job, terminal: JobState, **fields) -> None:
        """Cancel timer, write per-terminal fields, transition, emit tile event.

        Worktrees are pruned on DONE/CANCELLED (clean exit, no inspection value),
        kept on STUCK/ERRORED so the user can see what the worker did wrong.
        If job.notify is True, also schedules a wake-up reply to the parent session.
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
        if terminal in (JobState.DONE, JobState.CANCELLED) and fresh and fresh.worktree_path:
            self._prune_worktree_bg(fresh.worktree_path)
            self._jobs.update(job.id, worktree_path=None)
            fresh = self._jobs.get(job.id)
        self._emit_job_event(fresh)
        if fresh and _should_notify(fresh, terminal):
            self._schedule_notify(fresh)

    def _schedule_notify(self, job: Job) -> None:
        """Post a one-line wake-up message into the parent session so its agent
        learns the Job finished. Best-effort: errors are logged, not raised."""
        message = _build_notify_message(job)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("No running loop; skipping notify for job '%s'", job.id)
            return

        async def _send():
            try:
                await self._runner.reply_to_session(
                    job.parent_session_id,
                    message,
                    source="job_complete",
                    metadata={"job_id": job.id, "kind": "job_notify"},
                )
            except Exception:
                logger.exception("Failed to notify parent of job '%s' completion", job.id)

        task = loop.create_task(_send())
        self._notify_tasks.add(task)
        task.add_done_callback(self._notify_tasks.discard)

    def _prune_worktree_bg(self, worktree_path: str) -> None:
        """Remove a worktree off the event loop.

        `git worktree remove` shells out and can stall on a large tree or a busy
        `.git` lock; running it inline on the daemon's single loop would freeze
        every other session, SSE stream, and timer until it returns. Fire-and-
        forget: prune failures are logged inside `_prune_worktree`, and the
        caller clears `worktree_path` regardless, so a dropped task only leaves
        disk for a later `git worktree prune` to reap.

        Falls back to an inline prune when no loop is running (sync CLI/test
        context) - there's nothing to block there.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _prune_worktree(worktree_path)
            return
        task = loop.create_task(asyncio.to_thread(_prune_worktree, worktree_path))
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

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
        if job is None or job.state in _TERMINAL_JOB_STATES:
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

    def _get_parent_session(self, parent_session_id: str) -> Optional[Session]:
        """Fetch a session, tolerating both the FakeStore None-on-miss contract and a
        real SessionStore raising ValueError/KeyError. Returns None if the runner has
        no store or the session is gone."""
        store = getattr(self._runner, "store", None)
        if store is None or not hasattr(store, "get_session"):
            return None
        try:
            return store.get_session(parent_session_id)
        except (ValueError, KeyError):
            return None

    def _resolve_adapter_key(self, parent_session_id: str) -> str:
        """Route Job spawns through the parent session's adapter so jobs stay on
        the same agent (credentials, tools, model defaults) as the chat that spawned them.
        Tolerates real SessionStore raising ValueError on missing sessions.
        """
        parent = self._get_parent_session(parent_session_id)
        if parent is not None:
            return parent.agent
        adapters = getattr(self._runner, "_adapters", {})
        if adapters:
            return next(iter(adapters))
        return "default"

    def _resolve_parent_sandbox_override(self, parent_session_id: str) -> Optional[dict]:
        """Resolve the parent session's agent sandbox config as an override dict,
        or None when that agent isn't sandboxed. Lets /job-created jobs inherit
        the agent's sandbox even though there's no running-agent context."""
        parent = self._get_parent_session(parent_session_id)
        adapters = getattr(self._runner, "_adapters", {})
        adapter = adapters.get(parent.agent) if parent is not None else None
        if adapter is None and adapters:
            adapter = next(iter(adapters.values()))
        sb = getattr(getattr(adapter, "agent_config", None), "sandbox", None)
        if sb is None or not sb.enabled:
            return None
        return {
            "enabled": True,
            "no_network": sb.no_network,
            "allow_domains": list(sb.allow_domains),
            "extra_ro_binds": [str(p) for p in sb.extra_ro_binds],
            "extra_rw_binds": [str(p) for p in sb.extra_rw_binds],
            "pass_env": list(getattr(sb, "pass_env", [])),
        }

    def _resolve_workspace_root(self, parent_session_id: str) -> Optional[Path]:
        """Workspace root a relative --repo is interpreted against: the parent
        session's workspace_override, else its adapter's configured workspace_dir.

        Returns None when nothing is resolvable, so a relative path falls back to
        the daemon CWD (prior behaviour). Mirrors _resolve_adapter_key's tolerance
        of a real SessionStore raising ValueError on a missing session.
        """
        parent = self._get_parent_session(parent_session_id)
        if parent is not None and getattr(parent, "workspace_override", None):
            return Path(parent.workspace_override)
        adapters = getattr(self._runner, "_adapters", {})
        adapter = adapters.get(parent.agent) if parent is not None else None
        if adapter is None and adapters:
            adapter = next(iter(adapters.values()))
        agent_config = getattr(adapter, "agent_config", None)
        workspace_dir = getattr(agent_config, "workspace_dir", None)
        return Path(workspace_dir) if workspace_dir is not None else None

    def _emit_job_event(self, job: Optional[Job]) -> None:
        if job is None:
            return
        payload = job.to_payload()
        # If a terminal_store is wired, resolve the worker's PTY (if any) so the
        # tile can mount its embedded xterm without a separate /api/terminals
        # round-trip per tile. None for LLM-only jobs that never spawn a PTY.
        if self._terminal_store and job.worker_session_id:
            try:
                terms = self._terminal_store.list_for_parent(job.worker_session_id)
                if terms:
                    payload["worker_terminal_id"] = terms[0].id
            except Exception:
                logger.debug("Worker-terminal lookup failed for job '%s'", job.id)
        # Persist into parent session JSONL so a page reload re-renders the tile.
        try:
            self._runner.store.append_event(job.parent_session_id, {"type": "job_status", **payload})
        except Exception:
            logger.exception("Failed to persist job_status event for job '%s'", job.id)
        if self._event_bus:
            try:
                self._event_bus.emit("job_update", payload)
            except Exception:
                logger.exception("Failed to broadcast job_update for job '%s'", job.id)


_TERMINAL_JOB_STATES = frozenset(
    {JobState.DONE.value, JobState.STUCK.value, JobState.CANCELLED.value, JobState.ERRORED.value}
)


def _should_notify(job: Job, terminal: JobState) -> bool:
    """Decide whether to wake the parent based on Job.notify_when.

    Recognised values: "done", "stuck", "errored", "terminal" (any terminal state),
    "never" (no-op). Anything else is treated as "never" (defensive - a typo on
    disk shouldn't spam the parent agent).
    """
    # Job.__post_init__ already maps legacy `notify=True` → notify_when="terminal"
    # and normalises a missing value to "never", so this is a plain attribute read.
    notify_when = job.notify_when
    state = terminal.value if isinstance(terminal, JobState) else terminal
    if notify_when == "never":
        return False
    if notify_when == "terminal":
        return state in _TERMINAL_JOB_STATES
    return notify_when == state


def build_worker_prompt(prompt: str, acceptance_criteria: list, repo: Optional[str]) -> str:
    """Compose the worker's initial user_prompt with AC and repo context inlined.

    Accepts AC as either legacy list[str] or normalised list[dict]; the worker
    prompt only needs the text, so both shapes render the same.
    """
    parts = [prompt]
    if acceptance_criteria:
        parts.append("")
        parts.append("Acceptance criteria (the verifier will grade your work against these):")
        for i, ac in enumerate(acceptance_criteria, 1):
            text = ac.get("text", "") if isinstance(ac, dict) else ac
            parts.append(f"{i}. {text}")
    if repo:
        parts.append("")
        parts.append(f"Working in repo: {repo}")
    return "\n".join(parts)


def _extract_failed_acs(ac_results) -> list[dict]:
    """Return failing-AC dicts from the verifier's ac_results list.

    Tolerates malformed entries (string, None, non-dict): each non-dict element
    becomes a synthetic failed AC so the orchestrator never crashes and the user
    sees a useful retry / stuck reason.
    """
    out: list[dict] = []
    if not isinstance(ac_results, list):
        return out
    for item in ac_results:
        if isinstance(item, dict):
            if not item.get("pass"):
                out.append(item)
        else:
            out.append({"ac_text": "(malformed verifier output)", "pass": False, "reason": repr(item)})
    return out


def _build_verifier_prompt(job: Job, worker_output: str, prose_acs: Optional[list[str]] = None) -> str:
    """Build the verifier prompt.

    If `prose_acs` is provided, only those ACs land in the prompt - predicate
    ACs are evaluated locally and must not be sent to the LLM (the verifier has
    no business grading something that's already been mechanically decided).
    """
    acs_to_render = prose_acs if prose_acs is not None else job.acceptance_criteria
    parts = ["Acceptance criteria:"]
    for i, ac in enumerate(acs_to_render, 1):
        parts.append(f"{i}. {ac}")
    parts.append("")
    parts.append("Worker output:")
    parts.append(worker_output.strip() or "(empty)")
    if job.repo:
        parts.append("")
        parts.append(f"Repo: {job.repo}")
        parts.append("(use `run` to inspect `git diff` or `git log` if relevant.)")
    return "\n".join(parts)


def _build_hint_prompt(job: Job, hint: str) -> str:
    """Compose the retry-with-hint prompt for a worker resurrected from STUCK."""
    parts = [
        "This job previously hit the verifier's max-attempt limit. A user has provided a hint:",
        "",
        hint,
        "",
        "Address the hint and produce a new structured summary in the same shape as before.",
        "",
        "Acceptance criteria the verifier will check:",
    ]
    for i, ac in enumerate(job.acceptance_criteria, 1):
        parts.append(f"{i}. {ac}")
    return "\n".join(parts)


def render_jobs_context_xml(job_store: JobStore, session_id: str, recent_limit: int = 3) -> str:
    """Render an XML block describing Jobs anchored on `session_id`, suitable for
    inclusion in `<message_context>` so the LLM is aware of what's running and
    what just finished without dumping full worker output into the chat.

    Returns "" when there are no jobs (so the surrounding template can omit the
    section entirely without empty whitespace).
    """
    if not session_id:
        return ""
    try:
        jobs = job_store.list_for_parent(session_id)
    except Exception:
        return ""
    if not jobs:
        return ""

    active_states = {JobState.QUEUED.value, JobState.RUNNING.value, JobState.VERIFYING.value}

    active = [j for j in jobs if j.state in active_states]
    recent = [j for j in jobs if j.state in _TERMINAL_JOB_STATES]
    recent.sort(key=lambda j: j.resolved_at or j.updated_at or "", reverse=True)
    recent = recent[: max(0, int(recent_limit))]

    if not active and not recent:
        return ""

    def _attrs(job: Job, is_active: bool) -> str:
        prompt_short = (job.prompt or "")[:80]
        if len(job.prompt or "") > 80:
            prompt_short += "…"
        prompt_escaped = prompt_short.replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
        ts_key = "created_at" if is_active else "resolved_at"
        ts_val = job.created_at if is_active else (job.resolved_at or job.updated_at)
        parts = [
            f'id="{job.id}"',
            f'state="{job.state}"',
            f'prompt="{prompt_escaped}"',
            f'{ts_key}="{ts_val or ""}"',
        ]
        if job.worker_session_id:
            parts.append(f'worker_session_id="{job.worker_session_id}"')
        if job.verifier_session_id:
            parts.append(f'verifier_session_id="{job.verifier_session_id}"')
        if job.verify_attempts:
            parts.append(f'verify_attempts="{job.verify_attempts}"')
        if job.error and job.state in (JobState.STUCK.value, JobState.ERRORED.value):
            err_short = job.error.splitlines()[0][:200].replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
            parts.append(f'error="{err_short}"')
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


def _build_notify_message(job: Job) -> str:
    """One-line wake-up message posted to the parent session on terminal transition.
    Brief by design - the parent agent should call get_job(job_id) for details."""
    prompt_short = (job.prompt or "")[:80]
    if len(job.prompt or "") > 80:
        prompt_short += "…"
    base = f"Job {job.id} finished with state '{job.state}': {prompt_short}"
    if job.state in (JobState.STUCK.value, JobState.ERRORED.value) and job.error:
        first_line = job.error.splitlines()[0][:200]
        base += f" - error: {first_line}"
    elif job.state == JobState.CANCELLED.value and job.error:
        base += f" - {job.error[:120]}"
    base += f". Use get_job('{job.id}') for details."
    return base


def _build_followup_prompt(job: Job, failed_acs: list[dict]) -> str:
    parts = ["Verifier flagged the following acceptance criteria as not met:"]
    for ac in failed_acs:
        parts.append(f"- {ac.get('ac_text', '?')}: {ac.get('reason', '?')}")
    parts.append("")
    parts.append("Address them and produce a new structured summary in the same shape as before.")
    return "\n".join(parts)


def _parse_ac_predicate(text: str) -> Optional[dict]:
    """Recognise a predicate AC prefix and return a structured dict.

    Returns None for free-text ACs (which then go to the LLM verifier as today).
    Recognised prefixes:
      `exit_code:<cmd>`       → {kind: "exit_code", cmd, expected: 0}
      `exit_code:<cmd>:<n>`   → {kind: "exit_code", cmd, expected: <n>}
      `cmd:<command>`         → {kind: "cmd", cmd} (sugar for exit_code:<cmd>:0)
      `file_exists:<path>`    → {kind: "file_exists", path}
    """
    if not text:
        return None
    s = text.strip()
    if not s:
        return None
    if s.startswith("exit_code:"):
        body = s[len("exit_code:") :].strip()
        if not body:
            return None
        # The exit code suffix is the trailing `:<int>`. Tolerate command strings
        # that themselves contain colons by splitting from the right and checking
        # if the last segment parses as an int.
        last_colon = body.rfind(":")
        if last_colon != -1:
            tail = body[last_colon + 1 :].strip()
            try:
                expected = int(tail)
                cmd = body[:last_colon].strip()
                if cmd:
                    return {"kind": "exit_code", "cmd": cmd, "expected": expected}
            except ValueError:
                pass
        return {"kind": "exit_code", "cmd": body, "expected": 0}
    if s.startswith("cmd:"):
        body = s[len("cmd:") :].strip()
        if not body:
            return None
        return {"kind": "cmd", "cmd": body}
    if s.startswith("file_exists:"):
        body = s[len("file_exists:") :].strip()
        if not body:
            return None
        return {"kind": "file_exists", "path": body}
    return None


def partition_acs(acs: list[str]) -> tuple[list[dict], list[dict]]:
    """Split an AC list into (predicates, prose), preserving original indices.

    Each predicate entry: {ac_index, ac_text, predicate}.
    Each prose entry:     {ac_index, ac_text}.
    Original ac_index is preserved so predicate and LLM results can be merged
    back into a single ac_results list spanning the full AC index space.
    """
    predicates: list[dict] = []
    prose: list[dict] = []
    for i, ac in enumerate(acs or []):
        parsed = _parse_ac_predicate(ac)
        if parsed is not None:
            predicates.append({"ac_index": i, "ac_text": ac, "predicate": parsed})
        else:
            prose.append({"ac_index": i, "ac_text": ac})
    return predicates, prose


def _resolve_predicate_cwd(job: Job) -> Optional[str]:
    """Pick cwd for predicate evaluation: worktree_path > repo > None."""
    if job.worktree_path:
        return job.worktree_path
    if job.repo:
        return job.repo
    return None


def _evaluate_predicate(
    predicate: dict,
    *,
    cwd: Optional[str],
    ac_index: int,
    ac_text: str,
    attempt: int,
    sandbox_override: Optional[dict] = None,
) -> dict:
    """Run a predicate locally and return an ac_results entry.

    `exit_code:` / `cmd:` predicates shell out. When the job is sandboxed
    (sandbox_override set, resolved from the agent's config or an inheriting
    parent), the command runs inside bubblewrap - filesystem-isolated to the
    predicate cwd (the worktree) and with no network - so a sandboxed agent's
    predicate ACs can't execute outside the sandbox. Otherwise they run with
    `shell=True` against the worktree, same surface as the worker session.
    """

    def verdict(passed: bool, reason: str) -> dict:
        return {
            "ac_index": ac_index,
            "ac_text": ac_text,
            "pass": passed,
            "reason": reason,
            "attempt": attempt,
        }

    kind = predicate.get("kind")
    try:
        if kind == "file_exists":
            path = predicate.get("path", "")
            p = Path(path)
            if not p.is_absolute() and cwd:
                p = Path(cwd) / path
            if p.exists():
                return verdict(True, "path exists")
            return verdict(False, f"path does not exist: {path}")
        if kind in ("exit_code", "cmd"):
            cmd = predicate.get("cmd", "")
            expected = predicate.get("expected", 0) if kind == "exit_code" else 0
            if cwd is None:
                logger.warning(
                    "Predicate eval has no cwd (job has neither worktree_path nor repo); "
                    "refusing to run '%s' in the daemon's cwd - marking criterion unmet",
                    cmd,
                )
                return verdict(False, "no working directory for command predicate (job has neither worktree nor repo)")
            if sandbox_override:
                from tsugite.core.sandbox import SandboxConfig, get_sandbox_class

                sandbox_cls = get_sandbox_class()
                if sandbox_cls is None:
                    return verdict(False, "command predicate requires a sandbox but no backend is installed")

                bwrap = sandbox_cls(
                    config=SandboxConfig(
                        no_network=True,
                        extra_ro_binds=[Path(p) for p in sandbox_override.get("extra_ro_binds", [])],
                        extra_rw_binds=[Path(p) for p in sandbox_override.get("extra_rw_binds", [])],
                        pass_env=list(sandbox_override.get("pass_env", [])),
                    ),
                    workspace_dir=Path(cwd),
                    state_dir=None,
                )
                run_cmd = bwrap.build_command(["sh", "-c", cmd])
                completed = subprocess.run(run_cmd, capture_output=True, timeout=_PREDICATE_TIMEOUT_SECONDS, cwd=cwd)
            else:
                completed = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    timeout=_PREDICATE_TIMEOUT_SECONDS,
                    cwd=cwd,
                )
            if completed.returncode == expected:
                return verdict(True, f"exit code {completed.returncode}")
            stderr = (completed.stderr or b"").decode("utf-8", "replace").strip()
            reason = f"exited with code {completed.returncode} (expected {expected})"
            if stderr:
                reason += f"; stderr: {stderr[:_PREDICATE_STDERR_TRUNCATE]}"
            return verdict(False, reason)
        # Unknown predicate kind - defensive; partition_acs only emits the
        # three above. Treat as a fail rather than silently passing.
        return verdict(False, f"unknown predicate kind: {kind!r}")
    except subprocess.TimeoutExpired:
        return verdict(False, "timeout")
    except Exception as e:
        return verdict(False, f"evaluation error: {e}")


def _parse_verifier_output(raw: str) -> Optional[dict]:
    """Parse the verifier's JSON output. The verifier agent file forces
    `response_format: json_object` so the LLM is required to return valid JSON;
    if parsing still fails we treat it as a verifier malfunction.

    Returns None for: empty input, malformed JSON, OR valid JSON that isn't an
    object (`42`, `null`, `true`, `[]`, `\"oops\"`) - `_handle_verifier_complete`
    must be able to do `parsed.get(...)` on the return value.
    """
    if not raw:
        return None
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


_WORKTREE_SUBDIR = ".tsugite-jobs"


def _provision_worktree(repo: str, job_id: str, workspace_root: Optional[Path] = None) -> str:
    """Add a git worktree at `<repo>/.tsugite-jobs/<job_id>` and return its absolute path.

    The worktree starts from the repo's current HEAD on a detached HEAD so the job
    can commit/branch freely without affecting the parent repo's branches.

    A relative `repo` is interpreted against `workspace_root` (the job's workspace),
    not the daemon process CWD. Absolute and `~`-expanded paths are left unchanged.
    """
    repo_path = Path(repo).expanduser()
    if not repo_path.is_absolute() and workspace_root is not None:
        repo_path = Path(workspace_root) / repo_path
    repo_path = repo_path.resolve()
    if not (repo_path / ".git").exists():
        raise ValueError(f"repo path is not a git repository: {repo_path}")
    target = repo_path / _WORKTREE_SUBDIR / job_id
    target.parent.mkdir(parents=True, exist_ok=True)
    # --detach: don't create a branch; the job can branch later if it wants.
    # HEAD: start from the repo's current commit.
    # LC_ALL=C pins git's error messages to English so log scrapes are deterministic.
    # GIT_TERMINAL_PROMPT=0 prevents git from blocking on credential prompts.
    env = {**os.environ, "LC_ALL": "C", "GIT_TERMINAL_PROMPT": "0"}
    try:
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(target), "HEAD"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        # Surface git's actual fatal message instead of the bare exit-status string.
        stderr_text = (e.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(f"git worktree add failed (exit {e.returncode}): {stderr_text or 'no stderr'}") from e
    return str(target)


def _prune_worktree(worktree_path: str) -> None:
    """Remove a previously-provisioned worktree. Errors are logged, not raised -
    cleanup must not fail a Job finalization.

    Safety: the rmtree fallback REQUIRES the path to live under our own
    `.tsugite-jobs/` subdir, so a corrupted or hand-edited worktree_path
    (e.g. an absolute path pointing at the repo root) cannot rm the wrong tree.

    `git worktree remove` needs to run inside a git repo, otherwise it errors with
    "not a git repository". `<repo>/.tsugite-jobs/<job_id>` is the layout so the
    parent's parent is the repo root - feed that as cwd. If the rmtree fallback
    fires, also run `git worktree prune` to clear the stale metadata so the next
    `worktree add` at the same path doesn't see a "missing but already registered"
    record.
    """
    wt = Path(worktree_path)
    if not wt.exists():
        return
    # The worktree path is `<repo>/.tsugite-jobs/<job_id>` - walk up two levels for the repo.
    repo_root = wt.parent.parent if wt.parent.name == _WORKTREE_SUBDIR else None
    # `git worktree remove --force` works even if the worktree has uncommitted changes.
    try:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(wt)],
            check=True,
            capture_output=True,
            cwd=str(repo_root) if repo_root else None,
        )
        return
    except Exception as e:
        stderr = getattr(e, "stderr", b"")
        stderr_text = (
            stderr.decode("utf-8", "replace").strip() if isinstance(stderr, (bytes, bytearray)) else str(stderr)
        )
        logger.warning("git worktree remove failed for %s: %s (stderr: %s); attempting rmtree", wt, e, stderr_text)

    # rmtree fallback - but ONLY if the path is structurally inside .tsugite-jobs/.
    # Any other location indicates corruption or tampering; refuse rather than
    # nuke an arbitrary tree.
    resolved = wt.resolve()
    if _WORKTREE_SUBDIR not in resolved.parts:
        logger.error(
            "Refusing rmtree of %s: path is not inside %s/ - corrupted Job.worktree_path?",
            resolved,
            _WORKTREE_SUBDIR,
        )
        return
    import shutil

    try:
        shutil.rmtree(wt, ignore_errors=True)
    except Exception:
        logger.exception("Failed to rmtree worktree at %s", wt)

    # Tell git the worktree is gone so a subsequent `worktree add` at the same path
    # doesn't trip on a stale registration.
    if repo_root and repo_root.exists():
        try:
            subprocess.run(
                ["git", "worktree", "prune"],
                check=False,
                capture_output=True,
                cwd=str(repo_root),
            )
        except Exception:
            logger.debug("git worktree prune fallback failed for %s", repo_root)
