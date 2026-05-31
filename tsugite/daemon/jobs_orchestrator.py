"""Jobs orchestrator — bridges Job state machine with SessionRunner lifecycle.

Subscribes to SessionRunner's notify_callback to detect worker/verifier
completions, transitions Job state, spawns verifier rounds, and broadcasts
tile updates to the parent chat.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from tsugite.daemon.job_store import Job, JobState, JobStateTransitionError, JobStore
from tsugite.daemon.session_store import Session, SessionSource, SessionStatus

logger = logging.getLogger(__name__)

MAX_VERIFY_ATTEMPTS = 3
VERIFIER_AGENT = "job_verifier"


class JobsOrchestrator:
    """Drives Job state in response to worker/verifier session completions."""

    def __init__(self, job_store: JobStore, session_runner, event_bus=None):
        self._jobs = job_store
        self._runner = session_runner
        self._event_bus = event_bus
        self._timeout_handles: dict[str, asyncio.TimerHandle] = {}

    def attach(self) -> None:
        """Wire this orchestrator into the SessionRunner notify_callback chain.

        Idempotent: re-attaching the same orchestrator is a no-op so test fixtures /
        hot-reload paths don't accidentally double-wrap and double-dispatch.
        """
        if getattr(self._runner, "_jobs_orchestrator_attached", None) is self:
            return
        existing = getattr(self._runner, "_notify_callback", None)

        async def _chained(session, result_str):
            try:
                await self.on_session_complete(session, result_str)
            except Exception as e:
                logger.exception("JobsOrchestrator.on_session_complete failed: %s", e)
            if existing:
                try:
                    await existing(session, result_str)
                except Exception:
                    logger.exception("Chained notify_callback failed")

        self._runner._notify_callback = _chained
        self._runner._jobs_orchestrator_attached = self

    def create_and_start_job(
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
        notify: bool = False,
        max_attempts: Optional[int] = None,
        notify_when: Optional[str] = None,
    ) -> tuple[Job, Session]:
        """Create a Job record + spawn the worker session in one step.

        Used by both the /job slash command and the spawn_job() agent tool.
        Returns (job, started_worker_session). The orchestrator's register_worker
        is called automatically so the timeout is scheduled and the tile event fires.

        notify: legacy bool; True maps to notify_when="terminal". Prefer notify_when.
        notify_when: one of "done", "stuck", "errored", "terminal", "never" (default).
        max_attempts: verifier-loop cap. Defaults to 3 when omitted.
        """
        worker_agent_file = agent or "job_worker"
        worker_adapter_key = self._resolve_adapter_key(parent_session_id)
        job_kwargs: dict = {}
        if max_attempts is not None:
            job_kwargs["max_attempts"] = max_attempts
        if notify_when:
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
                notify=notify,
                **job_kwargs,
            )
        )

        # Provision a fresh git worktree if --repo was given so the worker has
        # an isolated working tree (no clashes with the parent shell or other jobs).
        worktree_path: Optional[str] = None
        if repo:
            try:
                worktree_path = _provision_worktree(repo, job.id)
                self._jobs.update(job.id, worktree_path=worktree_path)
            except Exception as e:
                logger.exception("Failed to provision worktree for job '%s': %s", job.id, e)
                self._finalize(job, JobState.ERRORED, error=f"worktree provisioning failed: {e}")
                raise

        worker_prompt = build_worker_prompt(prompt, acceptance_criteria or [], repo)
        session = Session(
            id="",
            agent=worker_adapter_key,
            source=SessionSource.SPAWNED.value,
            prompt=worker_prompt,
            agent_file=worker_agent_file,
            model=model,
            workspace_override=worktree_path,
            metadata={"job_id": job.id},
        )
        try:
            started = self._runner.start_session(session)
        except Exception as e:
            # Don't leave the Job persisted in QUEUED; mark it ERRORED so it
            # doesn't accumulate as a zombie in jobs.json.
            self._finalize(job, JobState.ERRORED, error=f"worker spawn failed: {e}")
            raise
        self.register_worker(job.id, started.id, timeout_minutes=timeout_minutes)
        return job, started

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

    def _set_attempt_verifier(self, job_id: str, verifier_session_id: str) -> None:
        """Link a verifier session to the latest attempt entry."""
        job = self._jobs.get(job_id)
        if job is None or not job.attempts:
            return
        attempts = list(job.attempts)
        attempts[-1] = {**attempts[-1], "verifier_session_id": verifier_session_id}
        self._jobs.update(job_id, attempts=attempts)

    def _set_attempt_verdict(self, job_id: str, verifier_pass: bool) -> None:
        """Record the verifier's pass/fail verdict on the latest attempt entry."""
        job = self._jobs.get(job_id)
        if job is None or not job.attempts:
            return
        attempts = list(job.attempts)
        attempts[-1] = {**attempts[-1], "verifier_pass": bool(verifier_pass)}
        self._jobs.update(job_id, attempts=attempts)

    def _record_ac_results(self, job_id: str, raw_ac_results, attempt_num: int) -> None:
        """Append per-criterion verdicts for one verifier round to Job.ac_results.

        Replaces any prior entries tagged with the same attempt_num (defensive in
        case a verifier completion is delivered twice). The list itself grows by
        one batch per attempt so the UI can render historical verdicts.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return
        prior = [e for e in (job.ac_results or []) if e.get("attempt") != attempt_num]
        new_entries: list[dict] = []
        if isinstance(raw_ac_results, list):
            for i, item in enumerate(raw_ac_results):
                if isinstance(item, dict):
                    new_entries.append(
                        {
                            "ac_index": i,
                            "ac_text": item.get("ac_text", ""),
                            "pass": bool(item.get("pass")),
                            "reason": item.get("reason"),
                            "attempt": attempt_num,
                        }
                    )
                else:
                    # Malformed verifier output (string, None, etc) — keep a
                    # placeholder so the UI still shows something.
                    new_entries.append(
                        {
                            "ac_index": i,
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
        # First call to register_worker for this Job — record the initial attempt.
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

    async def cancel_job(self, job_id: str, reason: str = "cancelled by user") -> Job:
        """User-initiated cancel from the tile. No-op on terminal Jobs."""
        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(f"Unknown job: {job_id}")
        if job.state in (JobState.DONE.value, JobState.STUCK.value, JobState.CANCELLED.value, JobState.ERRORED.value):
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
        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(f"Unknown job: {job_id}")
        if job.state != JobState.STUCK.value:
            raise ValueError(f"mark_done_manual only valid on stuck jobs (job '{job_id}' is {job.state})")
        # Defensive: STUCK should have no pending timer (set in _finalize), but cancel
        # to be safe in case a future code path leaves one behind.
        self._cancel_timeout(job_id)
        # STUCK has no outgoing transitions in _VALID_TRANSITIONS by design (terminal).
        # The audit-stamped override deliberately goes around the state machine, so we
        # mutate state directly via update() and emit. _finalize would be rejected.
        result = dict(job.result or {})
        result["manual_done_reason"] = reason
        # Preserve the verifier's diagnostic so we don't lose the audit trail of why
        # the job was stuck in the first place.
        if job.error:
            result["stuck_error_at_override"] = job.error
        self._jobs.update(job_id, state=JobState.DONE.value, result=result, error=None, resolved_at=_iso_now())
        if job.worktree_path:
            _prune_worktree(job.worktree_path)
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
        """Give a STUCK Job one more shot, with the user's hint as the worker prompt.

        Args:
            job_id: STUCK job to resurrect.
            hint: Free-text guidance for the new worker.
            reset_counter: Zero out `verify_attempts` so the retry gets a full new
                budget of verifier rounds (the UI exposes this as "reset to 1").
                Defaults False to preserve the historical no-infinite-loops guard.
            fresh_workspace: When the job has a `repo` worktree, prune the existing
                tree and recreate it from HEAD before spawning. No-op when the Job
                was created without `repo`.
        """
        from pathlib import Path

        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(f"Unknown job: {job_id}")
        if job.state != JobState.STUCK.value:
            raise ValueError(f"retry_with_hint only valid on stuck jobs (job '{job_id}' is {job.state})")

        worktree_path = job.worktree_path
        if fresh_workspace and job.repo:
            try:
                if worktree_path and Path(worktree_path).exists():
                    _prune_worktree(worktree_path)
                worktree_path = _provision_worktree(job.repo, job.id)
                self._jobs.update(job_id, worktree_path=worktree_path)
            except Exception as e:
                logger.exception("retry_with_hint: fresh_workspace failed for job '%s': %s", job_id, e)
                raise ValueError(f"failed to recreate worktree: {e}") from e
        elif worktree_path and not Path(worktree_path).is_dir():
            # Worktree was hand-deleted between STUCK and retry — refuse rather
            # than spawn a worker into a missing directory.
            raise ValueError(
                f"retry_with_hint: worktree at '{worktree_path}' no longer exists; "
                f"cannot resume in a missing directory"
            )

        if reset_counter:
            self._jobs.update(job_id, verify_attempts=0)
            job = self._jobs.get(job_id)

        # Same out-of-band rationale as mark_done_manual: STUCK is terminal in the
        # state machine, and that's intentional. The hint-and-retry escape hatch is
        # explicitly out-of-band.
        worker_agent_file = job.agent or "job_worker"
        worker_adapter_key = self._resolve_adapter_key(job.parent_session_id)
        worker_session = Session(
            id="",
            agent=worker_adapter_key,
            source=SessionSource.SPAWNED.value,
            prompt=_build_hint_prompt(job, hint),
            agent_file=worker_agent_file,
            model=job.model,
            workspace_override=worktree_path,
            metadata={"job_id": job.id, "hint_attempt": True},
        )
        try:
            started = self._runner.start_session(worker_session)
        except Exception as e:
            logger.exception("retry_with_hint: failed to spawn worker for job '%s': %s", job_id, e)
            raise ValueError(f"failed to spawn retry worker: {e}") from e
        self._append_attempt(job_id, kind="hint", worker_session_id=started.id)
        # Move out of STUCK directly (terminal → RUNNING). update() goes around the
        # state machine but the audit is captured via the resolved_at clear + emit.
        self._jobs.update(
            job_id,
            state=JobState.RUNNING.value,
            worker_session_id=started.id,
            resolved_at=None,
            error=None,
        )
        self._emit_job_event(self._jobs.get(job_id))
        self._schedule_timeout(job_id, job.timeout_minutes)
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
            # Verifier crashed or got cancelled — infrastructure failure, NOT a
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
                "Worker completion for job '%s' in state '%s' (expected RUNNING) — ignoring",
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

        # Arm a fresh timer covering the verifier round; a hung verifier would
        # otherwise leave the Job pinned in VERIFYING forever.
        self._schedule_timeout(job.id, job.timeout_minutes)

        verifier_prompt = _build_verifier_prompt(job, worker_output=result_str)
        verifier_adapter_key = self._resolve_adapter_key(job.parent_session_id)
        verifier_session = Session(
            id="",
            agent=verifier_adapter_key,
            source=SessionSource.SPAWNED.value,
            prompt=verifier_prompt,
            agent_file=VERIFIER_AGENT,
            metadata={"job_id": job.id, "verifier_for": worker.id},
        )
        try:
            started_verifier = self._runner.start_session(verifier_session)
        except Exception as e:
            logger.exception("Failed to spawn verifier for job '%s': %s", job.id, e)
            self._finalize(job, JobState.ERRORED, error=f"verifier spawn failed: {e}")
            return
        # Store verifier session id so _on_timeout can cancel a hung verifier.
        self._jobs.update(job.id, verifier_session_id=started_verifier.id)
        self._set_attempt_verifier(job.id, started_verifier.id)

    async def _handle_worker_failed(self, job: Job, worker: Session, result_str: str) -> None:
        self._cancel_timeout(job.id)
        # User-initiated cancellation vs genuine failure — terminal state reflects intent.
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
                "Verifier completion for job '%s' in state '%s' (expected VERIFYING) — ignoring",
                job.id,
                job.state,
            )
            return

        # Attempt counter is 1-indexed (matches the UX of "attempt #N"). Captured
        # before _handle_verifier_failure bumps job.verify_attempts.
        attempt_num = job.verify_attempts + 1

        parsed = _parse_verifier_output(result_str)
        if parsed is None:
            self._set_attempt_verdict(job.id, False)
            # Synthetic one-entry batch so the UI sees *something* for this attempt.
            self._record_ac_results(
                job.id,
                [{"ac_text": "(verifier output)", "pass": False, "reason": "verifier output unparseable"}],
                attempt_num,
            )
            return await self._handle_verifier_failure(
                job,
                failed_acs=[{"ac_text": "(verifier output)", "pass": False, "reason": "verifier output unparseable"}],
            )
        # Strict True: a string like "false" / "no" is truthy in Python but
        # explicitly NOT a pass. Treat anything other than literal True as failure.
        ac_results_raw = parsed.get("ac_results", [])
        self._record_ac_results(job.id, ac_results_raw, attempt_num)
        if parsed.get("overall_pass") is True:
            self._set_attempt_verdict(job.id, True)
            fresh = self._jobs.get(job.id)
            result = dict(fresh.result or {})
            result["ac_results"] = ac_results_raw
            self._finalize(job, JobState.DONE, result=result)
            return
        self._set_attempt_verdict(job.id, False)
        await self._handle_verifier_failure(
            job,
            failed_acs=_extract_failed_acs(ac_results_raw),
        )

    async def _handle_verifier_failure(self, job: Job, failed_acs: list[dict]) -> None:
        # Empty failed_acs gives the retry worker zero signal and the STUCK error
        # zero diagnostic content — synthesize a placeholder so users see something.
        if not failed_acs:
            failed_acs = [
                {
                    "ac_text": "(verifier reported failure with no specific criteria)",
                    "pass": False,
                    "reason": "verifier set overall_pass=false but did not list which AC failed; review the AC list verbatim",
                }
            ]
        new_attempts = job.verify_attempts + 1
        # `MAX_VERIFY_ATTEMPTS` stays the module default (and the legacy contract
        # for older jobs without an explicit field); per-job override comes from
        # Job.max_attempts so the new-job modal's --max-attempts flag wins.
        cap = job.max_attempts if getattr(job, "max_attempts", None) else MAX_VERIFY_ATTEMPTS
        if new_attempts >= cap:
            error_lines = ["Verifier failed after max attempts:"]
            for ac in failed_acs:
                error_lines.append(f"- {ac.get('ac_text', '?')}: {ac.get('reason', '?')}")
            # STUCK preserves the verifier's structured ac_results alongside the worker
            # summary so UI/API consumers can render per-AC verdicts (mirrors DONE).
            fresh = self._jobs.get(job.id)
            result = dict(fresh.result or {})
            result["ac_results"] = failed_acs
            self._jobs.update(job.id, verify_attempts=new_attempts, result=result)
            self._finalize(job, JobState.STUCK, error="\n".join(error_lines))
            return

        # Retry: bump counter, spawn new worker, transition verifying → running.
        self._jobs.update(job.id, verify_attempts=new_attempts)
        followup = _build_followup_prompt(job, failed_acs)
        worker_agent_file = job.agent or "job_worker"
        worker_adapter_key = self._resolve_adapter_key(job.parent_session_id)
        worker_session = Session(
            id="",
            agent=worker_adapter_key,
            source=SessionSource.SPAWNED.value,
            prompt=followup,
            agent_file=worker_agent_file,
            model=job.model,
            workspace_override=job.worktree_path,
            # loop_attempt mirrors verify_attempts (1-indexed retry count).
            metadata={"job_id": job.id, "loop_attempt": new_attempts},
        )
        try:
            started = self._runner.start_session(worker_session)
        except Exception as e:
            logger.exception("Failed to spawn retry worker for job '%s': %s", job.id, e)
            self._finalize(job, JobState.ERRORED, error=f"retry worker spawn failed: {e}")
            return

        self._jobs.update(job.id, worker_session_id=started.id)
        self._append_attempt(job.id, kind="retry", worker_session_id=started.id)
        try:
            self._jobs.update_state(job.id, JobState.RUNNING.value)
        except JobStateTransitionError as e:
            logger.warning("Cannot transition job '%s' verifying → running: %s", job.id, e)
            return
        self._emit_job_event(self._jobs.get(job.id))
        self._schedule_timeout(job.id, job.timeout_minutes)

    def _finalize(self, job: Job, terminal: JobState, **fields) -> None:
        """Cancel timer, write per-terminal fields, transition, emit tile event.

        Worktrees are pruned on DONE/CANCELLED (clean exit, no inspection value),
        kept on STUCK/ERRORED so the user can see what the worker did wrong.
        If job.notify is True, also schedules a wake-up reply to the parent session.
        """
        self._cancel_timeout(job.id)
        if fields:
            self._jobs.update(job.id, **fields)
        try:
            self._jobs.update_state(job.id, terminal.value)
        except JobStateTransitionError as e:
            logger.warning("Cannot mark job '%s' %s: %s", job.id, terminal.value, e)
            return
        fresh = self._jobs.get(job.id)
        if terminal in (JobState.DONE, JobState.CANCELLED) and fresh and fresh.worktree_path:
            _prune_worktree(fresh.worktree_path)
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

        loop.create_task(_send())

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
        if job is None or job.state in (
            JobState.DONE.value,
            JobState.STUCK.value,
            JobState.CANCELLED.value,
            JobState.ERRORED.value,
        ):
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
        overwrite it via cancel_session — the session won. Tolerates both the
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

    def _resolve_adapter_key(self, parent_session_id: str) -> str:
        """Route Job spawns through the parent session's adapter so jobs stay on
        the same agent (credentials, tools, model defaults) as the chat that spawned them.
        Tolerates real SessionStore raising ValueError on missing sessions.
        """
        store = getattr(self._runner, "store", None)
        if store is not None and hasattr(store, "get_session"):
            try:
                parent = store.get_session(parent_session_id)
            except (ValueError, KeyError):
                parent = None
            if parent is not None:
                return parent.agent
        adapters = getattr(self._runner, "_adapters", {})
        if adapters:
            return next(iter(adapters))
        return "default"

    def _emit_job_event(self, job: Optional[Job]) -> None:
        if job is None:
            return
        payload = {
            "job_id": job.id,
            "parent_session_id": job.parent_session_id,
            "worker_session_id": job.worker_session_id,
            "verifier_session_id": job.verifier_session_id,
            "state": job.state,
            "prompt": (job.prompt or "")[:200],
            "verify_attempts": job.verify_attempts,
            "max_attempts": getattr(job, "max_attempts", 3),
            "notify_when": getattr(job, "notify_when", "never"),
            "error": job.error,
            "attempts": list(job.attempts or []),
            "acceptance_criteria": list(job.acceptance_criteria or []),
            "ac_results": list(getattr(job, "ac_results", None) or []),
            "result": job.result,
        }
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
    "never" (no-op). Anything else is treated as "never" (defensive — a typo on
    disk shouldn't spam the parent agent).
    """
    notify_when = getattr(job, "notify_when", None) or ("terminal" if getattr(job, "notify", False) else "never")
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


def _ac_text(ac) -> str:
    """Extract the human-readable text from an AC entry of either shape."""
    return ac.get("text", "") if isinstance(ac, dict) else str(ac)


def _build_verifier_prompt(job: Job, worker_output: str) -> str:
    parts = ["Acceptance criteria:"]
    for i, ac in enumerate(job.acceptance_criteria, 1):
        parts.append(f"{i}. {_ac_text(ac)}")
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
        parts.append(f"{i}. {_ac_text(ac)}")
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
    terminal_states = {JobState.DONE.value, JobState.STUCK.value, JobState.CANCELLED.value, JobState.ERRORED.value}

    active = [j for j in jobs if j.state in active_states]
    recent = [j for j in jobs if j.state in terminal_states]
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
    Brief by design — the parent agent should call get_job(job_id) for details."""
    prompt_short = (job.prompt or "")[:80]
    if len(job.prompt or "") > 80:
        prompt_short += "…"
    base = f"Job {job.id} finished with state '{job.state}': {prompt_short}"
    if job.state in (JobState.STUCK.value, JobState.ERRORED.value) and job.error:
        first_line = job.error.splitlines()[0][:200]
        base += f" — error: {first_line}"
    elif job.state == JobState.CANCELLED.value and job.error:
        base += f" — {job.error[:120]}"
    base += f". Use get_job('{job.id}') for details."
    return base


def _build_followup_prompt(job: Job, failed_acs: list[dict]) -> str:
    parts = ["Verifier flagged the following acceptance criteria as not met:"]
    for ac in failed_acs:
        parts.append(f"- {ac.get('ac_text', '?')}: {ac.get('reason', '?')}")
    parts.append("")
    parts.append("Address them and produce a new structured summary in the same shape as before.")
    return "\n".join(parts)


def _parse_verifier_output(raw: str) -> Optional[dict]:
    """Parse the verifier's JSON output. The verifier agent file forces
    `response_format: json_object` so the LLM is required to return valid JSON;
    if parsing still fails we treat it as a verifier malfunction.

    Returns None for: empty input, malformed JSON, OR valid JSON that isn't an
    object (`42`, `null`, `true`, `[]`, `\"oops\"`) — `_handle_verifier_complete`
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


def _provision_worktree(repo: str, job_id: str) -> str:
    """Add a git worktree at `<repo>/.tsugite-jobs/<job_id>` and return its absolute path.

    The worktree starts from the repo's current HEAD on a detached HEAD so the job
    can commit/branch freely without affecting the parent repo's branches.
    """
    import subprocess
    from pathlib import Path

    repo_path = Path(repo).expanduser().resolve()
    if not (repo_path / ".git").exists():
        raise ValueError(f"repo path is not a git repository: {repo_path}")
    target = repo_path / _WORKTREE_SUBDIR / job_id
    target.parent.mkdir(parents=True, exist_ok=True)
    # --detach: don't create a branch; the job can branch later if it wants.
    # HEAD: start from the repo's current commit.
    # LC_ALL=C pins git's error messages to English so log scrapes are deterministic.
    # GIT_TERMINAL_PROMPT=0 prevents git from blocking on credential prompts.
    import os as _os

    env = {**_os.environ, "LC_ALL": "C", "GIT_TERMINAL_PROMPT": "0"}
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
    """Remove a previously-provisioned worktree. Errors are logged, not raised —
    cleanup must not fail a Job finalization.

    Safety: the rmtree fallback REQUIRES the path to live under our own
    `.tsugite-jobs/` subdir, so a corrupted or hand-edited worktree_path
    (e.g. an absolute path pointing at the repo root) cannot rm the wrong tree.

    `git worktree remove` needs to run inside a git repo, otherwise it errors with
    "not a git repository". `<repo>/.tsugite-jobs/<job_id>` is the layout so the
    parent's parent is the repo root — feed that as cwd. If the rmtree fallback
    fires, also run `git worktree prune` to clear the stale metadata so the next
    `worktree add` at the same path doesn't see a "missing but already registered"
    record.
    """
    import subprocess
    from pathlib import Path

    wt = Path(worktree_path)
    if not wt.exists():
        return
    # The worktree path is `<repo>/.tsugite-jobs/<job_id>` — walk up two levels for the repo.
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

    # rmtree fallback — but ONLY if the path is structurally inside .tsugite-jobs/.
    # Any other location indicates corruption or tampering; refuse rather than
    # nuke an arbitrary tree.
    resolved = wt.resolve()
    if _WORKTREE_SUBDIR not in resolved.parts:
        logger.error(
            "Refusing rmtree of %s: path is not inside %s/ — corrupted Job.worktree_path?",
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
