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
        acceptance_criteria: Optional[list[str]] = None,
        repo: Optional[str] = None,
        model: Optional[str] = None,
        agent: Optional[str] = None,
        timeout_minutes: int = 30,
        spawned_by: str = "user-slash",
    ) -> tuple[Job, Session]:
        """Create a Job record + spawn the worker session in one step.

        Used by both the /job slash command and the spawn_job() agent tool.
        Returns (job, started_worker_session). The orchestrator's register_worker
        is called automatically so the timeout is scheduled and the tile event fires.
        """
        worker_agent_file = agent or "job_worker"
        worker_adapter_key = self._resolve_adapter_key(parent_session_id)

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
            )
        )

        worker_prompt = build_worker_prompt(prompt, acceptance_criteria or [], repo)
        session = Session(
            id="",
            agent=worker_adapter_key,
            source=SessionSource.SPAWNED.value,
            prompt=worker_prompt,
            agent_file=worker_agent_file,
            model=model,
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

    def register_worker(self, job_id: str, worker_session_id: str, timeout_minutes: int) -> None:
        """Record the worker session id and schedule the wall-clock timeout."""
        self._jobs.update(job_id, worker_session_id=worker_session_id)
        try:
            self._jobs.update_state(job_id, JobState.RUNNING.value)
        except JobStateTransitionError:
            # If we're being called again on a retry, the Job is already in RUNNING.
            pass
        self._emit_job_event(self._jobs.get(job_id))
        self._schedule_timeout(job_id, timeout_minutes)

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
        self._cancel_timeout(job.id)
        # Persist worker output before spawning the verifier so the verifier-pass
        # path can echo it back into job.result, AND so the UI sees the latest
        # payload even if the transition below is rejected.
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

        parsed = _parse_verifier_output(result_str)
        if parsed is None:
            return await self._handle_verifier_failure(
                job,
                failed_acs=[{"ac_text": "(verifier output)", "pass": False, "reason": "verifier output unparseable"}],
            )
        # Strict True: a string like "false" / "no" is truthy in Python but
        # explicitly NOT a pass. Treat anything other than literal True as failure.
        if parsed.get("overall_pass") is True:
            fresh = self._jobs.get(job.id)
            result = dict(fresh.result or {})
            result["ac_results"] = parsed.get("ac_results", [])
            self._finalize(job, JobState.DONE, result=result)
            return
        await self._handle_verifier_failure(
            job,
            failed_acs=_extract_failed_acs(parsed.get("ac_results", [])),
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
        if new_attempts >= MAX_VERIFY_ATTEMPTS:
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
        try:
            self._jobs.update_state(job.id, JobState.RUNNING.value)
        except JobStateTransitionError as e:
            logger.warning("Cannot transition job '%s' verifying → running: %s", job.id, e)
            return
        self._emit_job_event(self._jobs.get(job.id))
        self._schedule_timeout(job.id, job.timeout_minutes)

    def _finalize(self, job: Job, terminal: JobState, **fields) -> None:
        """Cancel timer, write per-terminal fields, transition, emit tile event."""
        self._cancel_timeout(job.id)
        if fields:
            self._jobs.update(job.id, **fields)
        try:
            self._jobs.update_state(job.id, terminal.value)
        except JobStateTransitionError as e:
            logger.warning("Cannot mark job '%s' %s: %s", job.id, terminal.value, e)
            return
        self._emit_job_event(self._jobs.get(job.id))

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
        overwrite it via cancel_session — the session won."""
        store = getattr(self._runner, "store", None)
        if store is None or not hasattr(store, "get_session"):
            return False
        session = store.get_session(session_id)
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
        """
        store = getattr(self._runner, "store", None)
        if store is not None and hasattr(store, "get_session"):
            parent = store.get_session(parent_session_id)
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
            "state": job.state,
            "prompt": (job.prompt or "")[:200],
            "verify_attempts": job.verify_attempts,
            "error": job.error,
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


def build_worker_prompt(prompt: str, acceptance_criteria: list[str], repo: Optional[str]) -> str:
    """Compose the worker's initial user_prompt with AC and repo context inlined."""
    parts = [prompt]
    if acceptance_criteria:
        parts.append("")
        parts.append("Acceptance criteria (the verifier will grade your work against these):")
        for i, ac in enumerate(acceptance_criteria, 1):
            parts.append(f"{i}. {ac}")
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


def _build_verifier_prompt(job: Job, worker_output: str) -> str:
    parts = ["Acceptance criteria:"]
    for i, ac in enumerate(job.acceptance_criteria, 1):
        parts.append(f"{i}. {ac}")
    parts.append("")
    parts.append("Worker output:")
    parts.append(worker_output.strip() or "(empty)")
    if job.repo:
        parts.append("")
        parts.append(f"Repo: {job.repo}")
        parts.append("(use `run` to inspect `git diff` or `git log` if relevant.)")
    return "\n".join(parts)


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
