"""Jobs tool - `spawn_job()` for agents, mirroring spawn_session for plain sessions.

Tool uses @tool(require_daemon=True) so it only appears in daemon mode.
"""

import concurrent.futures
from typing import Optional

from . import call_on_loop, tool

_jobs_orchestrator = None
_loop = None


def set_jobs_orchestrator(orchestrator, loop):
    """Called by the daemon to set/clear the orchestrator reference."""
    global _jobs_orchestrator, _loop
    _jobs_orchestrator = orchestrator
    _loop = loop


def get_jobs_orchestrator():
    """Public accessor for the wired orchestrator (None outside daemon mode).
    Used by the adapters for context injection and capability flags."""
    return _jobs_orchestrator


def _call(fn, *args, timeout=30, **kwargs):
    """Call an orchestrator method on the daemon loop thread (thread-safe)."""
    return call_on_loop(_loop, fn, *args, timeout=timeout, **kwargs)


@tool(require_daemon=True)
def spawn_job(
    prompt: str,
    acceptance_criteria: Optional[list] = None,
    repo: Optional[str] = None,
    model: Optional[str] = None,
    verifier_model: Optional[str] = None,
    model_ladder: Optional[list] = None,
    timeout_minutes: int = 30,
    agent: Optional[str] = None,
    max_attempts: Optional[int] = None,
    notify_when: Optional[str] = None,
    executor: str = "agent",
    effort: Optional[str] = None,
) -> dict:
    """Spawn a background Job with a verification loop.

    Creates a Job record + worker session pinned to the workspace default model
    (or the supplied `model`). On worker completion the orchestrator spawns a
    verifier sub-agent that judges the result against `acceptance_criteria`. If
    verification fails, the Job loops back up to `max_attempts` times before
    going `stuck` (default 3).

    Use this instead of `spawn_session` when you want structured verification
    of the work, not just a fire-and-forget background session.

    Args:
        prompt: Task instruction for the spawned Job.
        acceptance_criteria: List of criterion strings the verifier grades
            against. Empty list short-circuits verification (Job goes straight
            to done).
        repo: Repo path (workspace-relative or absolute). The worker runs in a
            fresh git worktree provisioned under `<repo>/.tsugite-jobs/<job_id>`,
            isolated from the parent checkout; it's pruned on done/cancelled and
            kept on stuck/errored for inspection.
        model: Optional model override for the worker; defaults to the
            workspace default.
        verifier_model: Optional separate model for the verifier round. Defaults
            to `model` (then the workspace default) when unset - handy to pin
            verification to a cheaper/more-available model than the worker.
        model_ladder: Ordered "cheap first" model list, e.g.
            ["claude_code:haiku", "claude_code:opus"]. The Job starts on the
            first entry; when a rung exhausts its verifier attempts or the
            worker dies on a usage/rate limit, the next attempt escalates to
            the next model with a fresh budget. Overrides `model`.
        timeout_minutes: Per-phase budget for each worker run and verifier round
            (re-armed each phase, not a whole-job wall clock); on expiry the Job
            transitions to `stuck`.
        agent: Worker agent file. Defaults to `job_worker`.
        max_attempts: Verifier-loop cap before stuck. Defaults to 3.
        notify_when: When to wake the parent: done | stuck | errored | terminal |
            never. Defaults to never - be deliberate when enabling, since each
            notification adds a turn to the parent conversation.
        executor: Which registered executor produces the work. "agent" (default)
            spawns a tsugite worker session; a plugin may register others (e.g. a
            PTY-driven CLI). An unknown name is rejected.
        effort: Optional per-job reasoning effort (low|medium|high|xhigh|max).
            For a cc executor job this maps to claude's --effort; ignored by
            executors that don't support it.

    Returns:
        Dict with job_id, worker_session_id, parent_session_id, state.
    """
    from tsugite_daemon.session_runner import get_current_session_id

    from tsugite.agent_runner.helpers import sandbox_context_to_override

    parent_session_id = get_current_session_id()
    if not parent_session_id:
        raise RuntimeError("spawn_job requires a current session context")
    if _jobs_orchestrator is None:
        raise RuntimeError("Jobs orchestrator not initialised")

    # Inherit the sandbox: worker + verifier sessions stay sandboxed if this agent
    # is, and predicate ACs are evaluated inside bwrap by the orchestrator.
    sandbox_override = sandbox_context_to_override()

    try:
        # Generous timeout: --repo provisioning runs `git worktree add`, which
        # can take minutes on a large checkout.
        job, started = _call(
            _jobs_orchestrator.create_and_start_job,
            parent_session_id=parent_session_id,
            prompt=prompt,
            acceptance_criteria=acceptance_criteria,
            repo=repo,
            model=model,
            verifier_model=verifier_model,
            model_ladder=model_ladder,
            agent=agent,
            timeout_minutes=timeout_minutes,
            spawned_by="agent-tool",
            max_attempts=max_attempts,
            notify_when=notify_when,
            sandbox_override=sandbox_override,
            executor=executor,
            effort=effort,
            timeout=180,
        )
    except concurrent.futures.TimeoutError:
        # The coroutine keeps running on the daemon loop and may still create
        # the job - a blind retry would duplicate it.
        raise RuntimeError(
            "spawn_job timed out waiting for the daemon; the job may still start in the "
            "background - check list_jobs before retrying to avoid a duplicate"
        ) from None

    return {
        "job_id": job.id,
        # None for a non-agent executor job (no worker Session is spawned).
        "worker_session_id": started.id if started else None,
        "parent_session_id": parent_session_id,
        "state": "running",
    }


@tool(require_daemon=True)
def get_job(job_id: str) -> dict:
    """Return the full record for a Job by id.

    Includes the structured worker summary (in `result.summary`), per-AC
    verifier verdicts (in `result.ac_results` when verifier ran), the
    worker/verifier session ids (so you can navigate via `session_status`),
    state, timestamps, and any error.

    Args:
        job_id: Job id (e.g. 'job-4f2a1b3c').

    Returns:
        Job record as a dict, or {"error": "..."} if not found.
    """
    if _jobs_orchestrator is None:
        return {"error": "Jobs orchestrator not initialised"}
    from dataclasses import asdict as _asdict

    job = _jobs_orchestrator._jobs.get(job_id)
    if job is None:
        return {"error": f"Unknown job: {job_id}"}
    return _asdict(job)


@tool(require_daemon=True)
def cancel_job(job_id: str, reason: str = "cancelled by agent") -> dict:
    """Cancel a Job (or dismiss a parked one).

    Stops the worker/verifier for a live Job (queued/running/verifying) and
    finalizes it as cancelled. For a stuck/errored Job this is the "give up /
    dismiss" action. No-op when the Job is already done/cancelled.

    Args:
        job_id: Job id (e.g. 'job-4f2a1b3c').
        reason: Short audit note recorded on the Job.

    Returns:
        Dict with job_id and the resulting state, or {"error": "..."}.
    """
    if _jobs_orchestrator is None:
        return {"error": "Jobs orchestrator not initialised"}
    try:
        job = _call(_jobs_orchestrator.cancel_job, job_id, reason)
    except Exception as e:
        return {"error": str(e)}
    return {"job_id": job.id, "state": job.state}


@tool(require_daemon=True)
def respond_to_job(job_id: str, message: str) -> dict:
    """Send input/steering to an executor Job's live worker.

    Use this to answer a question the worker is blocked on (a Job in the
    `awaiting_input` state - answering resumes it) or to add mid-flight guidance
    to a running one. The message is fed into the live worker session (for a cc
    job it is typed into the worker's terminal) and does not consume a
    verification attempt. Only valid for non-agent executor Jobs (e.g.
    executor="cc"); for stuck/errored Jobs use the retry action instead.

    Args:
        job_id: Job id (e.g. 'job-4f2a1b3c').
        message: The input/guidance to deliver to the worker.

    Returns:
        Dict with job_id and state, or {"error": "..."}.
    """
    if _jobs_orchestrator is None:
        return {"error": "Jobs orchestrator not initialised"}
    try:
        job = _call(_jobs_orchestrator.respond_to_job, job_id, message)
    except Exception as e:
        return {"error": str(e)}
    return {"job_id": job.id, "state": job.state}


@tool(require_daemon=True)
def list_jobs(
    session_id: Optional[str] = None,
    state: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = 10,
) -> list[dict]:
    """List Jobs with optional filters.

    Args:
        session_id: Only return Jobs whose parent is this session id.
        state: Only return Jobs in this state (queued|running|verifying|done|stuck|cancelled|errored).
        since: ISO timestamp; only return Jobs whose `created_at` is >= this.
        until: ISO timestamp; only return Jobs whose `created_at` is <= this.
        limit: Max number of Jobs to return (default 10, newest first).

    Returns:
        List of lean Job dicts (id, state, prompt[:120], created_at, resolved_at,
        worker_session_id, verifier_session_id, verify_attempts, error).
    """
    if _jobs_orchestrator is None:
        return [{"error": "Jobs orchestrator not initialised"}]
    jobs = _jobs_orchestrator._jobs.list_all()
    if session_id:
        jobs = [j for j in jobs if j.parent_session_id == session_id]
    if state:
        jobs = [j for j in jobs if j.state == state]
    if since:
        jobs = [j for j in jobs if (j.created_at or "") >= since]
    if until:
        jobs = [j for j in jobs if (j.created_at or "") <= until]
    # Newest first.
    jobs.sort(key=lambda j: j.created_at or "", reverse=True)
    return [
        {
            "id": j.id,
            "state": j.state,
            "prompt": (j.prompt or "")[:120],
            "created_at": j.created_at,
            "resolved_at": j.resolved_at,
            "worker_session_id": j.worker_session_id,
            "verifier_session_id": j.verifier_session_id,
            "verify_attempts": j.verify_attempts,
            "error": (j.error or "").splitlines()[0][:160] if j.error else None,
        }
        for j in jobs[: max(1, int(limit))]
    ]
