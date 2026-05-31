"""Jobs tool — `spawn_job()` for agents, mirroring spawn_session for plain sessions.

Tool uses @tool(require_daemon=True) so it only appears in daemon mode.
"""

import asyncio
from typing import Optional

from . import tool

_jobs_orchestrator = None
_loop = None


def set_jobs_orchestrator(orchestrator, loop):
    """Called by the daemon to set/clear the orchestrator reference."""
    global _jobs_orchestrator, _loop
    _jobs_orchestrator = orchestrator
    _loop = loop


def _call(fn, *args, timeout=30, **kwargs):
    """Call an orchestrator method on the daemon loop thread (thread-safe)."""

    async def _wrapper():
        result = fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    future = asyncio.run_coroutine_threadsafe(_wrapper(), _loop)
    return future.result(timeout=timeout)


@tool(require_daemon=True)
def spawn_job(
    prompt: str,
    acceptance_criteria: Optional[list] = None,
    repo: Optional[str] = None,
    model: Optional[str] = None,
    timeout_minutes: int = 30,
    agent: Optional[str] = None,
    notify: bool = True,
    max_attempts: Optional[int] = None,
    notify_when: Optional[str] = None,
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
        acceptance_criteria: List of criteria the verifier grades against. Each
            entry may be a plain string, a `text::kind` string (where kind is
            ui|test|cmd|llm), or a dict `{text, kind}`. Empty list short-circuits
            verification (Job goes straight to done).
        repo: Workspace-relative repo path. Accepted and persisted; enforcement
            (chroot/worktree) is deferred — for now this is informational.
        model: Optional model override; defaults to the workspace default.
        timeout_minutes: Wall-clock budget for the worker; on expiry the Job
            transitions to `stuck`.
        agent: Worker agent file. Defaults to `job_worker`.
        notify: Deprecated. When True (default for agent tool use) maps to
            notify_when="terminal" — the parent wakes on any terminal state.
            Prefer notify_when for finer control.
        max_attempts: Verifier-loop cap before stuck. Defaults to 3.
        notify_when: When to wake the parent: done | stuck | errored | terminal |
            never. Overrides `notify` when set.

    Returns:
        Dict with job_id, worker_session_id, parent_session_id, state.
    """
    from tsugite.daemon.session_runner import get_current_session_id

    parent_session_id = get_current_session_id()
    if not parent_session_id:
        raise RuntimeError("spawn_job requires a current session context")
    if _jobs_orchestrator is None:
        raise RuntimeError("Jobs orchestrator not initialised")

    job, started = _call(
        _jobs_orchestrator.create_and_start_job,
        parent_session_id=parent_session_id,
        prompt=prompt,
        acceptance_criteria=acceptance_criteria,
        repo=repo,
        model=model,
        agent=agent,
        timeout_minutes=timeout_minutes,
        spawned_by="agent-tool",
        notify=notify,
        max_attempts=max_attempts,
        notify_when=notify_when,
    )

    return {
        "job_id": job.id,
        "worker_session_id": started.id,
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
    jobs = list(_jobs_orchestrator._jobs._jobs.values())
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
