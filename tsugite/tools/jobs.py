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
    acceptance_criteria: Optional[list[str]] = None,
    repo: Optional[str] = None,
    model: Optional[str] = None,
    timeout_minutes: int = 30,
    agent: Optional[str] = None,
) -> dict:
    """Spawn a background Job with a verification loop.

    Creates a Job record + worker session pinned to the workspace default model
    (or the supplied `model`). On worker completion the orchestrator spawns a
    verifier sub-agent that judges the result against `acceptance_criteria`. If
    verification fails, the Job loops back up to 3 times before going `stuck`.

    Use this instead of `spawn_session` when you want structured verification
    of the work, not just a fire-and-forget background session.

    Args:
        prompt: Task instruction for the spawned Job.
        acceptance_criteria: Free-text list of criteria the verifier grades against.
            Empty list short-circuits verification (Job goes straight to done).
        repo: Workspace-relative repo path. Accepted and persisted; enforcement
            (chroot/worktree) is deferred — for now this is informational.
        model: Optional model override; defaults to the workspace default.
        timeout_minutes: Wall-clock budget for the worker; on expiry the Job
            transitions to `stuck`.
        agent: Worker agent file. Defaults to `job_worker`.

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
    )

    return {
        "job_id": job.id,
        "worker_session_id": started.id,
        "parent_session_id": parent_session_id,
        "state": "running",
    }
