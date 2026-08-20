"""Prompt and message text for Jobs: worker, verifier, hint, followup, notify.

Pure string building - no orchestrator state, no I/O.
"""

from collections import Counter
from typing import Optional

from tsugite_daemon.job_store import Job, JobState


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


def _retry_context_lines(job: Job) -> list[str]:
    """Shared preamble for retry/hint worker prompts.

    Retry workers are FRESH sessions with none of the prior attempt's
    conversation. Without the original task restated, the worker has nothing to
    ground its work in and (observed live) tends to emit a fabricated summary
    with zero tool calls instead of doing the work.
    """
    return [
        "You are a fresh session retrying an earlier Job attempt. Any files the previous",
        "attempt produced are in your working directory; you have none of its conversation.",
        "",
        "Original task:",
        "",
        job.prompt,
        "",
        "Use tools to verify the current state and to make the changes - do not claim work",
        "you have not performed in this session.",
    ]


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
    parts.append("")
    parts.append(
        "You are running in the worker's working directory, so files it produced are "
        "directly inspectable. For any criterion about a file's existence, contents, or "
        "structure, read the actual file (`read_file`, or `run` for listings) before "
        "deciding - do not fail a criterion just because the worker's summary doesn't "
        "inline the contents."
    )
    if job.repo:
        parts.append(f"Repo: {job.repo}")
        parts.append("(use `run` to inspect `git diff` or `git log` if relevant.)")
    return "\n".join(parts)


def _build_hint_prompt(job: Job, hint: str) -> str:
    """Compose the retry prompt for a worker resurrected from STUCK/ERRORED.
    With no hint (a pure retry-on-different-model), restate the failure instead."""
    parts = _retry_context_lines(job)
    parts.append("")
    if hint.strip():
        parts.append("This job previously hit the verifier's max-attempt limit. A user provided a hint:")
        parts.append("")
        parts.append(hint)
    else:
        parts.append("A previous attempt of this job failed; you are a fresh retry (possibly on a different model).")
        if job.error:
            parts.append(f"The recorded failure was: {job.error[:500]}")
    parts.append("")
    parts.append("Acceptance criteria the verifier will check:")
    for i, ac in enumerate(job.acceptance_criteria, 1):
        parts.append(f"{i}. {ac}")
    parts.append("")
    parts.append("Address the hint, then produce the structured summary required by your instructions.")
    return "\n".join(parts)


def _build_followup_prompt(job: Job, failed_acs: list[dict]) -> str:
    parts = _retry_context_lines(job)
    parts.append("")
    parts.append("The verifier flagged these acceptance criteria as not met:")
    for ac in failed_acs:
        parts.append(f"- {ac.get('ac_text', '?')}: {ac.get('reason', '?')}")
    parts.append("")
    parts.append("Address them, then produce the structured summary required by your instructions.")
    return "\n".join(parts)


def _short_prompt(prompt: Optional[str]) -> str:
    text = prompt or ""
    if len(text) <= 80:
        return text
    return text[:80] + "…"


def _build_notify_message(job: Job) -> str:
    """One-line wake-up message posted to the parent session on terminal transition.
    Brief by design - the parent agent should call get_job(job_id) for details."""
    base = f"Job {job.id} finished with state '{job.state}': {_short_prompt(job.prompt)}"
    if job.state in (JobState.STUCK.value, JobState.ERRORED.value) and job.error:
        base += f" - error: {job.error.splitlines()[0][:200]}"
    elif job.state == JobState.CANCELLED.value and job.error:
        base += f" - {job.error[:120]}"
    base += f". Use get_job('{job.id}') for details."
    return base


def _build_barrier_message(jobs: list[Job]) -> str:
    counts = Counter(job.state for job in jobs)
    breakdown = ", ".join(f"{n} {state}" for state, n in sorted(counts.items()))
    lines = [f"All {len(jobs)} background job(s) finished ({breakdown})."]
    for job in jobs:
        line = f"- {job.id} [{job.state}]: {_short_prompt(job.prompt)}"
        if job.error:
            line += f" - {job.error.splitlines()[0][:200]}"
        lines.append(line)
    lines.append("Use get_job('<job_id>') for details on any of them.")
    return "\n".join(lines)
