"""Pure hook-decision functions (no I/O) for the cc-driver.

Each function maps a Claude Code hook payload (see hook-payloads-reference.md) to
a decision, so the adapter's HTTP route stays a thin dispatcher and the driving
logic is unit-testable in isolation.

Driving protocol, per the spike findings:
  - The completion marker + "a supervisor may give follow-ups" note is baked into
    the INITIAL prompt (SessionStart hooks do not fire from --settings).
  - All injected/continue text is phrased as natural task guidance - Claude
    refuses injection-shaped instructions ("output exactly X and nothing else").
  - `stop_hook_active` is True only on a turn we drove; False marks a fresh
    human/CLI turn, which resets the continue budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def build_initial_prompt(goal: str, completion_marker: str, *, needs_input_marker: str) -> str:
    """The first-arg goal seed. Bakes the completion AND needs-input protocol
    into the prompt since SessionStart can't inject it. Phrased as natural
    guidance."""
    return (
        f"{goal}\n\n"
        f"When the task is fully complete, end your reply with the token "
        f"{completion_marker} so your supervisor knows you are done. A supervisor "
        f"may give you follow-up instructions after you stop; keep working until "
        f"the task genuinely meets its goal. If you are blocked on information or "
        f"a decision you cannot obtain yourself, do not guess - end your reply "
        f"with the line {needs_input_marker}: <your question> and stop; your "
        f"supervisor will answer."
    )


def continue_instruction(completion_marker: str, needs_input_marker: str) -> str:
    """Natural-language nudge injected via a Stop `additionalContext` block to
    drive another turn. Never phrased as 'output exactly X' (Claude refuses that)."""
    return (
        "You do not appear to be finished with the task yet. Please keep working "
        "toward the goal. When the task is genuinely complete, end your reply with "
        f"the token {completion_marker} so your supervisor knows you are done. If "
        f"you are blocked on something only your supervisor can answer, end your "
        f"reply with the line {needs_input_marker}: <your question> instead of guessing."
    )


@dataclass
class StopDecision:
    """Outcome of a Stop hook: the JSON body to return to Claude plus whether the
    attempt is finished, an optional needs-input question (a within-attempt
    pause), and the updated continue budget the caller must persist."""

    response: dict
    complete: bool
    summary: Optional[str]
    new_consecutive_continues: int
    needs_input: Optional[str] = None


def decide_stop(
    payload: dict,
    *,
    consecutive_continues: int,
    max_consecutive_continues: int,
    completion_marker: str,
    needs_input_marker: str,
) -> StopDecision:
    """Decide what to do when Claude stops.

    A fresh (human/CLI) turn resets the budget; the completion marker ends the
    attempt (the verifier then grades it); the needs-input marker pauses the
    attempt - checked BEFORE budget exhaustion, or a blocked worker near the
    nudge cap would be force-completed into a verification it already knows it
    can't pass; an exhausted continue budget also ends the attempt (maybe it IS
    done - let the verifier decide); otherwise return a `decision: block` that
    drives one more turn.
    """
    last_msg = payload.get("last_assistant_message") or ""
    stop_hook_active = bool(payload.get("stop_hook_active"))
    # A turn we did not drive (human/CLI) resets the continue budget.
    count = consecutive_continues if stop_hook_active else 0

    if completion_marker and completion_marker in last_msg:
        return StopDecision(response={}, complete=True, summary=last_msg, new_consecutive_continues=count)

    if needs_input_marker and needs_input_marker in last_msg:
        question = last_msg.split(needs_input_marker, 1)[1].lstrip(":").strip()[:500]
        return StopDecision(
            response={},
            complete=False,
            summary=None,
            new_consecutive_continues=count,
            needs_input=question or "the worker asked for supervisor input (no question text)",
        )

    if count >= max_consecutive_continues:
        # Budget exhausted: end the attempt and let the verifier judge it.
        return StopDecision(response={}, complete=True, summary=last_msg, new_consecutive_continues=count)

    block = {
        "decision": "block",
        "hookSpecificOutput": {
            "hookEventName": "Stop",
            "additionalContext": continue_instruction(completion_marker, needs_input_marker),
        },
    }
    return StopDecision(response=block, complete=False, summary=None, new_consecutive_continues=count + 1)


def decide_stop_failure(payload: dict) -> str:
    """Extract a failure reason from a StopFailure payload (a Stop hook / turn that
    errored). The caller routes this into fail_worker."""
    return (
        payload.get("error")
        or payload.get("message")
        or payload.get("last_assistant_message")
        or "Claude Code reported a Stop failure"
    )


def notification_attention(payload: dict) -> Optional[str]:
    """Return a human-readable attention message when a Notification is a
    permission prompt (the job may be blocked awaiting input), else None."""
    if payload.get("notification_type") == "permission_prompt":
        return payload.get("message") or "Claude Code needs attention"
    return None
