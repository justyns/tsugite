"""Backend-driven approval primitive.

A human-in-the-loop gate that pauses a turn before an action and asks the user
to approve or deny it, once or persistently. It rides the same InteractionBackend
used by ``ask_user``, so an approval surfaces on whatever adapter the user is on
(CLI prompt, web SSE frame, etc.).

Fail-closed by design: when no interactive surface is available (no backend, or a
non-interactive scheduler/headless backend), or when the reply is unrecognized,
the decision is always ``"deny"`` and the user is never prompted. It never
auto-approves.
"""

from typing import Literal, Optional

from tsugite.interaction import NonInteractiveBackend, get_interaction_backend

ApprovalDecision = Literal["approve", "deny", "always", "never"]

APPROVE_LABEL = "Approve"
DENY_LABEL = "Deny"
ALWAYS_LABEL = "Always allow"
NEVER_LABEL = "Always deny"

_LABEL_TO_DECISION = {
    APPROVE_LABEL: "approve",
    DENY_LABEL: "deny",
    ALWAYS_LABEL: "always",
    NEVER_LABEL: "never",
}


def request_approval(
    prompt: str,
    *,
    allow_always: bool = False,
    allow_never: bool = False,
    extra_options: Optional[dict[str, str]] = None,
    detail: Optional[str] = None,
) -> str:
    """Ask the user to approve an action, returning an :data:`ApprovalDecision`.

    Offers ``Approve``/``Deny`` plus the optional choices below as an
    ``"approval"`` question on the active interaction backend, and maps the
    chosen label back to a decision.

    Args:
        prompt: The action to approve, phrased as a question.
        allow_always: Offer an "Always allow" option (a persistent allow).
        allow_never: Offer an "Always deny" option (a persistent deny).
        extra_options: Caller-specific choices, as ``{label: decision}``.
        detail: Extra context appended below the prompt for the user to read.

    Returns:
        One of ``"approve"``, ``"deny"``, ``"always"``, ``"never"``, or a value of ``extra_options``.

    Fail-closed: returns ``"deny"`` WITHOUT prompting when there is no backend or
    the backend is non-interactive, and for any unrecognized reply.
    """
    backend = get_interaction_backend()
    if backend is None or isinstance(backend, NonInteractiveBackend):
        return "deny"

    options = [APPROVE_LABEL, DENY_LABEL]
    if allow_always:
        options.append(ALWAYS_LABEL)
    if allow_never:
        options.append(NEVER_LABEL)
    decisions = _LABEL_TO_DECISION | (extra_options or {})
    options.extend(extra_options or ())

    question = f"{prompt}\n\n{detail}" if detail else prompt
    answer = backend.ask_user(question, "approval", options)
    return decisions.get(answer, "deny")
