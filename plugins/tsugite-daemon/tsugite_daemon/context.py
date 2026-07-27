"""Context captures: attach a specific session or job as chat context by an
explicit action, never by scanning the message for ids.

Registered under the ``tsugite.context_providers`` entry point (``daemon``). Two
capture-only providers, ``session`` and ``job``, each build a compact summary for
a given record id and return it as a single trusted context item. The terminal
case reuses tsugite-pty's own ``terminal`` provider, so it is not duplicated here.
All three are driven from the web UI (an "add to chat" button or a reference
paste) via ``POST /api/context-providers/{key}/capture``, so ``session`` and
``job`` set ``menu=False``: they stay out of the composer's add-context menu and
are not detectors, only reachable through the capture endpoint.

These are the user's own records, so items attach as trusted context (not
``untrusted`` like a fetched web page). A store whose seam isn't wired in this
process (no jobs orchestrator) yields an empty capture, never an error.
"""

from __future__ import annotations

import logging
from typing import Optional

from tsugite.attachments.base import Attachment
from tsugite.context import ContextProvider, register_context_provider
from tsugite.history import get_history_backend
from tsugite.tools.jobs import get_jobs_orchestrator

logger = logging.getLogger(__name__)

# Per-item value cap, matching the sibling providers (the fold enforces the same).
_MAX_VALUE_CHARS = 4000
# A derived session/job title is the head of the first user message / the prompt.
_TITLE_CHARS = 60
# The last exchange (and a job prompt) is a compact preview, not a transcript.
_PREVIEW_CHARS = 400
# A job's last error is shown as a single line.
_ERROR_CHARS = 200


def _session_status(summary) -> str:
    """A human status derived from the session summary: ``active`` while no
    ``session_end`` has landed, ``completed`` for a success end, else the raw end
    status (``error`` / ``interrupted``) with its message when present."""
    raw = summary.status
    if raw is None:
        return "active"
    if raw == "success":
        return "completed"
    message = (summary.error_message or "").strip()
    return f"{raw}: {message[:_ERROR_CHARS]}" if message else raw


def capture_session(arg: Optional[str], context: dict) -> list[Attachment]:
    """A compact summary for a real session id, or ``[]`` when ``arg`` is empty or
    names no session.

    Sourced from ``summary()`` (status/model/message count/token total/last
    assistant reply); the title and last user message come from the ``user_input``
    events. Deliberately a summary, not a transcript.
    """
    if not arg:
        return []
    backend = get_history_backend()
    if not backend.exists(arg):
        return []
    session = backend.load(arg)
    summary = session.summary()
    user_texts = [(e.data.get("text") or "").strip() for e in session.iter_events(("user_input",))]
    user_texts = [text for text in user_texts if text]

    title = user_texts[0][:_TITLE_CHARS] if user_texts else "(no messages)"
    # Name the record so the model knows what it's looking at: "session:<id>" and
    # bare title/status don't say this is a past tsugite conversation.
    lines = [
        "kind: tsugite session (a past conversation with this agent, summarized below)",
        f"title: {title}",
        f"status: {_session_status(summary)}",
    ]
    if summary.model:
        lines.append(f"model: {summary.model}")
    lines.append(f"messages: {summary.turn_count}")
    if summary.total_tokens:
        lines.append(f"tokens: {summary.total_tokens}")
    if user_texts:
        lines.append(f"last user: {user_texts[-1][:_PREVIEW_CHARS]}")
    last_assistant = (summary.last_response_text or "").strip()
    if last_assistant:
        lines.append(f"last assistant: {last_assistant[:_PREVIEW_CHARS]}")

    value = "\n".join(lines)[:_MAX_VALUE_CHARS]
    return [Attachment.context(key=f"session:{arg}", label=title if user_texts else arg, value=value)]


def capture_job(arg: Optional[str], context: dict) -> list[Attachment]:
    """A compact job record for a real job id, or ``[]`` when ``arg`` is empty, the
    orchestrator is unwired (no daemon), or the id is unknown."""
    if not arg:
        return []
    orchestrator = get_jobs_orchestrator()
    if orchestrator is None:
        return []
    job = orchestrator.get_job(arg)
    if job is None:
        return []

    prompt = (job.prompt or "").strip()
    lines = [
        "kind: tsugite background job",
        f"state: {job.state}",
        f"executor: {job.executor}",
        f"attempts: {job.verify_attempts}/{job.max_attempts}",
        f"prompt: {prompt[:_PREVIEW_CHARS]}",
    ]
    if job.error:
        first_line = job.error.splitlines()[0].strip() if job.error.strip() else ""
        if first_line:
            lines.append(f"last error: {first_line[:_ERROR_CHARS]}")

    value = "\n".join(lines)[:_MAX_VALUE_CHARS]
    return [Attachment.context(key=f"job:{arg}", label=prompt[:_TITLE_CHARS] or arg, value=value)]


register_context_provider(
    ContextProvider(key="session", label="Session", icon="chat", capture=capture_session, menu=False)
)
register_context_provider(ContextProvider(key="job", label="Job", icon="jobs", capture=capture_job, menu=False))
