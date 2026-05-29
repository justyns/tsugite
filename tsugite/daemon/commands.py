"""Adapter command registry — define commands once, auto-register across all adapters."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from tsugite.daemon.adapters.base import BaseAdapter

logger = logging.getLogger(__name__)

_COMMANDS: dict[str, AdapterCommand] = {}


@dataclass
class CommandParam:
    name: str
    type: type
    description: str
    required: bool = True
    choices: list[str] | None = None


@dataclass
class AdapterCommand:
    name: str
    description: str
    handler: Callable
    params: list[CommandParam] = field(default_factory=list)


def adapter_command(
    name: str,
    description: str,
    params: list[CommandParam] | None = None,
):
    """Decorator to register an adapter command."""

    def decorator(fn: Callable) -> Callable:
        if name in _COMMANDS:
            logger.warning("Overwriting existing adapter command '%s'", name)
        _COMMANDS[name] = AdapterCommand(
            name=name,
            description=description,
            handler=fn,
            params=params or [],
        )
        return fn

    return decorator


def get_commands() -> dict[str, AdapterCommand]:
    return _COMMANDS


# ---------------------------------------------------------------------------
# Built-in commands
# ---------------------------------------------------------------------------


@adapter_command(
    name="bg",
    description="Run a task in the background",
    params=[
        CommandParam("prompt", str, "The task to run"),
        CommandParam("agent", str, "Target agent", required=False),
    ],
)
async def cmd_bg(adapter: BaseAdapter, prompt: str, agent: str | None = None) -> str:
    """Start a background session with the given prompt."""
    from tsugite.daemon.session_store import Session, SessionSource
    from tsugite.tools.sessions import _session_runner

    if not _session_runner:
        return "Background sessions require the daemon session runner to be enabled."

    target_agent = agent or adapter.agent_name

    session = Session(
        id="",
        agent=target_agent,
        source=SessionSource.BACKGROUND.value,
        prompt=prompt,
    )

    try:
        result = _session_runner.start_session(session)
    except Exception as e:
        return f"Failed to start background session: {e}"

    return f"Background session started (ID: {result.id})"


@adapter_command(
    name="job",
    description="Spawn a background Job with optional acceptance criteria, verified by a sub-agent on completion",
    params=[
        CommandParam("user_id", str, "User in whose chat this Job is anchored"),
        CommandParam(
            "session_id",
            str,
            "Active session that should host the Job tile (auto-injected by the web UI from the currently-open chat)",
            required=False,
        ),
        CommandParam("prompt", str, "The task to run as a Job"),
        CommandParam(
            "acceptance_criteria",
            str,
            "Pipe-separated free-text criteria the verifier grades against (e.g. 'tests pass|PR open'). Or a JSON array.",
            required=False,
        ),
        CommandParam("repo", str, "Workspace-relative repo path (persisted; enforcement deferred)", required=False),
        CommandParam("model", str, "Model override; defaults to workspace default", required=False),
        CommandParam("timeout_minutes", int, "Wall-clock timeout for the worker (default 30)", required=False),
        CommandParam("agent", str, "Worker agent file (default job_worker)", required=False),
    ],
)
async def cmd_job(
    adapter: BaseAdapter,
    user_id: str,
    prompt: str,
    session_id: str | None = None,
    acceptance_criteria: str | list[str] | None = None,
    repo: str | None = None,
    model: str | None = None,
    timeout_minutes: int | None = None,
    agent: str | None = None,
) -> str:
    """Create a Job, spawn a worker session, and return the Job + worker IDs."""
    from tsugite.tools.jobs import _jobs_orchestrator

    if _jobs_orchestrator is None:
        return "Jobs require the daemon session runner + orchestrator to be enabled."

    # Prefer the active session the user is in (passed from the web UI); fall back
    # to the user's primary session for that agent for non-UI callers (e.g. a
    # script POSTing directly to the API).
    parent_session_id: str | None = None
    if session_id:
        parent = (
            adapter.session_store.get_session(session_id) if hasattr(adapter.session_store, "get_session") else None
        )
        if parent is None:
            return f"Session '{session_id}' not found — cannot anchor Job."
        parent_session_id = parent.id
    else:
        parent = adapter.session_store.find_default_session(user_id, adapter.agent_name)
        if parent is None:
            return "No parent session found — send a message in this chat first, then run /job."
        parent_session_id = parent.id

    ac_list = _parse_acceptance_criteria(acceptance_criteria)

    try:
        job, started = _jobs_orchestrator.create_and_start_job(
            parent_session_id=parent_session_id,
            prompt=prompt,
            acceptance_criteria=ac_list,
            repo=repo,
            model=model,
            agent=agent,
            timeout_minutes=timeout_minutes or 30,
            spawned_by="user-slash",
        )
    except Exception as e:
        return f"Failed to spawn job worker: {e}"

    return f"Job {job.id} spawned (worker session: {started.id})"


def _parse_acceptance_criteria(raw: str | list[str] | None) -> list[str]:
    """Normalise the slash-command AC param into a list of strings.

    Accepts: None, an existing list (callers via Python API), JSON-array string,
    or pipe-separated string. Pipe is chosen over comma so AC texts can contain
    commas naturally.
    """
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    text = raw.strip()
    if text.startswith("["):
        try:
            import json

            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in text.split("|") if part.strip()]


@adapter_command(
    name="compact",
    description="Compact the conversation. Optional: add instructions to shape the summary",
    params=[
        CommandParam("user_id", str, "User whose session to compact"),
        CommandParam(
            "message", str, "Extra instructions for compaction (e.g. remember/forget specific things)", required=False
        ),
    ],
)
async def cmd_compact(adapter: BaseAdapter, user_id: str, message: str | None = None) -> str:
    """Compact the interactive session for the given user."""
    session = adapter.session_store.find_default_session(user_id, adapter.agent_name)
    if session is None or session.message_count == 0:
        return "No conversation to compact."

    old_id = session.id
    if not adapter.session_store.begin_compaction(user_id, adapter.agent_name, session_id=old_id):
        return "Compaction already in progress."

    adapter._broadcast_compaction("compaction_started", adapter.agent_name, old_id)
    new_session = None
    try:
        new_session = await adapter._compact_session(session.id, instructions=message, reason="manual")
    except Exception as e:
        return f"Compaction failed: {e}"
    finally:
        adapter.session_store.end_compaction(user_id, adapter.agent_name, session_id=old_id)
        adapter._broadcast_compaction("compaction_finished", adapter.agent_name, old_id)

    if new_session is None:
        return f"Nothing to compact (id: {old_id[:12]})"
    return f"Session compacted (old: {old_id[:12]}, new: {new_session.id[:12]})"


@adapter_command(
    name="status",
    description="Show agent status and context usage",
    params=[CommandParam("user_id", str, "User to check status for")],
)
async def cmd_status(adapter: BaseAdapter, user_id: str) -> str:
    """Show current agent status, token usage, and context window info."""
    session = adapter.session_store.find_default_session(user_id, adapter.agent_name)
    if session is None:
        return "No active session. Send a message to start one."
    context_limit = adapter.session_store.get_context_limit(adapter.agent_name)
    tokens = session.cumulative_tokens
    pct = int(tokens / context_limit * 100) if context_limit else 0
    compacting = adapter.session_store.is_compacting(user_id, adapter.agent_name)

    lines = [
        f"Model: {adapter.resolve_model()}",
        f"Context: {tokens:,} / {context_limit:,} tokens ({pct}%)",
        f"Messages: {session.message_count}",
    ]
    if compacting:
        lines.append("Compaction: in progress")
    return "\n".join(lines)


@adapter_command(
    name="context",
    description="Show prompt context breakdown by category",
    params=[CommandParam("user_id", str, "User to check context for")],
)
async def cmd_context(adapter: BaseAdapter, user_id: str) -> str:
    """Show per-category token breakdown from the latest prompt snapshot."""
    session = adapter.session_store.find_default_session(user_id, adapter.agent_name)
    if session is None:
        return "No active session. Send a message to start one."
    events = adapter.session_store.read_events(session.id)
    snapshots = [e for e in events if e.get("type") == "prompt_snapshot" and e.get("token_breakdown")]
    if not snapshots:
        return "No context data available yet. Send a message first."

    breakdown = snapshots[-1]["token_breakdown"]
    categories = breakdown.get("categories", [])
    total = breakdown.get("total", 0)

    def fmt(n):
        return f"{n:,}" if n < 1000 else f"{n / 1000:.1f}k"

    lines = [f"Context Breakdown (~{fmt(total)} tokens)"]
    for cat in categories:
        if cat["tokens"] == 0:
            continue
        name = cat["name"]
        if cat.get("items"):
            name += f" ({len(cat['items'])})"
        lines.append(f"  {name:<20} {fmt(cat['tokens']):>6}")
    return "\n".join(lines)


@adapter_command(
    name="sessions",
    description="List active and recent background sessions",
    params=[CommandParam("status", str, "Filter by status (running, completed, failed)", required=False)],
)
async def cmd_sessions(adapter: BaseAdapter, status: str | None = None) -> str:
    """List background sessions for the current agent."""
    sessions = adapter.session_store.list_sessions(agent=adapter.agent_name, status=status)
    if not sessions:
        return "No sessions found."
    lines = []
    for s in sessions[:10]:
        label = s.title or (s.prompt or "")[:60]
        lines.append(f"[{s.status}] {s.id[:12]} — {label}")
    if len(sessions) > 10:
        lines.append(f"... and {len(sessions) - 10} more")
    return "\n".join(lines)
