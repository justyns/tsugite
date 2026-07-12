"""Adapter command registry — define commands once, auto-register across all adapters."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from tsugite_daemon.adapters.base import BaseAdapter

logger = logging.getLogger(__name__)

_COMMANDS: dict[str, AdapterCommand] = {}


class CommandError(Exception):
    """User-facing command failure.

    Adapters surface str(e) as the reply text; the HTTP command endpoint maps
    it to a 400 with a machine-readable error field so the web UI can branch on
    status instead of prefix-matching English prose.
    """


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
    from tsugite.tools.sessions import _session_runner
    from tsugite_daemon.session_store import Session, SessionSource

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
        CommandParam("prompt", str, "The task to run as a Job"),
        CommandParam(
            "session_id",
            str,
            "Active session that should host the Job tile (auto-injected by the web UI from the currently-open chat)",
            required=False,
        ),
        CommandParam(
            "acceptance_criteria",
            str,
            "Pipe-separated free-text criteria the verifier grades against (e.g. 'tests pass|PR open'). Or a JSON array.",
            required=False,
        ),
        CommandParam("repo", str, "Workspace-relative repo path (persisted; enforcement deferred)", required=False),
        CommandParam("model", str, "Model override; defaults to workspace default", required=False),
        CommandParam(
            "model_ladder",
            str,
            "Pipe-separated cheap-first escalation ladder (e.g. 'claude_code:haiku|claude_code:opus'); overrides model",
            required=False,
        ),
        CommandParam("timeout_minutes", int, "Per-phase timeout, re-armed each phase (default 30)", required=False),
        CommandParam("agent", str, "Worker agent file (default job_worker)", required=False),
        CommandParam(
            "max_attempts",
            int,
            "Max verifier rounds before stuck (default 3)",
            required=False,
        ),
        CommandParam(
            "notify_when",
            str,
            "When to wake the parent: done|stuck|errored|terminal|never (default never)",
            required=False,
        ),
        CommandParam(
            "executor",
            str,
            "Which registered executor runs the job (default agent)",
            required=False,
        ),
        CommandParam(
            "effort",
            str,
            "Reasoning effort for the worker: low|medium|high|xhigh|max (cc executor -> claude --effort)",
            required=False,
        ),
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
    model_ladder: str | list[str] | None = None,
    timeout_minutes: int | None = None,
    agent: str | None = None,
    max_attempts: int | None = None,
    notify_when: str | None = None,
    executor: str | None = None,
    effort: str | None = None,
) -> str:
    """Create a Job, spawn a worker session, and return the Job + worker IDs."""
    from tsugite.tools.jobs import _jobs_orchestrator

    if _jobs_orchestrator is None:
        raise CommandError("Jobs require the daemon session runner + orchestrator to be enabled.")

    # Resolve where the Job's tile lives:
    #   - session_id given (composer `/job` typed inside a chat): anchor there.
    #     A stale/invalid id returns a friendly message rather than a 500 -
    #     get_session raises ValueError, it never returns None.
    #   - no session_id (the Jobs-tab "new job" modal, or `/job` with no chat
    #     open): provision a fresh host session. The old fallback guessed the
    #     user's primary chat via find_default_session, which silently attached
    #     Jobs-tab jobs to whatever conversation happened to be primary - the
    #     "wrong session" the tile showed up under.
    if session_id:
        try:
            parent = adapter.session_store.get_session(session_id)
        except (ValueError, KeyError):
            raise CommandError(f"Session '{session_id}' not found - cannot anchor Job.") from None
        parent_session_id = parent.id
    else:
        parent_session_id = _create_job_host_session(adapter, user_id, prompt)

    ac_list = _parse_acceptance_criteria(acceptance_criteria)
    ladder = model_ladder.split("|") if isinstance(model_ladder, str) and model_ladder else model_ladder

    try:
        job, started = await _jobs_orchestrator.create_and_start_job(
            parent_session_id=parent_session_id,
            prompt=prompt,
            acceptance_criteria=ac_list,
            repo=repo,
            model=model,
            model_ladder=ladder,
            agent=agent,
            timeout_minutes=timeout_minutes or 30,
            spawned_by="user-slash",
            max_attempts=max_attempts,
            notify_when=notify_when,
            executor=executor or "agent",
            effort=effort,
        )
    except Exception as e:
        raise CommandError(f"Failed to spawn job worker: {e}") from e

    if started is None:
        return f"Job {job.id} spawned (executor: {executor or 'agent'})"
    return f"Job {job.id} spawned (worker session: {started.id})"


def _create_job_host_session(adapter: "BaseAdapter", user_id: str, prompt: str) -> str:
    """Provision a fresh, non-primary interactive session to host a Job spawned
    outside any conversation (the Jobs-tab "new job" modal).

    Returns the new session id. Titled from the prompt's first line so it's
    recognisable in the sidebar, and broadcast as a session_update so the UI
    surfaces it without a manual refresh. Deliberately NOT marked primary - a Job
    must never steal the user's primary-chat flag.
    """
    from tsugite_daemon.session_store import METADATA_JOB_HOST, create_interactive_session

    first_line = next((ln.strip() for ln in (prompt or "").splitlines() if ln.strip()), "")
    title = f"Job: {first_line[:60]}".rstrip() if first_line else "Job"
    return create_interactive_session(
        adapter.session_store,
        adapter.agent_name,
        user_id,
        title=title,
        event_bus=getattr(adapter, "event_bus", None),
        metadata={METADATA_JOB_HOST: True},
    )


def _resolve_command_session(adapter: "BaseAdapter", user_id: str, session_id: str | None):
    """Resolve the session a session-scoped command targets: the explicitly
    passed id (the web UI auto-injects the open chat's id) or the user's
    default/primary session (the Discord and legacy path). Returns None when
    neither resolves; a stale explicit id falls back to the default rather
    than erroring, mirroring cmd_job's tolerance."""
    if session_id:
        try:
            return adapter.session_store.get_session(session_id)
        except (ValueError, KeyError):
            pass
    return adapter.session_store.find_default_session(user_id, adapter.agent_name)


def _parse_acceptance_criteria(raw: str | list | None) -> list[str]:
    """Normalise the slash-command AC param into a plain list of strings.

    Accepts: None, an existing list, JSON-array string, or pipe-separated
    string. Pipe is chosen over comma so AC texts can contain commas naturally.
    """
    from tsugite_daemon.job_store import _coerce_ac_list

    if not raw:
        return []
    if isinstance(raw, list):
        return _coerce_ac_list(raw)
    text = raw.strip()
    if text.startswith("["):
        try:
            import json

            parsed = json.loads(text)
            if isinstance(parsed, list):
                return _coerce_ac_list(parsed)
        except json.JSONDecodeError:
            pass
    return _coerce_ac_list([part for part in text.split("|") if part.strip()])


@adapter_command(
    name="compact",
    description="Compact the conversation. Optional: add instructions to shape the summary",
    params=[
        CommandParam("user_id", str, "User whose session to compact"),
        CommandParam(
            "message", str, "Extra instructions for compaction (e.g. remember/forget specific things)", required=False
        ),
        CommandParam(
            "session_id",
            str,
            "Session to compact (auto-injected by the web UI from the open chat)",
            required=False,
        ),
    ],
)
async def cmd_compact(
    adapter: BaseAdapter, user_id: str, message: str | None = None, session_id: str | None = None
) -> str:
    """Compact the targeted session (the open chat in the web UI, else the user's default)."""
    session = _resolve_command_session(adapter, user_id, session_id)
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
    params=[
        CommandParam("user_id", str, "User to check status for"),
        CommandParam(
            "session_id",
            str,
            "Session to inspect (auto-injected by the web UI from the open chat)",
            required=False,
        ),
    ],
)
async def cmd_status(adapter: BaseAdapter, user_id: str, session_id: str | None = None) -> str:
    """Show current agent status, token usage, and context window info."""
    session = _resolve_command_session(adapter, user_id, session_id)
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
    params=[
        CommandParam("user_id", str, "User to check context for"),
        CommandParam(
            "session_id",
            str,
            "Session to inspect (auto-injected by the web UI from the open chat)",
            required=False,
        ),
    ],
)
async def cmd_context(adapter: BaseAdapter, user_id: str, session_id: str | None = None) -> str:
    """Show per-category token breakdown from the latest prompt snapshot."""
    session = _resolve_command_session(adapter, user_id, session_id)
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


@adapter_command(
    name="run",
    description="Spawn a terminal session running the given command",
    params=[
        CommandParam("cmd", str, "Command to run in the terminal"),
        CommandParam("cwd", str, "Working directory", required=False),
        CommandParam("parent_session_id", str, "Chat session that spawned this terminal", required=False),
    ],
)
async def cmd_run(
    adapter: BaseAdapter,
    cmd: str,
    cwd: str | None = None,
    parent_session_id: str | None = None,
) -> str:
    """Spawn a PTY-backed terminal session. Returns the terminal id for the
    frontend to navigate to and stream output from."""
    from tsugite_pty.terminal_runtime import spawn_terminal

    terminal_store = getattr(adapter, "terminal_store", None)
    pty_manager = getattr(adapter, "pty_manager", None)
    if terminal_store is None or pty_manager is None:
        raise CommandError("Terminal sessions require the daemon terminal runtime to be enabled.")

    on_state_change = getattr(adapter, "terminal_state_change_callback", None)
    try:
        terminal = spawn_terminal(
            store=terminal_store,
            manager=pty_manager,
            cmd=cmd,
            cwd=cwd,
            parent_session_id=parent_session_id,
            on_state_change=on_state_change,
        )
    except ValueError as e:
        raise CommandError(f"Invalid command: {e}") from e
    except Exception as e:
        raise CommandError(f"Failed to spawn terminal: {e}") from e
    return f"Terminal started (id: {terminal.id}, state: {terminal.state})"
