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
    name="compact",
    description="Compact the current conversation to free context space",
    params=[
        CommandParam("user_id", str, "User whose session to compact"),
        CommandParam(
            "message", str, "Extra instructions for compaction (e.g. remember/forget specific things)", required=False
        ),
    ],
)
async def cmd_compact(adapter: BaseAdapter, user_id: str, message: str | None = None) -> str:
    """Compact the interactive session for the given user."""
    session = adapter.session_store.get_or_create_interactive(user_id, adapter.agent_name)

    if session.message_count == 0:
        return "No conversation to compact."

    if not adapter.session_store.begin_compaction(user_id, adapter.agent_name):
        return "Compaction already in progress."

    old_id = session.id
    adapter._broadcast_compaction(adapter.agent_name, started=True)
    try:
        await adapter._compact_session(session.id, instructions=message, reason="manual")
    except Exception as e:
        return f"Compaction failed: {e}"
    finally:
        adapter.session_store.end_compaction(user_id, adapter.agent_name)
        adapter._broadcast_compaction(adapter.agent_name, started=False)

    new_session = adapter.session_store.get_or_create_interactive(user_id, adapter.agent_name)
    return f"Session compacted (old: {old_id[:12]}, new: {new_session.id[:12]})"


@adapter_command(
    name="status",
    description="Show agent status and context usage",
    params=[CommandParam("user_id", str, "User to check status for")],
)
async def cmd_status(adapter: BaseAdapter, user_id: str) -> str:
    """Show current agent status, token usage, and context window info."""
    session = adapter.session_store.get_or_create_interactive(user_id, adapter.agent_name)
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
