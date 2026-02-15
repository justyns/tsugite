"""Schedule tools for agents to manage daemon schedules directly.

Tools use @tool(require_daemon=True) so they only appear in daemon mode.
"""

import asyncio
from dataclasses import asdict
from typing import Optional

from . import tool

_scheduler = None
_loop = None
_channel_names: set[str] = set()


def set_scheduler(scheduler, loop=None, channel_names=None):
    """Called by the daemon to set/clear the scheduler reference."""
    global _scheduler, _loop, _channel_names
    _scheduler = scheduler
    _loop = loop
    _channel_names = channel_names or set()


def _call(fn, *args, **kwargs):
    """Call a scheduler method on the event loop thread (thread-safe)."""

    async def _wrapper():
        return fn(*args, **kwargs)

    future = asyncio.run_coroutine_threadsafe(_wrapper(), _loop)
    return future.result(timeout=10)


@tool(require_daemon=True)
def schedule_create(
    id: str,
    prompt: str,
    agent: Optional[str] = None,
    cron: Optional[str] = None,
    run_at: Optional[str] = None,
    timezone: str = "UTC",
    notify: Optional[list[str]] = None,
    notify_tool: bool = False,
) -> dict:
    """Create a recurring (cron) or one-off schedule to run an agent.

    IMPORTANT: Always confirm with the user before calling this tool. Show them the exact
    prompt, schedule, and timezone you plan to use and wait for approval. Never schedule
    destructive or dangerous actions (file deletion, force-push, infrastructure changes, etc.).

    Args:
        id: Unique schedule name (e.g., "daily-backup")
        prompt: Clear, direct instruction for the agent. Do NOT copy the user's words verbatim — interpret their intent and write a self-contained instruction the agent can execute autonomously.
        agent: Agent name configured in daemon. Defaults to the current agent if omitted.
        cron: Cron expression for recurring (e.g., "0 9 * * *" = daily at 9am). Mutually exclusive with run_at.
        run_at: ISO datetime for one-off execution (e.g., "2026-02-13T14:00:00-06:00"). Mutually exclusive with cron.
        timezone: IANA timezone (default: UTC)
        notify: List of notification channel names to deliver results to on completion.
        notify_tool: If true, gives the agent the notify_user tool so it can send messages during execution. Requires notify to be set.

    Returns:
        Created schedule details including computed next_run
    """
    if not cron and not run_at:
        raise ValueError("Provide either 'cron' or 'run_at'")
    if cron and run_at:
        raise ValueError("Provide 'cron' or 'run_at', not both")

    if notify_tool and not notify:
        raise ValueError("notify_tool=True requires a non-empty 'notify' list")
    if notify:
        unknown = set(notify) - _channel_names
        if unknown:
            raise ValueError(f"Unknown notification channel(s): {', '.join(sorted(unknown))}")

    if agent is None:
        from tsugite.agent_runner.helpers import get_current_agent

        agent = get_current_agent() or "default"

    from tsugite.daemon.scheduler import ScheduleEntry

    entry = ScheduleEntry(
        id=id,
        agent=agent,
        prompt=prompt,
        schedule_type="once" if run_at else "cron",
        cron_expr=cron,
        run_at=run_at,
        notify=notify or [],
        notify_tool=notify_tool,
        timezone=timezone,
    )
    result = _call(_scheduler.add, entry)
    return asdict(result)


@tool(require_daemon=True)
def schedule_list() -> list:
    """List all configured schedules with their status.

    Returns:
        List of schedules with id, agent, type, enabled, next_run, last_status
    """
    entries = _call(_scheduler.list)
    return [asdict(e) for e in entries]


@tool(require_daemon=True)
def schedule_remove(id: str) -> dict:
    """Remove a schedule.

    Args:
        id: Schedule ID to remove

    Returns:
        Confirmation of removal
    """
    _call(_scheduler.remove, id)
    return {"status": "removed", "id": id}


@tool(require_daemon=True)
def schedule_enable(id: str) -> dict:
    """Enable a disabled schedule.

    Args:
        id: Schedule ID to enable

    Returns:
        Confirmation with updated schedule details
    """
    _call(_scheduler.enable, id)
    return asdict(_call(_scheduler.get, id))


@tool(require_daemon=True)
def schedule_disable(id: str) -> dict:
    """Disable a schedule without removing it.

    Args:
        id: Schedule ID to disable

    Returns:
        Confirmation with updated schedule details
    """
    _call(_scheduler.disable, id)
    return asdict(_call(_scheduler.get, id))


@tool(require_daemon=True)
def schedule_update(
    id: str,
    prompt: Optional[str] = None,
    cron: Optional[str] = None,
    run_at: Optional[str] = None,
    timezone: Optional[str] = None,
    notify: Optional[list[str]] = None,
    notify_tool: Optional[bool] = None,
) -> dict:
    """Update fields on an existing schedule.

    Args:
        id: Schedule ID to update
        prompt: New prompt text (optional)
        cron: New cron expression (optional)
        run_at: New run_at ISO datetime (optional)
        timezone: New IANA timezone (optional)
        notify: New notification channel list (optional)
        notify_tool: Enable/disable notify_user tool (optional)

    Returns:
        Updated schedule details
    """
    fields = {}
    if prompt is not None:
        fields["prompt"] = prompt
    if cron is not None:
        fields["cron_expr"] = cron
    if run_at is not None:
        fields["run_at"] = run_at
    if timezone is not None:
        fields["timezone"] = timezone
    if notify is not None:
        unknown = set(notify) - _channel_names
        if unknown:
            raise ValueError(f"Unknown notification channel(s): {', '.join(sorted(unknown))}")
        fields["notify"] = notify
    if notify_tool is not None:
        if notify_tool:
            effective_notify = notify if notify is not None else _call(_scheduler.get, id).notify
            if not effective_notify:
                raise ValueError("notify_tool=True requires a non-empty 'notify' list")
        fields["notify_tool"] = notify_tool

    if not fields:
        raise ValueError("No fields to update")

    result = _call(_scheduler.update, id, **fields)
    return asdict(result)
