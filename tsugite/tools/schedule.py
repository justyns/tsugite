"""Schedule tools for agents to manage daemon schedules directly."""

from typing import Optional
from uuid import uuid4

from . import call_on_loop, deny_when_sandboxed, tool
from .sessions import CURRENT_SESSION, get_current_session_id


def _entry_to_dict(entry):
    """Lazily import the daemon serializer so this module loads without the daemon battery."""
    from tsugite_daemon.scheduler import entry_to_dict

    return entry_to_dict(entry)


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
    return call_on_loop(_loop, fn, *args, timeout=10, **kwargs)


def _validate_notify(notify, notify_tool: bool) -> Optional[list[str]]:
    """Normalize and validate the notify argument; returns the coerced list.

    Agents pass booleans and bare strings here in practice - those must coerce
    or produce a message that names the argument and the expected type, never
    leak "'bool' object is not iterable"."""
    if notify is True or (notify is not None and notify is not False and not isinstance(notify, (list, str))):
        channels = f"Available channels: {', '.join(sorted(_channel_names))}" if _channel_names else ""
        got = "True" if notify is True else type(notify).__name__
        raise ValueError(f"notify must be a list of channel names, got {got}. {channels}".strip())
    if notify is False:
        notify = None
    elif isinstance(notify, str):
        notify = [notify]
    if notify_tool and not notify:
        raise ValueError("notify_tool=True requires a non-empty 'notify' list")
    if notify:
        unknown = set(notify) - _channel_names
        if unknown:
            raise ValueError(f"Unknown notification channel(s): {', '.join(sorted(unknown))}")
    return notify


def _resolve_target_session(target_session: Optional[str], current_session_id: Optional[str]) -> Optional[str]:
    """Resolved at creation, not fire time: a schedule fires in its own session,
    where there is no current chat.
    """
    if target_session != CURRENT_SESSION:
        if target_session:
            from tsugite_daemon.session_store import alias_from_ref, validate_alias

            alias = alias_from_ref(target_session)
            if alias is not None:
                validate_alias(alias)
        return target_session
    if not current_session_id:
        raise ValueError('target_session="current" requires a session context (daemon mode)')
    return current_session_id


@tool(require_daemon=True)
@deny_when_sandboxed
def schedule_create(
    id: str,
    prompt: str = "",
    cron: Optional[str] = None,
    run_at: Optional[str] = None,
    timezone: str = "UTC",
    notify: Optional[list[str]] = None,
    notify_tool: bool = False,
    inject_history: bool = True,
    model: Optional[str] = None,
    agent_file: Optional[str] = None,
    execution_type: str = "agent",
    command: Optional[str] = None,
    script_timeout: int = 60,
    expires_at: Optional[str] = None,
    max_runs: Optional[int] = None,
    session_id: Optional[str] = None,
    target_session: Optional[str] = None,
    delivery_mode: str = "existing_session",
    delivery_kind: str = "fyi",
    incident_key: Optional[str] = None,
    incident_title: Optional[str] = None,
) -> dict:
    """Create a recurring (cron) or one-off schedule to run an agent or script.

    IMPORTANT: Always confirm with the user before calling this tool. Show them the exact
    prompt, schedule, and timezone you plan to use and wait for approval. Never schedule
    destructive or dangerous actions (file deletion, force-push, infrastructure changes, etc.).

    Args:
        id: Unique schedule name (e.g., "daily-backup")
        prompt: Clear, direct instruction for the agent. Do NOT copy the user's words verbatim — interpret their intent and write a self-contained instruction the agent can execute autonomously. Can be empty when agent_file is set or execution_type is "script". For "session_message" this is the message the target session receives.
        cron: Cron expression for recurring (e.g., "0 9 * * *" = daily at 9am). Mutually exclusive with run_at.
        run_at: ISO datetime for one-off execution (e.g., "2026-02-13T14:00:00-06:00"). Mutually exclusive with cron.
        timezone: IANA timezone (default: UTC)
        notify: List of notification channel names to deliver results to on completion.
        notify_tool: If true, gives the agent the notify_user tool so it can send messages during execution. Requires notify to be set.
        inject_history: If true (default), delivers the task result into the recipient's chat session so the agent has context when they reply.
        model: Optional model override (e.g., "openai:gpt-4o-mini"). When set, this schedule uses this model instead of the agent's default.
        agent_file: Agent name (e.g., "+reporter") or path to a tsugite agent .md file. Hot-loaded on each run — edit the file and the next execution picks up changes.
        execution_type: "agent" (default) runs an LLM agent, "script" runs a shell command directly without LLM,
            "session_message" sends `prompt` into target_session and lets that conversation take a turn. Reminders and
            follow-ups ("check on that job in 2 hours") want this one; it requires target_session, usually "current".
        command: Shell command to execute when execution_type is "script". Required for script type.
        script_timeout: Max seconds for script execution (default: 60). Only used when execution_type is "script".
        expires_at: ISO datetime after which the schedule auto-disables (e.g., "2026-04-01T00:00:00Z").
        max_runs: Auto-disable after this many successful executions.
        session_id: If set, all runs of this schedule share the same session (persistent LLM context across runs). If omitted, each run gets its own isolated session.
        target_session: Where the run's result is delivered. Distinct from session_id (which controls the agent's run session). Legal forms:
            None (default) - fallback chain: primary -> originating -> none
            "primary" - primary lookup only (no fallback)
            "originating" - originating_session_id only
            "none" - skip delivery
            "name:<n>" - the session holding alias <n>, created if none holds it
            "current" - the session creating the schedule, stored as its id
            "<sid>" - bare session id
        delivery_mode: Which session the result is delivered into. "existing_session" (default) uses target_session
            routing, "parent_session" the session that created the schedule, "new_session" a dedicated incident
            session, "auto" picks existing_session for fyi and new_session for needs_ack.
        delivery_kind: "fyi" (default) or "needs_ack". A needs_ack delivery flags the session as needing a reply
            and pings the configured notification channels.
        incident_key: Dedupe key so repeat firings of this schedule share one incident session instead of opening
            a new one each run.
        incident_title: Title for the incident session (default: "Incident: <id>").

    Returns:
        Created schedule details including computed next_run
    """
    if not cron and not run_at:
        raise ValueError("Provide either 'cron' or 'run_at'")
    if cron and run_at:
        raise ValueError("Provide 'cron' or 'run_at', not both")

    if execution_type == "script":
        if not command:
            raise ValueError("'command' is required when execution_type is 'script'")
    else:
        if not prompt and not agent_file:
            raise ValueError("Provide at least one of 'prompt' or 'agent_file'")

    notify = _validate_notify(notify, notify_tool)

    originating_session_id = get_current_session_id()
    target_session = _resolve_target_session(target_session, originating_session_id)

    from tsugite_daemon.scheduler import ScheduleEntry
    from tsugite_daemon.session_runner import get_current_chain_depth

    chain_depth = get_current_chain_depth() if execution_type == "session_message" else 0

    entry = ScheduleEntry(
        id=id,
        prompt=prompt,
        schedule_type="once" if run_at else "cron",
        cron_expr=cron,
        run_at=run_at,
        notify=notify or [],
        notify_tool=notify_tool,
        inject_history=inject_history,
        model=model,
        timezone=timezone,
        agent_file=agent_file,
        execution_type=execution_type,
        command=command,
        script_timeout=script_timeout,
        expires_at=expires_at,
        max_runs=max_runs,
        session_id=session_id,
        target_session=target_session,
        originating_session_id=originating_session_id,
        chain_depth=chain_depth,
        delivery_mode=delivery_mode,
        delivery_kind=delivery_kind,
        incident_key=incident_key,
        incident_title=incident_title,
    )
    result = _call(_scheduler.add, entry)
    return _entry_to_dict(result)


@tool(require_daemon=True)
def schedule_list() -> list:
    """List all configured schedules with their status.

    Returns:
        List of schedules with id, type, enabled, next_run, last_status.
        Per-run history is omitted to keep the result small; fetch it for one
        schedule with schedule_status(id).
    """
    entries = _call(_scheduler.list)
    result = []
    for e in entries:
        d = _entry_to_dict(e)
        # Every schedule keeps up to 20 run_history entries; dumping them all can
        # balloon the listing past the exec-output cap. last_run/last_status/
        # run_count already summarize health, so drop the array here.
        d.pop("run_history", None)
        result.append(d)
    return result


@tool(require_daemon=True)
@deny_when_sandboxed
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
@deny_when_sandboxed
def schedule_enable(id: str) -> dict:
    """Enable a disabled schedule.

    Args:
        id: Schedule ID to enable

    Returns:
        Confirmation with updated schedule details
    """
    _call(_scheduler.enable, id)
    return _entry_to_dict(_call(_scheduler.get, id))


@tool(require_daemon=True)
@deny_when_sandboxed
def schedule_disable(id: str) -> dict:
    """Disable a schedule without removing it.

    Args:
        id: Schedule ID to disable

    Returns:
        Confirmation with updated schedule details
    """
    _call(_scheduler.disable, id)
    return _entry_to_dict(_call(_scheduler.get, id))


@tool(require_daemon=True)
@deny_when_sandboxed
def schedule_update(
    id: str,
    prompt: Optional[str] = None,
    cron: Optional[str] = None,
    run_at: Optional[str] = None,
    timezone: Optional[str] = None,
    notify: Optional[list[str]] = None,
    notify_tool: Optional[bool] = None,
    inject_history: Optional[bool] = None,
    model: Optional[str] = None,
    agent_file: Optional[str] = None,
    execution_type: Optional[str] = None,
    command: Optional[str] = None,
    script_timeout: Optional[int] = None,
    expires_at: Optional[str] = None,
    max_runs: Optional[int] = None,
    session_id: Optional[str] = None,
    target_session: Optional[str] = None,
    delivery_mode: Optional[str] = None,
    delivery_kind: Optional[str] = None,
    incident_key: Optional[str] = None,
    incident_title: Optional[str] = None,
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
        inject_history: Enable/disable result injection into user chat sessions (optional)
        model: Model override for this schedule (optional). Set to empty string to clear.
        agent_file: Agent name (e.g., "+reporter") or path to agent .md file (optional). Set to empty string to clear.
        execution_type: Change to "agent", "script" or "session_message" (optional).
        command: Shell command for script execution (optional). Set to empty string to clear.
        script_timeout: Max seconds for script execution (optional).
        expires_at: ISO datetime for auto-disable (optional). Set to empty string to clear.
        max_runs: Auto-disable after N successful runs (optional).
        session_id: Persistent session ID for this schedule (optional). Set to empty string to clear (reverts to per-run sessions).
        target_session: Routing target for the delivered result (optional). See schedule_create for legal forms. Set to empty string to clear (reverts to fallback chain).
        delivery_mode: New delivery mode (optional). See schedule_create for legal values.
        delivery_kind: New delivery kind (optional). See schedule_create for legal values.
        incident_key: New incident dedupe key (optional). Set to empty string to clear.
        incident_title: New incident session title (optional). Set to empty string to clear.

    Returns:
        Updated schedule details
    """
    # Build fields dict from provided params (rename cron → cron_expr)
    simple = {
        "prompt": prompt,
        "cron_expr": cron,
        "run_at": run_at,
        "timezone": timezone,
        "inject_history": inject_history,
        "execution_type": execution_type,
        "script_timeout": script_timeout,
        "max_runs": max_runs,
        "delivery_mode": delivery_mode,
        "delivery_kind": delivery_kind,
    }
    fields = {k: v for k, v in simple.items() if v is not None}

    if notify is not None:
        notify = _validate_notify(notify, False)
        fields["notify"] = notify or []
    if notify_tool is not None:
        if notify_tool:
            effective_notify = notify if notify is not None else _call(_scheduler.get, id).notify
            if not effective_notify:
                raise ValueError("notify_tool=True requires a non-empty 'notify' list")
        fields["notify_tool"] = notify_tool

    # Clearable fields: empty/falsy value → None (clears the field)
    for param_name, value in [
        ("model", model),
        ("agent_file", agent_file),
        ("command", command),
        ("expires_at", expires_at),
        ("session_id", session_id),
        ("target_session", target_session),
        ("incident_key", incident_key),
        ("incident_title", incident_title),
    ]:
        if value is not None:
            fields[param_name] = value or None

    if not fields:
        raise ValueError("No fields to update")

    result = _call(_scheduler.update, id, **fields)
    return _entry_to_dict(result)


@tool(require_daemon=True)
@deny_when_sandboxed
def schedule_cleanup() -> dict:
    """Remove all orphaned one-off schedules (disabled, already fired).

    Returns:
        Dict with removed schedule IDs and count
    """
    removed = _call(_scheduler.cleanup)
    return {"removed": removed, "count": len(removed)}


@tool(require_daemon=True)
@deny_when_sandboxed
def schedule_run(id: str) -> dict:
    """Fire an existing schedule immediately in the background.

    The schedule runs asynchronously — this tool returns immediately.
    Results are delivered via the schedule's configured notification channels.

    Args:
        id: Schedule ID to fire

    Returns:
        Confirmation that the schedule was triggered
    """
    _call(_scheduler.fire_now, id)
    return {"status": "triggered", "id": id}


@tool(require_daemon=True)
def schedule_status(id: str, history_limit: int = 10) -> dict:
    """Get the current status of a schedule, including whether it's running and recent run history.

    Args:
        id: Schedule ID to check
        history_limit: Max number of recent runs to return (default: 10)

    Returns:
        Dict with schedule state including is_running flag and run_history
    """
    entry = _call(_scheduler.get, id)
    running_ids = _call(_scheduler.get_running_ids)
    return {
        "id": entry.id,
        "is_running": entry.id in running_ids,
        "last_status": entry.last_status,
        "last_run": entry.last_run,
        "last_error": entry.last_error,
        "next_run": entry.next_run,
        "enabled": entry.enabled,
        "run_count": entry.run_count,
        "execution_type": entry.execution_type,
        "session_id": entry.session_id,
        "run_history": entry.run_history[-history_limit:],
    }


def _get_running_tasks_snapshot():
    """Collect running task details in a single scheduler-thread call."""
    running_ids = _scheduler.get_running_ids()
    return [{"id": e.id, "prompt": e.prompt[:200]} for rid in running_ids if (e := _scheduler.get(rid))]


@tool(require_daemon=True)
def list_running_tasks() -> list:
    """List all currently running schedule tasks.

    Returns:
        List of dicts with id and prompt (truncated) for each running task
    """
    return _call(_get_running_tasks_snapshot)


@tool(require_daemon=True)
@deny_when_sandboxed
def background_task(
    prompt: str = "",
    notify: Optional[list[str]] = None,
    notify_tool: bool = False,
    inject_history: bool = False,
    model: Optional[str] = None,
    max_turns: Optional[int] = None,
    execution_type: str = "agent",
    command: Optional[str] = None,
    script_timeout: int = 60,
    on_complete: Optional[dict] = None,
    target_session: Optional[str] = None,
) -> dict:
    """Launch a background task that auto-replies with results when complete.

    Creates a one-off schedule and fires it immediately. When the task finishes,
    results are automatically processed on the user's conversation session and
    delivered as a human-friendly response via notification channels.

    Use this for tasks that may take a while (research, long-running commands, etc.)
    so the user doesn't have to wait. For deterministic commands (curl, ping, df, etc.),
    use execution_type="script" to skip the LLM entirely.

    IMPORTANT: Always confirm with the user before launching background tasks.

    Args:
        prompt: Clear, self-contained instruction for the background agent. Optional when execution_type is "script".
        notify: Notification channels for result delivery. Required for auto-reply.
        notify_tool: If true, gives the background agent the notify_user tool.
        inject_history: If true, inject raw result into user session (in addition to auto-reply).
        model: Optional model override (e.g., "openai:gpt-4o-mini").
        max_turns: Optional max reasoning turns for the agent. Limits how many LLM iterations the task can use.
        execution_type: "agent" (default) runs an LLM agent, "script" runs a shell command directly.
        command: Shell command to execute when execution_type is "script".
        script_timeout: Max seconds for script execution (default: 60).
        on_complete: Completion callback. Currently supports {"action": "reply"} to auto-reply
            to the originating session when the task finishes, allowing the agent to chain work.
        target_session: Routing target for the delivered result. See schedule_create for legal forms.
            Defaults to "originating" when on_complete is set so completion replies still land in the
            spawning session; otherwise defaults to None (fallback chain).

    Returns:
        Dict with status and generated task ID
    """
    if execution_type not in ("agent", "script"):
        raise ValueError(
            f"background_task runs 'agent' or 'script', got '{execution_type}'; use schedule_create for session_message"
        )
    if execution_type == "script":
        if not command:
            raise ValueError("'command' is required when execution_type is 'script'")
    elif not prompt:
        raise ValueError("'prompt' is required when execution_type is 'agent'")

    if on_complete and (not isinstance(on_complete, dict) or on_complete.get("action") != "reply"):
        raise ValueError("on_complete must be {'action': 'reply'}")

    notify = _validate_notify(notify, notify_tool)

    from tsugite_daemon.scheduler import ScheduleEntry
    from tsugite_daemon.session_runner import get_current_chain_depth

    originating_session_id = get_current_session_id()
    if on_complete and not originating_session_id:
        raise ValueError("on_complete requires a session context (daemon mode)")
    chain_depth = get_current_chain_depth() if on_complete else 0
    target_session = _resolve_target_session(target_session, originating_session_id)

    if target_session is None and on_complete:
        from tsugite_daemon.scheduler import TARGET_SESSION_ORIGINATING

        target_session = TARGET_SESSION_ORIGINATING

    task_id = f"bg-{uuid4().hex[:8]}"
    # run_at in the past so it's immediately eligible
    run_at = "2000-01-01T00:00:00Z"

    entry = ScheduleEntry(
        id=task_id,
        prompt=prompt,
        schedule_type="once",
        run_at=run_at,
        notify=notify or [],
        notify_tool=notify_tool,
        inject_history=inject_history,
        auto_reply=bool(notify),
        model=model,
        max_turns=max_turns,
        execution_type=execution_type,
        command=command,
        script_timeout=script_timeout,
        originating_session_id=originating_session_id,
        on_complete=on_complete,
        chain_depth=chain_depth,
        target_session=target_session,
    )
    _call(_scheduler.add, entry)
    _call(_scheduler.fire_now, task_id)
    return {"status": "started", "id": task_id}
