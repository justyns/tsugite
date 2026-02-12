"""Schedule tools for agents to manage daemon schedules via HTTP API."""

from typing import Optional

from . import tool

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8321


def _schedule_request(method: str, path: str, **kwargs):
    """Make a request to the daemon schedule API."""
    import httpx

    url = f"http://{_DEFAULT_HOST}:{_DEFAULT_PORT}{path}"
    try:
        resp = httpx.request(method, url, timeout=10, **kwargs)
    except httpx.ConnectError:
        raise RuntimeError("Daemon not running — schedule tools require the daemon")

    if resp.status_code >= 400:
        try:
            msg = resp.json().get("error", resp.text)
        except Exception:
            msg = resp.text
        raise ValueError(f"Schedule API error ({resp.status_code}): {msg}")

    return resp.json()


@tool
def schedule_create(
    id: str,
    agent: str,
    prompt: str,
    cron: Optional[str] = None,
    run_at: Optional[str] = None,
    timezone: str = "UTC",
) -> dict:
    """Create a recurring (cron) or one-off schedule to run an agent.

    Args:
        id: Unique schedule name (e.g., "daily-backup")
        agent: Agent name configured in daemon
        prompt: Prompt to send the agent
        cron: Cron expression for recurring (e.g., "0 9 * * *" = daily at 9am). Mutually exclusive with run_at.
        run_at: ISO datetime for one-off execution (e.g., "2026-02-13T14:00:00-06:00"). Mutually exclusive with cron.
        timezone: IANA timezone (default: UTC)

    Returns:
        Created schedule details including computed next_run
    """
    if not cron and not run_at:
        raise ValueError("Provide either 'cron' or 'run_at'")
    if cron and run_at:
        raise ValueError("Provide 'cron' or 'run_at', not both")

    body = {
        "id": id,
        "agent": agent,
        "prompt": prompt,
        "schedule_type": "once" if run_at else "cron",
        "timezone": timezone,
    }
    if cron:
        body["cron_expr"] = cron
    if run_at:
        body["run_at"] = run_at

    return _schedule_request("POST", "/api/schedules", json=body)


@tool
def schedule_list() -> list:
    """List all configured schedules with their status.

    Returns:
        List of schedules with id, agent, type, enabled, next_run, last_status
    """
    data = _schedule_request("GET", "/api/schedules")
    return data.get("schedules", [])


@tool
def schedule_remove(id: str) -> dict:
    """Remove a schedule.

    Args:
        id: Schedule ID to remove

    Returns:
        Confirmation of removal
    """
    return _schedule_request("DELETE", f"/api/schedules/{id}")


@tool
def schedule_enable(id: str) -> dict:
    """Enable a disabled schedule.

    Args:
        id: Schedule ID to enable

    Returns:
        Confirmation
    """
    return _schedule_request("POST", f"/api/schedules/{id}/enable")


@tool
def schedule_disable(id: str) -> dict:
    """Disable a schedule without removing it.

    Args:
        id: Schedule ID to disable

    Returns:
        Confirmation
    """
    return _schedule_request("POST", f"/api/schedules/{id}/disable")
