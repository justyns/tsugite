"""Usage tracking CLI commands."""

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()

usage_app = typer.Typer(help="View token usage and cost tracking")


def _parse_since(since: Optional[str]) -> Optional[str]:
    """Parse a --since value into an ISO datetime string.

    Accepts relative durations (7d, 30d, 4w, 2m) or ISO dates (2026-01-01).
    """
    if not since:
        return None

    # Relative duration patterns
    match = re.match(r"^(\d+)([dwm])$", since.strip())
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        now = datetime.now(timezone.utc)
        if unit == "d":
            dt = now - timedelta(days=amount)
        elif unit == "w":
            dt = now - timedelta(weeks=amount)
        elif unit == "m":
            dt = now - timedelta(days=amount * 30)
        else:
            dt = now
        return dt.isoformat()

    # Try ISO date parsing
    try:
        dt = datetime.fromisoformat(since)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except ValueError:
        console.print(f"[red]Invalid --since value: {since}[/red]")
        console.print("[dim]Use relative (7d, 4w, 2m) or ISO date (2026-01-01)[/dim]")
        raise typer.Exit(1)


def _format_cost(cost: float) -> str:
    """Format cost for display."""
    if cost == 0:
        return "$0.00"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def _format_tokens(tokens: int) -> str:
    """Format token count for display."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    if tokens >= 1_000:
        return f"{tokens / 1_000:.1f}k"
    return str(tokens)


@usage_app.command("summary")
def usage_summary(
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Filter by agent name"),
    schedule: Optional[str] = typer.Option(None, "--schedule", "-s", help="Filter by schedule ID"),
    since: Optional[str] = typer.Option(None, "--since", help="Time range (e.g. 7d, 30d, 2m, 2026-01-01)"),
    model: Optional[str] = typer.Option(None, "--model", help="Filter by model name"),
):
    """Show usage summary with totals.

    Examples:
        tsugite usage summary
        tsugite usage summary --agent chat --since 7d
        tsugite usage summary --schedule morning-digest --since 30d
    """
    from tsugite.usage import get_usage_store

    store = get_usage_store()
    since_iso = _parse_since(since)
    data = store.summary(agent=agent, schedule_id=schedule, model=model, since=since_iso)

    if data["run_count"] == 0:
        console.print("[yellow]No usage data found[/yellow]")
        if not since:
            console.print("[dim]Usage is recorded automatically when you run agents.[/dim]")
        return

    title_parts = ["Usage Summary"]
    if agent:
        title_parts.append(f"agent={agent}")
    if schedule:
        title_parts.append(f"schedule={schedule}")
    if since:
        title_parts.append(f"since={since}")
    if model:
        title_parts.append(f"model={model}")

    table = Table(title=" | ".join(title_parts))
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total Runs", str(data["run_count"]))
    table.add_row("Total Tokens", _format_tokens(data["total_tokens"]))
    table.add_row("  Input Tokens", _format_tokens(data["input_tokens"]))
    table.add_row("  Output Tokens", _format_tokens(data["output_tokens"]))
    if data["cached_tokens"]:
        table.add_row("  Cached Tokens", _format_tokens(data["cached_tokens"]))
    table.add_row("Total Cost", _format_cost(data["total_cost"]))
    table.add_row("Avg Cost/Run", _format_cost(data["avg_cost_per_run"]))

    console.print(table)


@usage_app.command("history")
def usage_history(
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Filter by agent name"),
    schedule: Optional[str] = typer.Option(None, "--schedule", "-s", help="Filter by schedule ID"),
    since: Optional[str] = typer.Option(None, "--since", help="Time range (e.g. 7d, 30d, 2026-01-01)"),
    model: Optional[str] = typer.Option(None, "--model", help="Filter by model name"),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number of results"),
):
    """Show per-run usage log.

    Examples:
        tsugite usage history
        tsugite usage history --agent chat --limit 10
        tsugite usage history --since 7d
    """
    from tsugite.usage import get_usage_store

    store = get_usage_store()
    since_iso = _parse_since(since)
    records = store.query(agent=agent, schedule_id=schedule, model=model, since=since_iso, limit=limit)

    if not records:
        console.print("[yellow]No usage records found[/yellow]")
        return

    table = Table(title=f"Usage History (showing {len(records)} runs)")
    table.add_column("Timestamp", style="dim", no_wrap=True)
    table.add_column("Agent", style="green")
    table.add_column("Model", style="dim")
    table.add_column("Input", justify="right")
    table.add_column("Output", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Cost", justify="right", style="yellow")
    table.add_column("Duration", justify="right", style="dim")

    for r in records:
        try:
            ts = datetime.fromisoformat(r.timestamp).strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            ts = r.timestamp[:16] if r.timestamp else "unknown"

        duration = f"{r.duration_ms / 1000:.1f}s" if r.duration_ms else "-"

        table.add_row(
            ts,
            r.agent,
            r.model,
            _format_tokens(r.input_tokens),
            _format_tokens(r.output_tokens),
            _format_tokens(r.total_tokens),
            _format_cost(r.cost),
            duration,
        )

    console.print(table)


@usage_app.command("breakdown")
def usage_breakdown(
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Filter by agent name"),
    schedule: Optional[str] = typer.Option(None, "--schedule", "-s", help="Filter by schedule ID"),
    since: Optional[str] = typer.Option(None, "--since", help="Time range (e.g. 7d, 30d, 2026-01-01)"),
    model: Optional[str] = typer.Option(None, "--model", help="Filter by model name"),
    period: str = typer.Option("day", "--period", "-p", help="Group by: day, week, month"),
):
    """Show usage breakdown by time period.

    Examples:
        tsugite usage breakdown --period day --since 7d
        tsugite usage breakdown --period month --agent chat
        tsugite usage breakdown --period week --schedule morning-digest
    """
    if period not in ("day", "week", "month"):
        console.print(f"[red]Invalid period: {period}. Use day, week, or month.[/red]")
        raise typer.Exit(1)

    from tsugite.usage import get_usage_store

    store = get_usage_store()
    since_iso = _parse_since(since)
    rows = store.aggregate(group_by=period, agent=agent, schedule_id=schedule, model=model, since=since_iso)

    if not rows:
        console.print("[yellow]No usage data found[/yellow]")
        return

    title_parts = [f"Usage by {period}"]
    if agent:
        title_parts.append(f"agent={agent}")
    if schedule:
        title_parts.append(f"schedule={schedule}")

    table = Table(title=" | ".join(title_parts))
    table.add_column("Period", style="cyan")
    table.add_column("Runs", justify="right")
    table.add_column("Input", justify="right")
    table.add_column("Output", justify="right")
    table.add_column("Total Tokens", justify="right")
    table.add_column("Cost", justify="right", style="yellow")

    for row in rows:
        table.add_row(
            row["period"],
            str(row["run_count"]),
            _format_tokens(row["input_tokens"]),
            _format_tokens(row["output_tokens"]),
            _format_tokens(row["total_tokens"]),
            _format_cost(row["total_cost"]),
        )

    console.print(table)
