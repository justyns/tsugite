"""CLI commands for usage tracking and cost analytics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

usage_app = typer.Typer(help="View token usage and cost analytics.")
console = Console()


def _fmt_tokens(n: int | None) -> str:
    if n is None:
        return "-"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_cost(c: float | None) -> str:
    if c is None:
        return "-"
    if c < 0.01:
        return f"${c:.4f}"
    return f"${c:.2f}"


def _since_arg(since: str | None, default_days: int = 30) -> str:
    if since:
        return since
    dt = datetime.now(timezone.utc) - timedelta(days=default_days)
    return dt.isoformat()


@usage_app.command("summary")
def summary(
    period: str = typer.Option("day", "--period", "-p", help="Aggregation period: day, week, month"),
    since: Optional[str] = typer.Option(None, "--since", help="Start date (ISO format, default: 30 days ago)"),
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Filter to specific agent"),
) -> None:
    """Show usage summary by time period."""
    from tsugite.usage import get_usage_store

    store = get_usage_store()
    rows = store.summary(agent=agent, period=period, since=_since_arg(since))

    if not rows:
        console.print("[dim]No usage data found.[/dim]")
        return

    table = Table(show_header=True, header_style="bold cyan", padding=(0, 1))
    table.add_column("Period")
    table.add_column("Runs", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right")

    for row in rows:
        table.add_row(
            row["period"],
            str(row["runs"]),
            _fmt_tokens(row["total_tokens"]),
            _fmt_cost(row["total_cost"]),
        )

    console.print(table)

    total_runs = sum(r["runs"] for r in rows)
    total_tokens = sum(r["total_tokens"] or 0 for r in rows)
    total_cost = sum(r["total_cost"] or 0 for r in rows)
    console.print(f"\n[dim]Total: {total_runs} runs, {_fmt_tokens(total_tokens)} tokens, {_fmt_cost(total_cost)}[/dim]")


def _render_top_table(label: str, rows: list[dict], key: str) -> None:
    if not rows:
        console.print("[dim]No usage data found.[/dim]")
        return
    table = Table(show_header=True, header_style="bold cyan", padding=(0, 1))
    table.add_column(label)
    table.add_column("Runs", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right")
    for row in rows:
        table.add_row(
            row[key] or "(unknown)",
            str(row["runs"]),
            _fmt_tokens(row["total_tokens"]),
            _fmt_cost(row["total_cost"]),
        )
    console.print(table)


@usage_app.command("agents")
def agents(
    since: Optional[str] = typer.Option(None, "--since", help="Start date (ISO format, default: 30 days ago)"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of agents to show"),
) -> None:
    """Show top agents by cost."""
    from tsugite.usage import get_usage_store

    rows = get_usage_store().top_agents(since=_since_arg(since), limit=limit)
    _render_top_table("Agent", rows, "agent")


@usage_app.command("models")
def models(
    since: Optional[str] = typer.Option(None, "--since", help="Start date (ISO format, default: 30 days ago)"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of models to show"),
) -> None:
    """Show top models by cost."""
    from tsugite.usage import get_usage_store

    rows = get_usage_store().top_models(since=_since_arg(since), limit=limit)
    _render_top_table("Model", rows, "model")


@usage_app.command("total")
def total(
    since: Optional[str] = typer.Option(None, "--since", help="Start date (ISO format)"),
) -> None:
    """Show grand total usage."""
    from tsugite.usage import get_usage_store

    store = get_usage_store()
    t = store.total(since=since)

    console.print(f"[bold]Runs:[/bold]   {t['runs']}")
    console.print(f"[bold]Tokens:[/bold] {_fmt_tokens(t['total_tokens'])}")
    console.print(f"[bold]Cost:[/bold]   {_fmt_cost(t['total_cost'])}")
