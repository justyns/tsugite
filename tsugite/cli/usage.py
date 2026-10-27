"""CLI commands for usage tracking and cost analytics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
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


def _fmt_rate(r: float | None) -> str:
    return "-" if r is None else f"{r:.1%}"


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


def _render_top_table(label: str, rows: list[dict], key: str, extra: Sequence[tuple[str, str, Callable]] = ()) -> None:
    if not rows:
        console.print("[dim]No usage data found.[/dim]")
        return
    table = Table(show_header=True, header_style="bold cyan", padding=(0, 1))
    table.add_column(label)
    table.add_column("Runs", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right")
    for header, _, _ in extra:
        table.add_column(header, justify="right")
    for row in rows:
        table.add_row(
            row[key] or "(unknown)",
            str(row["runs"]),
            _fmt_tokens(row["total_tokens"]),
            _fmt_cost(row["total_cost"]),
            *(fmt(row[row_key]) for _, row_key, fmt in extra),
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
    """Show top models by cost, with their cache read/write breakdown."""
    from tsugite.usage import get_usage_store

    rows = get_usage_store().top_models(since=_since_arg(since), limit=limit)
    extra = [
        ("Cached Read", "cached_read_tokens", _fmt_tokens),
        ("Cache Write", "cache_write_tokens", _fmt_tokens),
        ("Uncached In", "uncached_input_tokens", _fmt_tokens),
        ("Hit Rate", "cache_hit_rate", _fmt_rate),
    ]
    _render_top_table("Model", rows, "model", extra)


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


def _fmt_reset(resets_at: str | None) -> str:
    # resets_at is UTC by contract; sliced rather than parsed, so say so.
    return f"{resets_at[:16].replace('T', ' ')} UTC" if resets_at else "-"


@usage_app.command("providers")
def providers() -> None:
    """Show how much of each provider's subscription or budget is used."""
    from tsugite.usage import collect_quota_reports

    rows = collect_quota_reports()
    if not rows:
        console.print("[dim]No provider quota sources registered.[/dim]")
        return

    table = Table(show_header=True, header_style="bold cyan", padding=(0, 1))
    table.add_column("Provider")
    table.add_column("Window")
    table.add_column("Used", justify="right")
    table.add_column("Resets")

    for row in rows:
        if row["error"]:
            table.add_row(row["label"], f"[red]{row['error']}[/red]", "-", "-")
            continue
        for i, window in enumerate(row["windows"]):
            table.add_row(
                row["label"] if i == 0 else "",
                window["label"],
                f"{window['used_pct']:g}%",
                _fmt_reset(window["resets_at"]),
            )

    console.print(table)


@usage_app.command("anomalies")
def anomalies(
    day: Optional[str] = typer.Option(None, "--day", help="Day to check, ISO date (default: today, UTC)"),
    window: int = typer.Option(14, "--window", min=1, help="Trailing window length in days"),
    multiple: float = typer.Option(
        3.0, "--multiple", min=1.0, help="Flag a day at or above this multiple of the trailing median"
    ),
    ceiling: Optional[float] = typer.Option(
        None, "--ceiling", min=0.0, help="Also flag a model whose day total is above this USD amount"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON instead of a table"),
) -> None:
    """Compare each model's day spend to its trailing median and report outliers.

    A model with fewer than 3 trailing days with a reported cost, or a
    trailing median of 0, has no usable ratio, so only --ceiling can flag it.
    When --day is today, its total is a partial day compared against
    full-day medians. Prints nothing on a clean day and exits 0 either way.
    """
    from tsugite.usage import detect_spend_anomalies, get_usage_store

    if day:
        try:
            target_day = date.fromisoformat(day)
        except ValueError:
            console.print(f"[red]Invalid --day value: {day}[/red]")
            raise typer.Exit(1)
    else:
        target_day = datetime.now(timezone.utc).date()
    results = detect_spend_anomalies(
        get_usage_store(), target_day, window_days=window, multiple=multiple, ceiling=ceiling
    )

    if json_output:
        console.print_json(data=[asdict(a) for a in results])
    elif results:
        table = Table(show_header=True, header_style="bold cyan", padding=(0, 1))
        table.add_column("Model")
        table.add_column("Day Total", justify="right")
        table.add_column("Trailing Median", justify="right")
        table.add_column("Ratio", justify="right")
        table.add_column("Rule")
        for a in results:
            table.add_row(
                a.model,
                _fmt_cost(a.day_total),
                _fmt_cost(a.trailing_median),
                f"{a.ratio:.1f}x" if a.ratio is not None else "-",
                a.rule,
            )
        console.print(table)
