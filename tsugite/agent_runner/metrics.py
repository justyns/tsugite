"""Metrics tracking and display for multi-step agent execution."""

from dataclasses import dataclass
from typing import Any, List, Optional

from rich.table import Table

from .helpers import get_display_console


@dataclass
class StepMetrics:
    """Metrics for a single step execution."""

    step_name: str
    step_number: int
    duration: float  # seconds
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    status: str = "success"  # success, failed, skipped
    error: Optional[str] = None


def display_step_metrics(metrics: List[StepMetrics], custom_logger: Optional[Any] = None):
    """Display step execution metrics in a table."""
    console = get_display_console(custom_logger)

    table = Table(title="Multi-Step Execution Metrics", show_header=True)
    table.add_column("Step", style="cyan")
    table.add_column("Duration", justify="right", style="yellow")
    table.add_column("Status", justify="center")

    total_duration = 0
    successful = 0
    failed = 0
    skipped = 0

    for m in metrics:
        status_color = {
            "success": "green",
            "failed": "red",
            "skipped": "yellow",
        }.get(m.status, "white")

        status_symbol = {
            "success": "✓",
            "failed": "✗",
            "skipped": "⚠",
        }.get(m.status, "?")

        table.add_row(
            f"{m.step_number}. {m.step_name}",
            f"{m.duration:.1f}s",
            f"[{status_color}]{status_symbol} {m.status}[/{status_color}]",
        )

        total_duration += m.duration
        if m.status == "success":
            successful += 1
        elif m.status == "failed":
            failed += 1
        elif m.status == "skipped":
            skipped += 1

    console.print()
    console.print(table)

    # Summary line
    summary_parts = []
    summary_parts.append(f"Total: {total_duration:.1f}s")
    if successful > 0:
        summary_parts.append(f"[green]Success: {successful}[/green]")
    if skipped > 0:
        summary_parts.append(f"[yellow]Skipped: {skipped}[/yellow]")
    if failed > 0:
        summary_parts.append(f"[red]Failed: {failed}[/red]")

    console.print(" | ".join(summary_parts))
    console.print()
