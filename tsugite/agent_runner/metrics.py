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
    """Display step execution metrics in a table using event system."""
    from .helpers import get_ui_handler

    # Build metrics table
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

    # Build summary line
    summary_parts = [f"Total: {total_duration:.1f}s"]
    if successful > 0:
        summary_parts.append(f"[green]Success: {successful}[/green]")
    if skipped > 0:
        summary_parts.append(f"[yellow]Skipped: {skipped}[/yellow]")
    if failed > 0:
        summary_parts.append(f"[red]Failed: {failed}[/red]")

    summary = " | ".join(summary_parts)

    # Emit through event system
    ui_handler = get_ui_handler(custom_logger)
    if ui_handler:
        from tsugite.events import EventBus, InfoEvent

        event_bus = EventBus()
        event_bus.subscribe(ui_handler.handle_event)

        # `capture` borrows the live console for its width, then hands it back;
        # assigning `.file` would leave the UI printing into a dead buffer.
        console = get_display_console(custom_logger)
        with console.capture() as capture:
            console.print()
            console.print(table)
            console.print()
            console.print(summary)
            console.print()

        event_bus.emit(InfoEvent(message=capture.get()))
