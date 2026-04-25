"""Skill management CLI commands."""

import sys
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tsugite.skill_discovery import scan_skills_with_issues

console = Console()

skills_app = typer.Typer(help="Inspect and validate skills")


@skills_app.command("check")
def skill_check(
    paths: Optional[List[str]] = typer.Option(
        None,
        "--path",
        "-p",
        help="Extra skill directory to scan (repeatable). Defaults to discovery roots.",
    ),
):
    """Scan skill directories and report any validation issues.

    Walks the standard skill discovery roots (and any --path entries) and
    surfaces problems that the daemon would otherwise log silently:
    invalid YAML, missing/duplicate names, malformed triggers, bad TTL, etc.

    Exits 1 if any error-severity issues are found, 0 otherwise.
    """
    skills, issues = scan_skills_with_issues(extra_paths=list(paths) if paths else None)

    if not issues:
        console.print()
        console.print(
            Panel(
                f"[bold]{len(skills)}[/bold] skills discovered, no issues",
                title="Skill Check",
                border_style="green",
            )
        )
        sys.exit(0)

    table = Table(title="Skill Validation Issues", show_header=True, header_style="bold cyan")
    table.add_column("Severity", justify="center")
    table.add_column("Name", style="bold")
    table.add_column("Path", style="dim")
    table.add_column("Message")

    error_count = 0
    warning_count = 0
    for issue in issues:
        if issue.severity == "error":
            severity_cell = "[red]error[/red]"
            error_count += 1
        else:
            severity_cell = "[yellow]warning[/yellow]"
            warning_count += 1
        table.add_row(severity_cell, issue.name or "-", str(issue.path), issue.message)

    console.print()
    console.print(table)
    console.print()

    summary_style = "red" if error_count > 0 else "yellow"
    summary_text = (
        f"[bold]{len(skills)}[/bold] skills loaded · "
        f"[bold]{error_count}[/bold] errors · "
        f"[bold]{warning_count}[/bold] warnings"
    )
    console.print(Panel(summary_text, title="Summary", border_style=summary_style))

    sys.exit(1 if error_count > 0 else 0)
