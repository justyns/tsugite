"""Validation command for agent frontmatter."""

import sys
from pathlib import Path
from typing import List

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tsugite.agent_runner.validation import validate_agent_file
from tsugite.md_agents import parse_agent_file

console = Console()


def validate_command(
    files: List[Path] = typer.Argument(..., help="Agent file(s) to validate (supports glob patterns)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed validation information"),
):
    """Validate agent frontmatter configuration against the schema.

    Checks that YAML frontmatter in agent markdown files is valid according to
    the AgentConfig schema. This validates field types, required fields, and
    catches typos in field names.

    Examples:
        tsugite validate agents/my_agent.md
        tsugite validate agents/*.md
        tsugite validate .tsugite/default.md agents/*.md
    """
    # Expand glob patterns
    agent_files = []
    for pattern in files:
        if "*" in str(pattern):
            # Use glob matching
            parent = pattern.parent if pattern.parent.exists() else Path.cwd()
            matches = list(parent.glob(pattern.name))
            if matches:
                agent_files.extend(matches)
            else:
                console.print(f"[yellow]Warning: No files match pattern: {pattern}[/yellow]")
        elif pattern.exists():
            agent_files.append(pattern)
        else:
            console.print(f"[red]Error: File not found: {pattern}[/red]")

    if not agent_files:
        console.print("[red]Error: No agent files to validate[/red]")
        sys.exit(1)

    # Validate each file
    results = []
    for agent_file in agent_files:
        try:
            # Try parsing the agent file
            agent = parse_agent_file(agent_file)

            # Use existing validation
            is_valid, error_msg = validate_agent_file(agent_file)

            results.append(
                {
                    "file": agent_file,
                    "valid": is_valid,
                    "error": error_msg if not is_valid else None,
                    "config": agent.config if is_valid else None,
                }
            )

        except ValidationError as e:
            # Pydantic validation error - extract field errors
            errors = []
            for error in e.errors():
                field = ".".join(str(x) for x in error["loc"])
                msg = error["msg"]
                errors.append(f"{field}: {msg}")

            results.append(
                {
                    "file": agent_file,
                    "valid": False,
                    "error": "\n  ".join(errors),
                    "config": None,
                }
            )

        except Exception as e:
            results.append(
                {
                    "file": agent_file,
                    "valid": False,
                    "error": str(e),
                    "config": None,
                }
            )

    # Display results
    valid_count = sum(1 for r in results if r["valid"])
    invalid_count = len(results) - valid_count

    # Create table
    table = Table(title="Agent Validation Results", show_header=True, header_style="bold cyan")
    table.add_column("File", style="dim")
    table.add_column("Status", justify="center")
    table.add_column("Details")

    for result in results:
        file_path = str(result["file"])
        if result["valid"]:
            status = "[green]✓ Valid[/green]"
            details = ""
            if verbose and result["config"]:
                config = result["config"]
                details = f"name={config.name}, model={config.model or 'default'}, tools={len(config.tools)}"
        else:
            status = "[red]✗ Invalid[/red]"
            details = f"[red]{result['error']}[/red]" if result["error"] else "[red]Unknown error[/red]"

        table.add_row(file_path, status, details)

    console.print()
    console.print(table)
    console.print()

    # Summary panel
    summary_style = "green" if invalid_count == 0 else "yellow"
    summary_text = f"[bold]{valid_count}[/bold] valid, [bold]{invalid_count}[/bold] invalid"

    console.print(
        Panel(
            summary_text,
            title="Summary",
            border_style=summary_style,
        )
    )

    # Exit with appropriate status code
    if invalid_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)
