"""History management CLI commands."""

import json
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    pass

console = Console()

history_app = typer.Typer(help="Manage conversation history")


@history_app.command("list")
def history_list(
    machine: str = typer.Option(None, "--machine", help="Filter by machine name"),
    agent: str = typer.Option(None, "--agent", help="Filter by agent name"),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number of results"),
):
    """List conversations with optional filters.

    Examples:
        tsugite history list
        tsugite history list --machine laptop
        tsugite history list --agent chat_assistant --limit 10
    """
    from tsugite.history import SessionStorage, get_history_dir, list_session_files

    try:
        session_files = list_session_files()

        if not session_files:
            console.print("[yellow]No conversations found[/yellow]")

            history_dir = get_history_dir()
            if not history_dir.exists():
                console.print(f"\nHistory directory doesn't exist yet: {history_dir}")
                console.print("Conversations will be saved automatically when you use chat mode.")
            return

        table = Table(title=f"Conversation History ({len(session_files)} found)")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Agent", style="green")
        table.add_column("Machine", style="dim")
        table.add_column("Turns", justify="right")
        table.add_column("Tokens", justify="right", style="dim")
        table.add_column("Updated", style="dim")

        count = 0
        for session_file in session_files:
            if count >= limit:
                break

            try:
                storage = SessionStorage.load(session_file)

                # Apply filters
                if machine and storage.machine != machine:
                    continue
                if agent and storage.agent != agent:
                    continue

                # Format date
                try:
                    updated_str = storage.created_at.strftime("%Y-%m-%d %H:%M") if storage.created_at else "unknown"
                except (ValueError, TypeError):
                    updated_str = "unknown"

                table.add_row(
                    storage.session_id,
                    storage.agent or "unknown",
                    storage.machine or "unknown",
                    str(storage.turn_count),
                    f"{storage.total_tokens:,}" if storage.total_tokens > 0 else "-",
                    updated_str,
                )
                count += 1

            except Exception:
                continue

        console.print(table)

        console.print("\n[dim]Use 'tsugite history show CONVERSATION_ID' to view details[/dim]")

    except Exception as e:
        console.print(f"[red]Failed to list conversations: {e}[/red]")
        raise typer.Exit(1)


@history_app.command("show")
def history_show(
    conversation_id: str = typer.Argument(help="Conversation ID to show"),
    format: str = typer.Option("plain", "--format", "-f", help="Output format (plain, json, markdown)"),
):
    """Show full conversation details.

    Examples:
        tsugite history show 20251024_103000_chat_abc123
        tsugite history show 20251024_103000_chat_abc123 --format json
        tsugite history show 20251024_103000_chat_abc123 --format markdown
    """
    from tsugite.history import SessionStorage, Turn, get_history_dir

    try:
        session_path = get_history_dir() / f"{conversation_id}.jsonl"
        storage = SessionStorage.load(session_path)
        records = storage.load_records()

        if not records:
            console.print(f"[yellow]Conversation '{conversation_id}' is empty[/yellow]")
            return

        if format == "json":
            records_as_dicts = [r.model_dump(mode="json") for r in records]
            output = json.dumps(records_as_dicts, indent=2, ensure_ascii=False)
            print(output)

        elif format == "markdown":
            console.print(f"# Conversation: {conversation_id}\n")
            if storage.agent:
                console.print(f"- **Agent**: {storage.agent}")
            if storage.model:
                console.print(f"- **Model**: {storage.model}")
            if storage.machine:
                console.print(f"- **Machine**: {storage.machine}")
            if storage.created_at:
                console.print(f"- **Created**: {storage.created_at}\n")
            console.print("---\n")

            for record in records:
                if isinstance(record, Turn):
                    console.print(f"## Turn ({record.timestamp})\n")

                    # Show user summary or first message
                    if record.user_summary:
                        console.print(f"**User**: {record.user_summary}\n")

                    if record.final_answer:
                        console.print(f"**Assistant**: {record.final_answer}\n")

                    if record.functions_called:
                        console.print(f"*Functions*: {', '.join(record.functions_called)}\n")

                    console.print(f"*Tokens*: {record.tokens or 0} | *Cost*: ${record.cost or 0.0:.4f}\n")
                    console.print("---\n")

        else:
            # Plain text output
            console.print("=" * 60)
            console.print(f"Conversation: {conversation_id}")
            console.print(f"Agent: {storage.agent or 'unknown'}")
            console.print(f"Model: {storage.model or 'unknown'}")
            console.print(f"Machine: {storage.machine or 'unknown'}")
            console.print(f"Created: {storage.created_at}")
            console.print("=" * 60)
            console.print()

            for record in records:
                if isinstance(record, Turn):
                    console.print(f"[{record.timestamp}]")
                    if record.user_summary:
                        console.print(f"User: {record.user_summary}")
                    if record.final_answer:
                        console.print(f"Assistant: {record.final_answer}")
                    if record.functions_called:
                        console.print(f"  Functions: {', '.join(record.functions_called)}")
                    console.print(f"  Tokens: {record.tokens or 0} | Cost: ${record.cost or 0.0:.4f}")
                    console.print()
                    console.print("-" * 60)
                    console.print()

    except FileNotFoundError:
        console.print(f"[red]Conversation '{conversation_id}' not found[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to load conversation: {e}[/red]")
        raise typer.Exit(1)


@history_app.command("rebuild-index")
def history_rebuild_index():
    """Rebuild conversation index from JSONL files.

    Note: V2 storage format no longer uses a separate index file.
    This command scans all session files to verify they are valid.
    """
    from tsugite.history import SessionStorage, list_session_files

    try:
        console.print("Scanning session files...")
        session_files = list_session_files()
        valid_count = 0
        for session_file in session_files:
            try:
                SessionStorage.load(session_file)
                valid_count += 1
            except Exception as e:
                console.print(f"[yellow]Skipping invalid session {session_file.stem}: {e}[/yellow]")

        console.print(f"[green]✓ Found {valid_count} valid sessions[/green]")

    except Exception as e:
        console.print(f"[red]Failed to scan sessions: {e}[/red]")
        raise typer.Exit(1)
