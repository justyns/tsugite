"""Slash command handlers for REPL mode."""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from rich.console import Console
from rich.table import Table

from tsugite.history import SessionStorage, list_session_files

if TYPE_CHECKING:
    from tsugite.ui.chat import ChatManager


def handle_help(console: Console) -> None:
    """Show help with available commands.

    Args:
        console: Rich console for output
    """
    table = Table(title="Available Commands", show_header=True, title_style="bold cyan")
    table.add_column("Command", style="cyan", width=22)
    table.add_column("Description", style="white")

    commands = [
        ("/help", "Show this help message"),
        ("/exit, /quit", "Exit the REPL"),
        ("/clear", "Clear the screen"),
        ("/model <name>", "Switch to a different model"),
        ("/agent <name>", "Switch to a different agent"),
        ("/attach <path>", "Attach a file for context"),
        ("/detach <path>", "Remove an attachment"),
        ("/list-attachments", "Show current attachments"),
        ("/continue [id]", "Resume a previous conversation"),
        ("/history [limit]", "Show recent conversations (default: 10)"),
        ("/save <path>", "Export conversation to file"),
        ("/stats", "Show token and cost statistics"),
        ("/tools", "Show available tools"),
        ("/stream on|off", "Toggle streaming mode"),
        ("/verbose on|off", "Toggle verbose mode (show raw tool output)"),
    ]

    for cmd, desc in commands:
        table.add_row(cmd, desc)

    console.print(table)
    console.print()
    console.print("[dim]Type your message to chat with the agent.[/dim]")


def handle_clear(console: Console) -> None:
    """Clear the screen.

    Args:
        console: Rich console for output
    """
    console.clear()


def handle_stats(console: Console, manager: "ChatManager") -> None:
    """Show session statistics.

    Args:
        console: Rich console for output
        manager: Chat manager with session data
    """
    table = Table(title="Session Statistics", title_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Turns", str(manager.turn_count))
    table.add_row("Total Tokens", f"{manager.total_tokens:,}")
    table.add_row("Total Cost", f"${manager.total_cost:.6f}")

    if hasattr(manager, "cached_tokens") and manager.cached_tokens:
        table.add_row("Cached Tokens", f"{manager.cached_tokens:,}")

    console.print(table)


def handle_history(console: Console, limit: int = 10) -> None:
    """Show conversation history.

    Args:
        console: Rich console for output
        limit: Maximum number of conversations to show
    """
    session_files = list_session_files()

    if not session_files:
        console.print("[yellow]No conversation history found.[/yellow]")
        return

    table = Table(title="Recent Conversations", title_style="bold cyan")
    table.add_column("ID", style="cyan", width=12)
    table.add_column("Agent", style="yellow")
    table.add_column("Model", style="green")
    table.add_column("Date", style="blue")
    table.add_column("Turns", justify="right")
    table.add_column("Tokens", justify="right")

    count = 0
    for session_file in session_files:
        if count >= limit:
            break

        try:
            storage = SessionStorage.load(session_file)

            # Format date
            if storage.created_at:
                try:
                    created_at = storage.created_at.strftime("%Y-%m-%d %H:%M")
                except (ValueError, AttributeError):
                    created_at = "unknown"
            else:
                created_at = "unknown"

            # Truncate ID for display
            short_id = storage.session_id[:12]

            # Format tokens with commas
            tokens_str = f"{storage.total_tokens:,}" if storage.total_tokens else "0"

            table.add_row(
                short_id,
                storage.agent or "unknown",
                storage.model or "unknown",
                created_at,
                str(storage.turn_count),
                tokens_str,
            )
            count += 1

        except Exception:
            continue

    console.print(table)
    console.print()
    console.print("[dim]Use `/continue <id>` to resume a conversation[/dim]")


def handle_attach(console: Console, path: str, manager: "ChatManager") -> None:
    """Attach a file for context.

    Args:
        console: Rich console for output
        path: Path to file to attach
        manager: Chat manager
    """
    # Expand user home directory
    expanded_path = os.path.expanduser(path)
    file_path = Path(expanded_path)

    if not file_path.exists():
        console.print(f"[red]✗ File not found: {path}[/red]")
        return

    if not file_path.is_file():
        console.print(f"[red]✗ Not a file: {path}[/red]")
        return

    try:
        # Read file content
        content = file_path.read_text()

        # Add to manager's attachments
        if not hasattr(manager, "attachments"):
            manager.attachments = {}

        manager.attachments[str(file_path)] = content
        console.print(f"[green]✓ Attached: {file_path}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to attach file: {e}[/red]")


def handle_detach(console: Console, path: str, manager: "ChatManager") -> None:
    """Remove an attachment.

    Args:
        console: Rich console for output
        path: Path to file to detach
        manager: Chat manager
    """
    if not hasattr(manager, "attachments") or not manager.attachments:
        console.print("[yellow]No attachments to remove.[/yellow]")
        return

    expanded_path = os.path.expanduser(path)
    file_path = str(Path(expanded_path))

    if file_path in manager.attachments:
        del manager.attachments[file_path]
        console.print(f"[green]✓ Detached: {path}[/green]")
    else:
        console.print(f"[yellow]File not attached: {path}[/yellow]")


def handle_list_attachments(console: Console, manager: "ChatManager") -> None:
    """List current attachments.

    Args:
        console: Rich console for output
        manager: Chat manager
    """
    if not hasattr(manager, "attachments") or not manager.attachments:
        console.print("[yellow]No attachments.[/yellow]")
        return

    table = Table(title="Attachments", title_style="bold cyan")
    table.add_column("File", style="cyan")
    table.add_column("Size", style="green", justify="right")

    for path, content in manager.attachments.items():
        size = len(content)
        if size < 1024:
            size_str = f"{size}B"
        elif size < 1024 * 1024:
            size_str = f"{size / 1024:.1f}KB"
        else:
            size_str = f"{size / (1024 * 1024):.1f}MB"

        table.add_row(path, size_str)

    console.print(table)


def handle_save(console: Console, path: str, manager: "ChatManager") -> None:
    """Export conversation to file.

    Args:
        console: Rich console for output
        path: Path to save conversation
        manager: Chat manager
    """
    if not manager.conversation_id:
        console.print("[yellow]No active conversation to save.[/yellow]")
        return

    try:
        from tsugite.history import Turn, get_history_dir

        # Load conversation turns
        session_path = get_history_dir() / f"{manager.conversation_id}.jsonl"
        storage = SessionStorage.load(session_path)
        records = storage.load_records()

        # Format as markdown
        output = f"# Conversation: {manager.conversation_id}\n\n"
        output += f"**Agent:** {storage.agent or 'unknown'}\n"
        output += f"**Model:** {storage.model or 'unknown'}\n\n"
        output += "---\n\n"

        for record in records:
            if isinstance(record, Turn):
                output += f"## User\n\n{record.user_summary or ''}\n\n"
                output += f"## Assistant\n\n{record.final_answer or ''}\n\n"
                output += "---\n\n"

        # Write to file
        expanded_path = os.path.expanduser(path)
        file_path = Path(expanded_path)
        file_path.write_text(output)

        console.print(f"[green]✓ Conversation saved to: {file_path}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to save conversation: {e}[/red]")


def handle_tools(console: Console, manager: "ChatManager") -> None:
    """Show available tools.

    Args:
        console: Rich console for output
        manager: Chat manager
    """
    if not hasattr(manager, "available_tools") or not manager.available_tools:
        console.print("[yellow]No tools available for this agent.[/yellow]")
        return

    table = Table(title="Available Tools", title_style="bold cyan")
    table.add_column("Tool", style="cyan")
    table.add_column("Description", style="white")

    for tool in manager.available_tools:
        name = getattr(tool, "name", str(tool))
        description = getattr(tool, "description", "")
        table.add_row(name, description)

    console.print(table)


def handle_stream(console: Console, value: Optional[str], manager: "ChatManager") -> None:
    """Toggle streaming mode.

    Args:
        console: Rich console for output
        value: "on" or "off" (if None, toggle)
        manager: Chat manager
    """
    if value is None:
        # Toggle
        manager.stream_enabled = not getattr(manager, "stream_enabled", False)
    elif value.lower() == "on":
        manager.stream_enabled = True
    elif value.lower() == "off":
        manager.stream_enabled = False
    else:
        console.print("[red]Invalid value. Use 'on' or 'off'.[/red]")
        return

    status = "enabled" if manager.stream_enabled else "disabled"
    console.print(f"[green]Streaming {status}[/green]")


def handle_verbose(console: Console, value: Optional[str], ui_handler) -> None:
    """Toggle verbose mode (show raw tool output).

    Args:
        console: Rich console for output
        value: "on" or "off" (if None, toggle)
        ui_handler: REPL UI handler to update
    """
    if value is None:
        # Toggle
        ui_handler.show_observations = not ui_handler.show_observations
    elif value.lower() == "on":
        ui_handler.show_observations = True
    elif value.lower() == "off":
        ui_handler.show_observations = False
    else:
        console.print("[red]Invalid value. Use 'on' or 'off'.[/red]")
        return

    status = "enabled" if ui_handler.show_observations else "disabled"
    console.print(f"[green]Verbose mode {status}[/green]")
    if ui_handler.show_observations:
        console.print("[dim]Now showing raw tool output[/dim]")
    else:
        console.print("[dim]Hiding raw tool output (cleaner view)[/dim]")


def parse_command(user_input: str) -> tuple[str, list[str], Optional[str]]:
    """Parse slash command and arguments with validation.

    Args:
        user_input: User input string

    Returns:
        Tuple of (command, args_list, error_message)
        error_message is None if command is valid
    """
    # Valid commands
    valid_commands = {
        "/help",
        "/exit",
        "/quit",
        "/clear",
        "/model",
        "/agent",
        "/attach",
        "/detach",
        "/list-attachments",
        "/continue",
        "/history",
        "/save",
        "/stats",
        "/tools",
        "/stream",
        "/verbose",
    }

    parts = user_input.split(maxsplit=1)
    command = parts[0]
    args = parts[1].split() if len(parts) > 1 else []

    # Validate command
    if command not in valid_commands:
        # Try to suggest similar command (simple string matching)
        suggestions = [cmd for cmd in valid_commands if cmd.startswith(command[:3])]
        if suggestions:
            error = f"Unknown command: {command}\n  Did you mean: {', '.join(suggestions)}?"
        else:
            error = f"Unknown command: {command}\n  Type /help for available commands."
        return command, args, error

    return command, args, None
