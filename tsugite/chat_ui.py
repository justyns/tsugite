"""CLI chat UI for interactive conversations."""

from pathlib import Path
from typing import List, Optional, Tuple

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table

from tsugite.chat import ChatManager
from tsugite.custom_ui import CustomUIHandler, CustomUILogger, UIEvent
from tsugite.md_agents import parse_agent_file

# Display constants
CODE_LENGTH_THRESHOLD = 80
MESSAGE_TRUNCATE_LENGTH = 60


def _parse_execution_result(result: str) -> Tuple[str, str]:
    """Parse execution result into logs and output.

    Args:
        result: Raw execution result string

    Returns:
        Tuple of (logs, output)
    """
    logs = ""
    output = ""

    if "Execution logs:" in result and "Out:" in result:
        # Split into logs and output
        parts = result.split("Out:")
        logs = parts[0].replace("Execution logs:", "").strip()
        output = parts[1].strip()
    elif "Execution logs:" in result:
        logs = result.replace("Execution logs:", "").strip()
    elif "Out:" in result:
        output = result.split("Out:")[-1].strip()
    else:
        output = result.strip()

    return logs, output


def _display_code(console: Console, code: str) -> None:
    """Display code with appropriate formatting.

    Args:
        console: Rich console
        code: Code to display
    """
    if "\n" in code and len(code) > CODE_LENGTH_THRESHOLD:
        # Multi-line code - show in a panel with syntax highlighting
        syntax = Syntax(code, "python", theme="monokai", line_numbers=False)
        console.print(Panel(syntax, border_style="yellow", padding=(0, 1)))
    else:
        # Single line or short code - show inline
        console.print(f"  [yellow]{code}[/yellow]")


def _display_output(console: Console, logs: str, output: str) -> None:
    """Display execution output in a panel.

    Args:
        console: Rich console
        logs: Execution logs (print statements)
        output: Return value
    """
    if not logs and not (output and output != "None"):
        return

    output_text = []
    if logs:
        output_text.append(logs)
    if output and output != "None":
        if logs:
            output_text.append(f"\n→ {output}")
        else:
            output_text.append(output)

    combined_output = "\n".join(output_text)

    # Syntax highlight the output (works well for Python data structures)
    syntax = Syntax(combined_output, "python", theme="monokai", line_numbers=False, word_wrap=True)
    console.print(Panel(syntax, title="[dim]Output[/dim]", border_style="dim", padding=(0, 1)))


class ChatUIHandler(CustomUIHandler):
    """Prettier UI handler for chat mode with live updates."""

    def __init__(self, console: Console):
        super().__init__(
            console=console,
            show_code=False,
            show_observations=False,
            show_llm_messages=False,
            show_execution_results=False,
            show_execution_logs=False,
            show_panels=False,
        )
        self.live_display: Optional[Live] = None
        self.tool_actions: List[dict] = []
        self.is_thinking = False
        self.current_tool = None

    def handle_event(self, event: UIEvent, data: dict) -> None:
        """Handle UI events for chat mode with prettier output."""
        # Debug: print all events (uncomment to debug)
        # self.console.print(f"[dim]DEBUG: {event.name} - {data}[/dim]")

        if event == UIEvent.TASK_START:
            self.tool_actions = []
            self.is_thinking = True
            self._show_spinner("Thinking...")

        elif event == UIEvent.STEP_START:
            if not self.is_thinking:
                self.is_thinking = True
                self._show_spinner("Processing...")

        elif event == UIEvent.TOOL_CALL:
            content = data.get("content", "")
            if "Calling tool:" in content:
                tool_name = content.split("Calling tool:")[1].split("with")[0].strip()

                # Skip only final_answer from display (it's implicit in the response)
                if tool_name != "final_answer":
                    self.current_tool = {"tool": tool_name, "args": None, "result": None}
                    self._show_spinner(f"Using {tool_name}...")

        elif event == UIEvent.CODE_EXECUTION:
            # Capture code being executed
            code = data.get("code", "")
            if code and "final_answer" not in code.lower():
                # Store code execution (unless it's just final_answer)
                self.current_tool = {"action": "code", "code": code}
                self._show_spinner("Executing code...")

        elif event == UIEvent.EXECUTION_RESULT:
            # Capture execution result
            content = data.get("content", "")
            if self.current_tool and content:
                # Store the execution result
                self.current_tool["result"] = content
                self.tool_actions.append(self.current_tool)
                self.current_tool = None

        elif event == UIEvent.FINAL_ANSWER:
            self._stop_spinner()
            self.is_thinking = False

        elif event == UIEvent.ERROR:
            self._stop_spinner()
            self.is_thinking = False

    def _show_spinner(self, message: str):
        """Show a spinner with message."""
        if self.live_display is None:
            spinner = Spinner("dots", text=f"[dim]{message}[/dim]")
            self.live_display = Live(spinner, console=self.console, refresh_per_second=10)
            self.live_display.start()
        else:
            # Update existing spinner
            self.live_display.update(Spinner("dots", text=f"[dim]{message}[/dim]"))

    def _stop_spinner(self):
        """Stop the spinner."""
        if self.live_display:
            self.live_display.stop()
            self.live_display = None


def run_chat_cli(
    agent_path: Path,
    model_override: Optional[str] = None,
    max_history: int = 50,
):
    """Run interactive chat CLI.

    Args:
        agent_path: Path to agent markdown file
        model_override: Optional model override
        max_history: Maximum conversation history turns
    """
    console = Console()

    # Show welcome
    console.print("\n[bold cyan]╔╦╗╔═╗╦ ╦╔═╗╦╔╦╗╔═╗[/bold cyan]")
    console.print("[bold cyan] ║ ╚═╗║ ║║ ╦║ ║ ║╣ [/bold cyan]")
    console.print("[bold cyan] ╩ ╚═╝╚═╝╚═╝╩ ╩ ╚═╝[/bold cyan]\n")

    console.print("[cyan]Chat mode[/cyan] - Type [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit")

    # Create UI handler
    ui_handler = ChatUIHandler(console)
    logger = CustomUILogger(ui_handler)

    # Create chat manager
    manager = ChatManager(
        agent_path=agent_path,
        model_override=model_override,
        max_history=max_history,
        custom_logger=logger,
    )

    agent = parse_agent_file(agent_path)
    agent_name = agent.config.name or agent_path.stem
    model = model_override or agent.config.model or "default"

    # Show agent info in a panel
    info = f"[bold]{agent_name}[/bold]\n[dim]{model}[/dim]"
    console.print(Panel(info, title="Agent", border_style="cyan", padding=(0, 1)))
    console.print()

    turn_count = 0

    def session_end_msg():
        return f"\n[cyan]✓ Chat session ended. {turn_count} exchanges.[/cyan]\n"

    while True:
        try:
            user_input = Prompt.ask("\n[bold bright_green]●[/bold bright_green] [bold]You[/bold]").strip()

            if not user_input:
                continue

            if user_input.startswith("/"):
                command, *args = user_input[1:].lower().split()

                if command in ("exit", "quit", "q"):
                    console.print(session_end_msg())
                    break

                elif command == "help":
                    show_help(console)
                    continue

                elif command == "clear":
                    manager.clear_history()
                    console.print("\n[green]✓[/green] [dim]History cleared[/dim]\n")
                    turn_count = 0
                    continue

                elif command == "save":
                    if not args:
                        console.print("\n[red]✗[/red] [dim]Usage: /save <filename>[/dim]\n")
                        continue
                    save_path = Path(args[0])
                    manager.save_conversation(save_path)
                    console.print(f"\n[green]✓[/green] [dim]Saved to {save_path}[/dim]\n")
                    continue

                elif command == "load":
                    if not args:
                        console.print("\n[red]✗[/red] [dim]Usage: /load <filename>[/dim]\n")
                        continue
                    load_path = Path(args[0])
                    if not load_path.exists():
                        console.print(f"\n[red]✗[/red] [dim]File not found: {load_path}[/dim]\n")
                        continue
                    manager.load_conversation(load_path)
                    turn_count = len(manager.conversation_history)
                    console.print(f"\n[green]✓[/green] [dim]Loaded {turn_count} turns from {load_path}[/dim]\n")
                    continue

                elif command == "history":
                    show_history(console, manager)
                    continue

                elif command == "stats":
                    show_stats(console, manager)
                    continue

                else:
                    console.print(f"[red]Unknown command: /{command}[/red]")
                    console.print("Type [bold]/help[/bold] for available commands")
                    continue

            # Run agent turn
            response = manager.run_turn(user_input)

            # Ensure spinner is stopped
            ui_handler._stop_spinner()

            # Display response with markdown rendering
            console.print()
            console.print("[bold bright_blue]●[/bold bright_blue] [bold]Agent[/bold]")
            console.print()

            # Render response as markdown for better formatting
            try:
                md = Markdown(response)
                console.print(md)
            except Exception:
                # Fallback to plain text if markdown fails
                console.print(response)

            # Show tool activity if any
            if ui_handler.tool_actions:
                console.print()
                console.print("[dim]Code executed:[/dim]")

                for i, action in enumerate(ui_handler.tool_actions, 1):
                    # Show code
                    code = action.get("code", "")
                    if code:
                        _display_code(console, code)

                    # Show result if available
                    result = action.get("result", "")
                    if result:
                        logs, output = _parse_execution_result(result)
                        _display_output(console, logs, output)

                    if i < len(ui_handler.tool_actions):
                        console.print()

                ui_handler.tool_actions = []

            console.print()

            turn_count += 1

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Use /exit to quit[/yellow]")
            continue
        except EOFError:
            console.print(f"\n{session_end_msg()}")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")
            continue


def show_help(console: Console):
    """Show help message."""
    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
    table.add_column("Command", style="bold")
    table.add_column("Description", style="dim")

    table.add_row("/help", "Show this help message")
    table.add_row("/exit, /quit", "Exit chat session")
    table.add_row("/clear", "Clear conversation history")
    table.add_row("/save <file>", "Save conversation to file")
    table.add_row("/load <file>", "Load conversation from file")
    table.add_row("/history", "Show conversation history")
    table.add_row("/stats", "Show session statistics")

    console.print()
    console.print(Panel(table, title="[bold]Commands[/bold]", border_style="cyan"))
    console.print("\n[dim]Press Ctrl+C to interrupt agent, Ctrl+D to exit[/dim]\n")


def show_history(console: Console, manager: ChatManager):
    """Show conversation history."""
    if not manager.conversation_history:
        console.print("\n[dim]No conversation history yet[/dim]\n")
        return

    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("#", style="dim", width=4)
    table.add_column("User", style="green")
    table.add_column("Agent", style="blue")

    for i, turn in enumerate(manager.conversation_history, 1):
        user_msg = (
            turn.user_message[:MESSAGE_TRUNCATE_LENGTH] + "..."
            if len(turn.user_message) > MESSAGE_TRUNCATE_LENGTH
            else turn.user_message
        )
        agent_msg = (
            turn.agent_response[:MESSAGE_TRUNCATE_LENGTH] + "..."
            if len(turn.agent_response) > MESSAGE_TRUNCATE_LENGTH
            else turn.agent_response
        )
        table.add_row(str(i), user_msg, agent_msg)

    console.print()
    console.print(
        Panel(table, title=f"[bold]History ({len(manager.conversation_history)} turns)[/bold]", border_style="cyan")
    )
    console.print()


def show_stats(console: Console, manager: ChatManager):
    """Show session statistics."""
    stats = manager.get_stats()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold")

    table.add_row("Total Turns", str(stats["total_turns"]))

    # Format token count nicely
    tokens = stats.get("total_tokens")
    if tokens is not None:
        # Format with commas for readability
        token_str = f"{tokens:,}"
    else:
        token_str = "[dim]N/A[/dim]"
    table.add_row("Total Tokens", token_str)

    # Format duration
    duration = stats["session_duration"]
    if duration >= 60:
        mins = int(duration // 60)
        secs = int(duration % 60)
        duration_str = f"{mins}m {secs}s"
    else:
        duration_str = f"{duration:.0f}s"
    table.add_row("Duration", duration_str)

    table.add_row("Agent", str(stats["agent"]))
    table.add_row("Model", str(stats["model"]))

    console.print()
    console.print(Panel(table, title="[bold]Session Statistics[/bold]", border_style="cyan"))
    console.print()
