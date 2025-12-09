"""REPL-style chat interface using prompt_toolkit for input and rich for output."""

from pathlib import Path
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from tsugite.config import get_xdg_data_path
from tsugite.md_agents import parse_agent_file
from tsugite.ui import CustomUILogger
from tsugite.ui.chat import ChatManager
from tsugite.ui.repl_commands import (
    handle_attach,
    handle_clear,
    handle_detach,
    handle_help,
    handle_history,
    handle_list_attachments,
    handle_multiline,
    handle_save,
    handle_stats,
    handle_stream,
    handle_tools,
    handle_verbose,
    parse_command,
)
from tsugite.ui.repl_completer import TsugiteCompleter
from tsugite.ui.repl_handler import ReplUIHandler


def get_repl_history_path() -> Path:
    """Get path to REPL command history file.

    Returns:
        Path to history file
    """
    data_dir = get_xdg_data_path()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "repl_history"


def show_welcome_banner(console: Console, agent_name: str, model: str, conversation_id: Optional[str] = None) -> None:
    """Show welcome banner with agent info.

    Args:
        console: Rich console for output
        agent_name: Name of the agent
        model: Model identifier
        conversation_id: Optional conversation ID if resuming
    """
    resume_text = ""
    if conversation_id:
        resume_text = f"\n**Conversation:** {conversation_id[:12]}... (resuming)"

    welcome = f"""# Tsugite REPL Chat

**Agent:** {agent_name}
**Model:** {model}{resume_text}

Type your message or `/help` for commands.
"""

    console.print(
        Panel(
            Markdown(welcome),
            title="🤖 Welcome",
            border_style="cyan",
        )
    )
    console.print()


def run_repl_chat(
    agent_path: Path,
    exec_options: "ExecutionOptions",
    history_options: "HistoryOptions",
    resume_turns: Optional[list] = None,
) -> None:
    """Run REPL-style chat session."""

    # Initialize console
    console = Console()

    # Parse agent info
    agent = parse_agent_file(agent_path)
    agent_name = agent.config.name or agent_path.stem
    model = exec_options.model_override or agent.config.model or "default"

    # Show welcome banner
    show_welcome_banner(console, agent_name, model, history_options.continue_id)

    # Initialize REPL UI handler and custom logger
    ui_handler = ReplUIHandler(console, compact=True, show_observations=False)
    custom_logger = CustomUILogger(ui_handler, console)

    try:
        # Initialize chat manager with custom logger
        manager = ChatManager(
            agent_path=agent_path,
            model_override=exec_options.model_override,
            max_history=history_options.max_turns,
            custom_logger=custom_logger,
            stream=exec_options.stream,
            disable_history=not history_options.enabled,
            resume_conversation_id=history_options.continue_id,
        )

        # Load history if resuming
        if resume_turns:
            manager.load_from_history(history_options.continue_id, resume_turns)
            console.print(f"[cyan]Loaded {len(resume_turns)} previous turns[/cyan]")
            console.print()

        # Track session stats
        turn_count = len(manager.conversation_history)

        # Initialize prompt session with history and completion
        history_file = get_repl_history_path()
        completer = TsugiteCompleter(current_agent_name=agent_name)

        # Custom key bindings
        kb = KeyBindings()

        @kb.add("c-c")
        def _(event):
            """Handle Ctrl+C - clear current line or exit if empty."""
            if event.current_buffer.text:
                event.current_buffer.reset()
            else:
                raise KeyboardInterrupt()

        session = PromptSession(
            history=FileHistory(str(history_file)),
            completer=completer,
            auto_suggest=AutoSuggestFromHistory(),
            complete_while_typing=False,
            enable_history_search=True,
            key_bindings=kb,
            multiline=False,
        )

        # Main REPL loop
        while True:
            try:
                # Create prompt with agent name and turn count
                prompt_html = HTML(f"<ansigreen>{agent_name}</ansigreen>:<b>{turn_count}</b> &gt; ")

                # Get user input
                user_input = session.prompt(prompt_html)

                # Skip empty input
                if not user_input.strip():
                    continue

                # Handle slash commands
                if user_input.startswith("/"):
                    command, args, error = parse_command(user_input)

                    # Show error if command is invalid
                    if error:
                        from rich.panel import Panel

                        console.print(Panel(f"[yellow]{error}[/yellow]", border_style="yellow", padding=(0, 1)))
                        continue

                    if command in ("/exit", "/quit"):
                        console.print("\n[dim]Goodbye![/dim]")
                        break

                    elif command == "/help":
                        handle_help(console)

                    elif command == "/clear":
                        handle_clear(console)

                    elif command == "/stats":
                        # Update stats from manager
                        stats = manager.get_stats()

                        # Create a simple object with the stats
                        class StatsHolder:
                            def __init__(self, stats_dict):
                                self.turn_count = stats_dict["total_turns"]
                                self.total_tokens = stats_dict["total_tokens"] or 0
                                self.total_cost = stats_dict["total_cost"] or 0.0

                        holder = StatsHolder(stats)
                        handle_stats(console, holder)

                    elif command == "/history":
                        limit = int(args[0]) if args else 10
                        handle_history(console, limit)

                    elif command == "/attach":
                        if not args:
                            console.print("[red]Usage: /attach <path>[/red]")
                        else:
                            handle_attach(console, " ".join(args), manager)

                    elif command == "/detach":
                        if not args:
                            console.print("[red]Usage: /detach <path>[/red]")
                        else:
                            handle_detach(console, " ".join(args), manager)

                    elif command == "/list-attachments":
                        handle_list_attachments(console, manager)

                    elif command == "/save":
                        if not args:
                            console.print("[red]Usage: /save <path>[/red]")
                        else:
                            handle_save(console, " ".join(args), manager)

                    elif command == "/tools":
                        handle_tools(console, manager)

                    elif command == "/stream":
                        value = args[0] if args else None
                        handle_stream(console, value, manager)

                    elif command == "/multiline":
                        value = args[0] if args else None
                        handle_multiline(console, value)

                    elif command == "/verbose":
                        value = args[0] if args else None
                        handle_verbose(console, value, ui_handler)

                    elif command == "/continue":
                        console.print("[yellow]Cannot switch conversations in active session.[/yellow]")
                        console.print("[dim]Exit and use `tsugite chat --continue <id>` to resume.[/dim]")

                    elif command == "/model":
                        if not args:
                            console.print(f"[cyan]Current model: {manager.model_override or model}[/cyan]")
                        else:
                            new_model = args[0]
                            manager.model_override = new_model
                            console.print(f"[green]Model changed to: {new_model}[/green]")

                    elif command == "/agent":
                        console.print("[yellow]Cannot switch agents in active session.[/yellow]")
                        console.print("[dim]Exit and start a new chat with a different agent.[/dim]")

                    continue

                # Execute agent turn
                try:
                    response = manager.run_turn(user_input)
                    turn_count += 1

                    # If response starts with "Error:", display it (run_turn caught an exception)
                    # This happens when the agent fails before emitting events (e.g., invalid model)
                    if response and response.startswith("Error:"):
                        console.print(f"\n[red]{response}[/red]")
                        console.print("[dim]Tip: Check your model name with /model[/dim]")

                    # Update stats display
                    stats = manager.get_stats()
                    if stats.get("total_cost"):
                        # Stats are already shown by cost summary event
                        pass

                except KeyboardInterrupt:
                    console.print("\n[yellow]Turn cancelled[/yellow]")
                    ui_handler.stop()
                except Exception as e:
                    console.print(f"\n[red]Error: {e}[/red]")
                    ui_handler.stop()

            except KeyboardInterrupt:
                # Ctrl+C pressed - continue to next prompt
                console.print()
                continue
            except EOFError:
                # Ctrl+D pressed - exit
                console.print("\n[dim]Goodbye![/dim]")
                break

    finally:
        # Stop any active status
        ui_handler.stop()
