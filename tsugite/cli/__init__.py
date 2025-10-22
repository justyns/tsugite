"""Tsugite CLI application - main entry point."""

from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.traceback import install

from .agents import agents_app
from .attachments import attachments_app
from .benchmark import benchmark_command
from .cache import cache_app
from .config import config_app
from .helpers import (
    assemble_prompt_with_attachments,
    change_to_root_directory,
    get_error_console,
    get_logo,
    get_output_console,
    load_and_validate_agent,
    parse_cli_arguments,
    print_plain_info,
)
from .init import init
from .mcp import mcp_app
from .tools import tools_app

# Install rich traceback handler for better error messages
# This is after imports to satisfy linting, but still early enough to catch runtime errors
install(show_locals=False, width=None, word_wrap=True)

app = typer.Typer(
    name="tsugite",
    help="Micro-agent runner for task automation using markdown definitions",
    no_args_is_help=True,
)

console = Console()


@app.command()
def run(
    args: List[str] = typer.Argument(
        ..., help="Agent(s) and optional prompt (e.g., +assistant 'task' or +a +b +c 'task' or +a create ticket)"
    ),
    root: Optional[str] = typer.Option(None, "--root", help="Working directory"),
    with_agents: Optional[str] = typer.Option(
        None, "--with-agents", help="Additional agents (comma or space separated)"
    ),
    model: Optional[str] = typer.Option(None, "--model", help="Override agent model"),
    ui: Optional[str] = typer.Option(
        None, "--ui", help="UI mode: rich (default), plain, minimal, headless, silent, or live"
    ),
    non_interactive: bool = typer.Option(False, "--non-interactive", help="Run without interactive prompts"),
    history_dir: Optional[str] = typer.Option(None, "--history-dir", help="Directory to store history files"),
    no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI colors"),
    log_json: bool = typer.Option(False, "--log-json", help="Machine-readable output"),
    debug: bool = typer.Option(False, "--debug", help="Show rendered prompt before execution"),
    native_ui: bool = typer.Option(False, "--native-ui", help="Use minimal output without custom UI panels"),
    silent: bool = typer.Option(False, "--silent", help="Suppress all agent output"),
    show_reasoning: bool = typer.Option(
        True, "--show-reasoning/--no-show-reasoning", help="Show LLM reasoning messages (default: enabled)"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Show all execution details"),
    headless: bool = typer.Option(
        False, "--headless", help="Headless mode for CI/scripts: result to stdout, optional progress to stderr"
    ),
    plain: bool = typer.Option(False, "--plain", help="Plain output without panels/boxes (copy-paste friendly)"),
    stream: bool = typer.Option(False, "--stream", help="Stream LLM responses in real-time"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show execution plan without running agent"),
    trust_mcp_code: bool = typer.Option(False, "--trust-mcp-code", help="Trust remote code from MCP servers"),
    attachment: Optional[List[str]] = typer.Option(
        None, "-f", "--attachment", help="Attachment(s) to include (repeatable)"
    ),
    refresh_cache: bool = typer.Option(False, "--refresh-cache", help="Force refresh cached attachment content"),
    docker: bool = typer.Option(False, "--docker", help="Run in Docker container (delegates to tsugite-docker)"),
    keep: bool = typer.Option(False, "--keep", help="Keep Docker container running (use with --docker)"),
    container: Optional[str] = typer.Option(None, "--container", help="Use existing Docker container"),
    network: str = typer.Option("host", "--network", help="Docker network mode (use with --docker)"),
):
    """Run an agent with the given prompt.

    Examples:
        tsu run agent.md "prompt"
        tsu run +assistant "prompt"
        tsu run +assistant +jira +coder "prompt"
        tsu run +assistant create a ticket for bug 123
        tsu run +assistant --with-agents "jira,coder" "prompt"
    """
    # Lazy imports - only load heavy dependencies when actually running agents
    from tsugite.agent_runner import get_agent_info, run_agent, validate_agent_execution
    from tsugite.ui import create_live_template_logger, create_plain_logger, create_silent_logger, custom_agent_ui
    from tsugite.utils import should_use_plain_output

    if history_dir:
        Path(history_dir).mkdir(parents=True, exist_ok=True)

    if no_color:
        console.no_color = True

    # Handle --ui flag (maps to existing UI flags for convenience)
    live_ui = False  # Initialize live_ui flag
    if ui:
        ui_lower = ui.lower()
        # Check for conflicts
        if any([plain, headless, silent, native_ui]):
            console.print("[red]Error: --ui cannot be used with --plain, --headless, --silent, or --native-ui[/red]")
            raise typer.Exit(1)

        # Map UI mode to appropriate flags
        if ui_lower == "rich":
            pass  # Default - no changes needed
        elif ui_lower == "plain":
            plain = True
        elif ui_lower == "minimal":
            native_ui = True
        elif ui_lower == "headless":
            headless = True
        elif ui_lower == "silent":
            silent = True
        elif ui_lower == "live":
            live_ui = True
        else:
            console.print(
                f"[red]Error: Invalid UI mode '{ui}'. Choose from: rich, plain, minimal, headless, silent, live[/red]"
            )
            raise typer.Exit(1)

    # Delegate to tsugite-docker wrapper if Docker flags are present
    if docker or container:
        import shutil
        import subprocess

        # Check if tsugite-docker is available
        wrapper_path = shutil.which("tsugite-docker")
        if not wrapper_path:
            console.print("[red]Error: tsugite-docker wrapper not found in PATH[/red]")
            console.print("[yellow]Install it by adding tsugite/bin/ to your PATH[/yellow]")
            console.print("[dim]See bin/README.md for installation instructions[/dim]")
            raise typer.Exit(1)

        # Build command for wrapper
        cmd = ["tsugite-docker"]

        # Add wrapper-specific flags
        if network != "host":
            cmd.extend(["--network", network])
        if keep:
            cmd.append("--keep")
        if container:
            cmd.extend(["--container", container])

        # Add 'run' subcommand
        cmd.append("run")

        # Add all the original args (agent refs and prompt words)
        cmd.extend(args)

        # Add tsugite flags (not wrapper flags)
        if model:
            cmd.extend(["--model", model])
        if with_agents:
            cmd.extend(["--with-agents", with_agents])
        if root:
            cmd.extend(["--root", str(root)])
        if history_dir:
            cmd.extend(["--history-dir", str(history_dir)])
        if ui:
            cmd.extend(["--ui", ui])
        if debug:
            cmd.append("--debug")
        if verbose:
            cmd.append("--verbose")
        if headless:
            cmd.append("--headless")
        if plain:
            cmd.append("--plain")
        if show_reasoning:
            cmd.append("--show-reasoning")
        if no_color:
            cmd.append("--no-color")
        if silent:
            cmd.append("--silent")
        if log_json:
            cmd.append("--log-json")
        if non_interactive:
            cmd.append("--non-interactive")
        if native_ui:
            cmd.append("--native-ui")
        if trust_mcp_code:
            cmd.append("--trust-mcp-code")
        if attachment:
            for att in attachment:
                cmd.extend(["--attachment", att])
        if refresh_cache:
            cmd.append("--refresh-cache")

        # Execute wrapper
        result = subprocess.run(cmd)
        raise typer.Exit(result.returncode)

    # Parse CLI arguments into agents and prompt
    try:
        agent_refs, prompt = parse_cli_arguments(args)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    # Parse agent references and resolve paths
    from tsugite.agent_composition import parse_agent_references
    from tsugite.builtin_agents import is_builtin_agent_path

    with change_to_root_directory(root, console):
        try:
            base_dir = Path.cwd()

            primary_agent_path, delegation_agents = parse_agent_references(agent_refs, with_agents, base_dir)

            # Validate primary agent
            # Built-in agents have special paths starting with "<builtin-"
            # These don't need to exist on disk, so only check exists() for real files
            if not is_builtin_agent_path(primary_agent_path) and not primary_agent_path.exists():
                console.print(f"[red]Agent file not found: {primary_agent_path}[/red]")
                raise typer.Exit(1)

            if not is_builtin_agent_path(primary_agent_path) and primary_agent_path.suffix != ".md":
                console.print(f"[red]Agent file must be a .md file: {primary_agent_path}[/red]")
                raise typer.Exit(1)

            # Don't resolve builtin agent paths
            if is_builtin_agent_path(primary_agent_path):
                agent_file = primary_agent_path
            else:
                agent_file = primary_agent_path.resolve()

        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        # Determine if we should use plain output (no panels/boxes)
        # Plain mode is enabled if:
        # 1. User explicitly sets --plain flag, OR
        # 2. User explicitly sets --ui minimal (native_ui), OR
        # 3. Output is being piped/redirected (auto-detection), OR
        # 4. NO_COLOR environment variable is set (auto-detection)
        use_plain_output = plain or native_ui or should_use_plain_output()

        # Get agent info for display
        agent_info = get_agent_info(agent_file)
        instruction_label = "runtime + agent" if agent_info.get("instructions") else "runtime default"

        # Assemble prompt with all attachments and file references
        prompt, expanded_files = assemble_prompt_with_attachments(
            prompt=prompt,
            agent_attachments=agent_info.get("attachments"),
            cli_attachments=attachment,
            base_dir=base_dir,
            refresh_cache=refresh_cache,
            console=console,
        )

        # Skip initial panel in headless mode
        if not headless:
            # Display ASCII logo (adaptive based on terminal width) - skip in plain mode
            if not use_plain_output:
                console.print(get_logo(console), style="cyan")
                console.print()  # blank line for spacing

            # Prepare info items
            info_items = {
                "Agent": agent_file.name,
                "Task": prompt,
                "Directory": str(Path.cwd()),
                "Model": model or agent_info.get("model", "unknown"),
                "Instructions": instruction_label,
                "Tools": ", ".join(agent_info.get("tools", [])),
            }

            # Add agent attachments if any
            if agent_info.get("attachments"):
                info_items["Agent Attachments"] = ", ".join(agent_info["attachments"])

            # Add CLI attachments if any
            if attachment:
                info_items["CLI Attachments"] = ", ".join(attachment)

            # Add expanded files if any
            if expanded_files:
                info_items["Expanded Files"] = ", ".join(expanded_files)

            # Display as panel or plain text based on mode
            if use_plain_output:
                print_plain_info(console, "Tsugite Agent Runner", info_items, style="cyan")
            else:
                # Build panel content
                panel_content = "\n".join([f"[cyan]{label}:[/cyan] {value}" for label, value in info_items.items()])
                console.print(
                    Panel(
                        panel_content,
                        title="Tsugite Agent Runner",
                        border_style="blue",
                    )
                )

        # Validate agent before execution
        is_valid, error_msg = validate_agent_execution(agent_file)
        if not is_valid:
            get_error_console(headless, console).print(f"[red]Agent validation failed: {error_msg}[/red]")
            raise typer.Exit(1)

        # Detect if this is a multi-step agent
        from tsugite.agent_runner import preview_multistep_agent, run_multistep_agent
        from tsugite.md_agents import has_step_directives

        # Handle builtin agents for step directive checking
        if is_builtin_agent_path(agent_file):
            from tsugite.md_agents import parse_agent_file

            agent_obj = parse_agent_file(agent_file)
            agent_text = agent_obj.content
        else:
            agent_text = agent_file.read_text()
        is_multistep = has_step_directives(agent_text)

        # Handle dry-run mode
        if dry_run:
            if is_multistep:
                preview_multistep_agent(
                    agent_path=agent_file,
                    prompt=prompt,
                    console=console,
                )
            else:
                console.print("[yellow]Dry-run mode is for multi-step agents only.[/yellow]")
                console.print("[dim]This is a single-step agent. Use --debug to see the rendered prompt.[/dim]")
            return

        # Choose executor function
        executor = run_multistep_agent if is_multistep else run_agent

        # Skip "Starting agent execution" in headless mode
        if not headless:
            execution_type = "multi-step agent" if is_multistep else "agent"
            console.print(f"[green]Starting {execution_type} execution...[/green]")

        try:
            # Choose execution mode based on flags
            if silent:
                # Completely silent execution
                result = executor(
                    agent_path=agent_file,
                    prompt=prompt,
                    model_override=model,
                    debug=debug,
                    custom_logger=create_silent_logger(),
                    trust_mcp_code=trust_mcp_code,
                    delegation_agents=delegation_agents,
                    stream=stream,
                )
            elif headless:
                # Headless mode: stderr for progress (if verbose), stdout for result
                stderr_console = get_error_console(True, console)

                with custom_agent_ui(
                    console=stderr_console,
                    show_code=verbose,
                    show_observations=verbose,
                    show_progress=False,  # No spinners in headless
                    show_llm_messages=verbose,
                    show_execution_results=verbose,
                    show_execution_logs=verbose,
                    show_panels=False,  # No panels in headless
                ) as custom_logger:
                    result = executor(
                        agent_path=agent_file,
                        prompt=prompt,
                        model_override=model,
                        debug=debug,
                        custom_logger=custom_logger,
                        trust_mcp_code=trust_mcp_code,
                        delegation_agents=delegation_agents,
                        stream=stream,
                    )
            elif native_ui:
                # Minimal: colors and animations, but no panel boxes
                with custom_agent_ui(
                    console=console,
                    show_code=not non_interactive,
                    show_observations=not non_interactive,
                    show_progress=not no_color,
                    show_llm_messages=show_reasoning,
                    show_execution_results=True,
                    show_execution_logs=verbose,
                    show_panels=False,  # No panel boxes - just colors and animations
                ) as custom_logger:
                    result = executor(
                        agent_path=agent_file,
                        prompt=prompt,
                        model_override=model,
                        debug=debug,
                        custom_logger=custom_logger,
                        trust_mcp_code=trust_mcp_code,
                        delegation_agents=delegation_agents,
                        stream=stream,
                    )
            elif live_ui:
                # Live UI: Rich Live Display with Tree visualization and interactive prompts
                custom_logger = create_live_template_logger(interactive=not non_interactive)
                with custom_logger.ui_handler.progress_context():
                    result = executor(
                        agent_path=agent_file,
                        prompt=prompt,
                        model_override=model,
                        debug=debug,
                        custom_logger=custom_logger,
                        trust_mcp_code=trust_mcp_code,
                        delegation_agents=delegation_agents,
                        stream=stream,
                    )
            else:
                # Choose UI handler based on plain mode
                if use_plain_output:
                    # Use PlainUIHandler for copy-paste friendly output
                    custom_logger = create_plain_logger()
                    # Wrap in progress_context to register UI handler for interactive tools
                    with custom_logger.ui_handler.progress_context():
                        result = executor(
                            agent_path=agent_file,
                            prompt=prompt,
                            model_override=model,
                            debug=debug,
                            custom_logger=custom_logger,
                            trust_mcp_code=trust_mcp_code,
                            delegation_agents=delegation_agents,
                            stream=stream,
                        )
                else:
                    # Use custom UI with panels and formatting
                    with custom_agent_ui(
                        console=console,
                        show_code=not non_interactive,
                        show_observations=not non_interactive,
                        show_progress=not no_color,
                        show_llm_messages=show_reasoning,
                        show_execution_results=True,
                        show_execution_logs=verbose,
                        show_panels=True,
                    ) as custom_logger:
                        result = executor(
                            agent_path=agent_file,
                            prompt=prompt,
                            model_override=model,
                            debug=debug,
                            custom_logger=custom_logger,
                            trust_mcp_code=trust_mcp_code,
                            delegation_agents=delegation_agents,
                            stream=stream,
                        )

            # Display result
            if headless:
                # Headless: plain result to stdout
                get_output_console().print(result)
            elif native_ui:
                # Minimal: just the result, no banner
                console.print(result)
            elif not silent:
                console.print("\n" + "=" * 50)
                console.print("[bold green]Agent Execution Complete[/bold green]")
                console.print("=" * 50)
                console.print(result)

        except ValueError as e:
            get_error_console(headless, console).print(f"[red]Configuration error: {e}[/red]")
            raise typer.Exit(1)
        except RuntimeError as e:
            get_error_console(headless, console).print(f"[red]Execution error: {e}[/red]")
            raise typer.Exit(1)
        except KeyboardInterrupt:
            get_error_console(headless, console).print("\n[yellow]Agent execution interrupted by user[/yellow]")
            raise typer.Exit(130)
        except Exception as e:
            err_console = get_error_console(headless, console)
            err_console.print(f"[red]Unexpected error: {e}[/red]")
            if not log_json:
                err_console.print("\n[dim]Use --log-json for machine-readable output[/dim]")
            raise typer.Exit(1)


@app.command()
def render(
    agent_path: str = typer.Argument(help="Path to agent markdown file or builtin agent name (e.g., +builtin-default)"),
    prompt: Optional[str] = typer.Argument(default="", help="Prompt/task for the agent (optional)"),
    root: Optional[str] = typer.Option(None, "--root", help="Working directory"),
    no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI colors"),
    raw: bool = typer.Option(False, "--raw", help="Show raw Jinja templates in instructions without rendering"),
    attachment: Optional[List[str]] = typer.Option(
        None, "-f", "--attachment", help="Attachment(s) to include (repeatable)"
    ),
    refresh_cache: bool = typer.Option(False, "--refresh-cache", help="Force refresh cached attachment content"),
):
    """Render an agent template without executing it.

    Examples:
        tsu render agent.md "prompt"
        tsu render +builtin-default "prompt"
        tsu render builtin-default "prompt"
    """
    # Lazy imports
    from tsugite.agent_preparation import AgentPreparer

    if no_color:
        console.no_color = True

    with change_to_root_directory(root, console):
        try:
            # Load and validate agent (handles both builtin and file-based agents)
            agent, agent_file_path, agent_display_name = load_and_validate_agent(agent_path, console)

            base_dir = Path.cwd()

            # Assemble prompt with all attachments and file references
            prompt_expanded, _ = assemble_prompt_with_attachments(
                prompt=prompt,
                agent_attachments=agent.config.attachments,
                cli_attachments=attachment,
                base_dir=base_dir,
                refresh_cache=refresh_cache,
                console=console,
            )

            # Prepare agent (all rendering + tool building logic)
            preparer = AgentPreparer()
            prepared = preparer.prepare(
                agent=agent,
                prompt=prompt_expanded,
                skip_tool_directives=True,  # Render doesn't execute tool directives
            )

            # Display what will be sent to LLM
            console.print(
                Panel(
                    f"[cyan]Agent:[/cyan] {agent_display_name}\n"
                    f"[cyan]Prompt:[/cyan] {prompt}\n"
                    f"[cyan]Directory:[/cyan] {Path.cwd()}",
                    title="Tsugite Template Renderer",
                    border_style="green",
                )
            )

            console.print("\n" + "=" * 50)
            console.print("[bold green]System Message[/bold green] [dim](sent to LLM)[/dim]")
            console.print("=" * 50)
            console.print(prepared.system_message)

            console.print("\n" + "=" * 50)
            console.print("[bold green]User Message[/bold green] [dim](sent to LLM)[/dim]")
            console.print("=" * 50)
            console.print(prepared.user_message)

        except Exception as e:
            console.print(f"[red]Render error: {e}[/red]")
            raise typer.Exit(1)


@app.command()
def history(
    action: str = typer.Argument(help="Action: show, clear"),
):
    """View or manage execution history."""
    console.print(f"[yellow]History {action} not yet implemented[/yellow]")


@app.command()
def version():
    """Show version information."""
    from tsugite import __version__

    console.print(f"Tsugite version {__version__}")


@app.command()
def chat(
    agent: Optional[str] = typer.Argument(None, help="Agent name or path (optional, uses default if not provided)"),
    model: Optional[str] = typer.Option(None, "--model", help="Override agent model"),
    max_history: int = typer.Option(50, "--max-history", help="Maximum turns to keep in context"),
    stream: bool = typer.Option(False, "--stream", help="Stream LLM responses in real-time"),
    root: Optional[str] = typer.Option(None, "--root", help="Working directory"),
):
    """Start an interactive chat session with an agent."""
    from tsugite.agent_composition import parse_agent_references
    from tsugite.builtin_agents import is_builtin_agent_path
    from tsugite.ui.textual_chat import run_textual_chat

    with change_to_root_directory(root, console):
        # Resolve agent path
        if agent:
            # Parse agent reference
            base_dir = Path.cwd()
            agent_refs = [agent]
            primary_agent_path, _ = parse_agent_references(agent_refs, None, base_dir)
        else:
            # Use built-in chat assistant by default
            # Users can override by creating .tsugite/chat_assistant.md or agents/chat_assistant.md
            base_dir = Path.cwd()
            agent_refs = ["builtin-chat-assistant"]
            primary_agent_path, _ = parse_agent_references(agent_refs, None, base_dir)

        # Built-in agents have special paths starting with "<builtin-"
        # These don't need to exist on disk, so only check exists() for real files
        if not is_builtin_agent_path(primary_agent_path) and not primary_agent_path.exists():
            console.print(f"[red]Agent file not found: {primary_agent_path}[/red]")
            raise typer.Exit(1)

        # Run chat with Textual UI
        run_textual_chat(
            agent_path=primary_agent_path,
            model_override=model,
            max_history=max_history,
            stream=stream,
        )


# Register subcommands from separate modules
app.add_typer(mcp_app, name="mcp")
app.add_typer(agents_app, name="agents")
app.add_typer(config_app, name="config")
app.add_typer(attachments_app, name="attachments")
app.add_typer(cache_app, name="cache")
app.add_typer(tools_app, name="tools")
app.command("benchmark")(benchmark_command)
app.command("init")(init)
