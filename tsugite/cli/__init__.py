"""Tsugite CLI application - main entry point."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    inject_auto_context_if_enabled,
    load_and_validate_agent,
    print_plain_info,
)
from .history import history_app
from .init import init
from .mcp import mcp_app
from .tools import tools_app

DEFAULT_MAX_CHAT_HISTORY = 50

# Install rich traceback handler for better error messages
# This is after imports to satisfy linting, but still early enough to catch runtime errors
install(show_locals=False, width=None, word_wrap=True)

app = typer.Typer(
    name="tsugite",
    help="Micro-agent runner for task automation using markdown definitions",
    no_args_is_help=True,
)

# Global console for CLI messages (version, help, errors) - uses stdout
console = Console()


def _build_docker_command(
    args: List[str],
    network: str,
    keep: bool,
    container: Optional[str],
    model: Optional[str],
    with_agents: Optional[str],
    root: Optional[str],
    history_dir: Optional[str],
    ui: Optional[str],
    debug: bool,
    verbose: bool,
    headless: bool,
    plain: bool,
    show_reasoning: bool,
    no_color: bool,
    final_only: bool,
    log_json: bool,
    non_interactive: bool,
    trust_mcp_code: bool,
    attachment: Optional[List[str]],
    refresh_cache: bool,
) -> List[str]:
    """Build Docker wrapper command with all flags.

    Args:
        args: Agent references and prompt
        network: Docker network mode
        keep: Keep container running flag
        container: Existing container name
        model: Model override
        with_agents: Additional agents
        root: Working directory
        history_dir: History directory
        ui: UI mode
        debug: Debug flag
        verbose: Verbose flag
        headless: Headless flag
        plain: Plain output flag
        show_reasoning: Show reasoning flag
        no_color: No color flag
        final_only: Final only flag
        log_json: JSON logging flag
        non_interactive: Non-interactive flag
        trust_mcp_code: Trust MCP code flag
        attachment: Attachment list
        refresh_cache: Refresh cache flag

    Returns:
        Complete command list for subprocess execution
    """
    cmd = ["tsugite-docker"]

    if network != "host":
        cmd.extend(["--network", network])
    if keep:
        cmd.append("--keep")
    if container:
        cmd.extend(["--container", container])

    cmd.append("run")

    cmd.extend(args)

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
    if final_only:
        cmd.append("--final-only")
    if log_json:
        cmd.append("--log-json")
    if non_interactive:
        cmd.append("--non-interactive")
    if trust_mcp_code:
        cmd.append("--trust-mcp-code")
    if attachment:
        for att in attachment:
            cmd.extend(["--attachment", att])
    if refresh_cache:
        cmd.append("--refresh-cache")

    return cmd


def _resolve_ui_mode(ui: Optional[str], plain: bool, headless: bool, console: Console) -> tuple[bool, bool, bool]:
    """Resolve UI mode flag to individual UI control flags.

    Args:
        ui: UI mode string (plain, headless, live)
        plain: Plain output flag
        headless: Headless mode flag
        console: Console for error output

    Returns:
        Tuple of (plain, headless, live_ui) flags

    Raises:
        typer.Exit: If invalid UI mode or conflicting flags
    """
    live_ui = False

    if not ui:
        return plain, headless, live_ui

    if any([plain, headless]):
        console.print("[red]Error: --ui cannot be used with --plain or --headless[/red]")
        raise typer.Exit(1)

    ui_modes = {
        "plain": {"plain": True},
        "headless": {"headless": True},
        "live": {"live_ui": True},
    }

    ui_lower = ui.lower()
    if ui_lower not in ui_modes:
        console.print(f"[red]Error: Invalid UI mode '{ui}'. Choose from: {', '.join(ui_modes.keys())}[/red]")
        raise typer.Exit(1)

    mode_settings = ui_modes[ui_lower]
    plain = mode_settings.get("plain", plain)
    headless = mode_settings.get("headless", headless)
    live_ui = mode_settings.get("live_ui", live_ui)

    return plain, headless, live_ui


def _build_executor_kwargs(
    agent_file: Path,
    prompt: str,
    model: Optional[str],
    debug: bool,
    custom_logger: Any,
    trust_mcp_code: bool,
    delegation_agents: Any,
    stream: bool,
    continue_conversation_id: Optional[str],
    resolved_attachments: List[Tuple[str, str]],
    should_save_history: bool,
    executor: Any,
) -> Dict[str, Any]:
    """Build executor kwargs dict for run_agent/run_multistep_agent.

    Args:
        agent_file: Path to agent file
        prompt: User prompt
        model: Model override
        debug: Debug flag
        custom_logger: Logger instance
        trust_mcp_code: Trust MCP code flag
        delegation_agents: Delegation agents
        stream: Stream flag
        continue_conversation_id: Continuation conversation ID
        resolved_attachments: Resolved attachments
        should_save_history: Whether to save history
        executor: Executor function (run_agent or run_multistep_agent)

    Returns:
        Dict of executor kwargs
    """
    from tsugite.agent_runner import run_agent

    kwargs = {
        "agent_path": agent_file,
        "prompt": prompt,
        "model_override": model,
        "debug": debug,
        "custom_logger": custom_logger,
        "trust_mcp_code": trust_mcp_code,
        "delegation_agents": delegation_agents,
        "stream": stream,
        "continue_conversation_id": continue_conversation_id,
        "attachments": resolved_attachments,
    }
    if should_save_history and executor == run_agent:
        kwargs["return_token_usage"] = True
    return kwargs


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
    ui: Optional[str] = typer.Option(None, "--ui", help="UI mode: plain, headless, or live (default: minimal)"),
    non_interactive: bool = typer.Option(False, "--non-interactive", help="Run without interactive prompts"),
    history_dir: Optional[str] = typer.Option(None, "--history-dir", help="Directory to store history files"),
    no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI colors"),
    log_json: bool = typer.Option(False, "--log-json", help="Machine-readable output"),
    debug: bool = typer.Option(False, "--debug", help="Show rendered prompt before execution"),
    final_only: bool = typer.Option(
        False, "--final-only", "--quiet", help="Output only the final answer (suppress progress)"
    ),
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
    auto_context: Optional[bool] = typer.Option(
        None,
        "--auto-context/--no-auto-context",
        help="Enable/disable auto-context attachments (overrides config/agent)",
    ),
    docker: bool = typer.Option(False, "--docker", help="Run in Docker container (delegates to tsugite-docker)"),
    keep: bool = typer.Option(False, "--keep", help="Keep Docker container running (use with --docker)"),
    container: Optional[str] = typer.Option(None, "--container", help="Use existing Docker container"),
    network: str = typer.Option("host", "--network", help="Docker network mode (use with --docker)"),
    no_history: bool = typer.Option(False, "--no-history", help="Disable conversation history persistence"),
    continue_conversation: bool = typer.Option(False, "--continue", "-c", help="Continue previous conversation"),
    conversation_id: Optional[str] = typer.Option(
        None, "--conversation-id", help="Specific conversation ID to continue (use with --continue)"
    ),
    subagent_mode: bool = typer.Option(
        False, "--subagent-mode", help="Run as subagent: read JSON from stdin, emit JSONL to stdout"
    ),
):
    """Run an agent with the given prompt.

    Examples:
        tsu run agent.md "prompt"
        tsu run +assistant "prompt"
        tsu run +assistant +jira +coder "prompt"
        tsu run +assistant create a ticket for bug 123
        tsu run +assistant --with-agents "jira,coder" "prompt"
        tsu run --continue "prompt"  # Continue latest conversation (auto-detect agent)
        tsu run +assistant --continue "prompt"  # Continue latest with specific agent
        tsu run --continue --conversation-id CONV_ID "prompt"  # Continue specific conversation
    """
    # Lazy imports - only load heavy dependencies when actually running agents
    from tsugite.agent_runner import get_agent_info, run_agent
    from tsugite.md_agents import validate_agent_execution
    from tsugite.ui import create_live_template_logger, create_plain_logger, custom_agent_ui
    from tsugite.utils import should_use_plain_output

    if history_dir:
        Path(history_dir).mkdir(parents=True, exist_ok=True)

    if no_color:
        console.no_color = True

    # Resolve UI mode to individual flags
    plain, headless, live_ui = _resolve_ui_mode(ui, plain, headless, console)

    # Handle subagent mode - override incompatible settings
    if subagent_mode:
        import os

        # Validate no conflicting flags
        if plain or headless or live_ui:
            console.print("[red]Error: --subagent-mode cannot be combined with --plain, --headless, or --live[/red]")
            raise typer.Exit(1)

        non_interactive = True
        no_history = True
        os.environ["TSUGITE_SUBAGENT_MODE"] = "1"

    if docker or container:
        import shutil
        import subprocess

        wrapper_path = shutil.which("tsugite-docker")
        if not wrapper_path:
            console.print("[red]Error: tsugite-docker wrapper not found in PATH[/red]")
            console.print("[yellow]Install it by adding tsugite/bin/ to your PATH[/yellow]")
            console.print("[dim]See bin/README.md for installation instructions[/dim]")
            raise typer.Exit(1)
        cmd = _build_docker_command(
            args,
            network,
            keep,
            container,
            model,
            with_agents,
            root,
            history_dir,
            ui,
            debug,
            verbose,
            headless,
            plain,
            show_reasoning,
            no_color,
            final_only,
            log_json,
            non_interactive,
            trust_mcp_code,
            attachment,
            refresh_cache,
        )
        result = subprocess.run(cmd, check=False)
        raise typer.Exit(result.returncode)

    # Handle conversation continuation - check before parsing args
    continue_conversation_id = None
    if continue_conversation:
        from tsugite.ui.chat_history import get_latest_conversation

        # Determine conversation ID
        if conversation_id:
            continue_conversation_id = conversation_id
            console.print(f"[cyan]Continuing conversation: {continue_conversation_id}[/cyan]")
        else:
            continue_conversation_id = get_latest_conversation()
            if not continue_conversation_id:
                console.print("[red]No conversations found to resume[/red]")
                raise typer.Exit(1)
            console.print(f"[cyan]Continuing latest conversation: {continue_conversation_id}[/cyan]")

    # Parse CLI arguments into agents and prompt (allow empty agents when continuing)
    try:
        from tsugite.cli.helpers import parse_cli_arguments

        agent_refs, prompt, stdin_attachment = parse_cli_arguments(
            args, allow_empty_agents=continue_conversation, check_stdin=not continue_conversation
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    from tsugite.agent_composition import parse_agent_references

    with change_to_root_directory(root, console):
        try:
            base_dir = Path.cwd()

            if not agent_refs and continue_conversation:
                from tsugite.history import get_conversation_metadata

                metadata = get_conversation_metadata(continue_conversation_id)
                if not metadata:
                    console.print(f"[red]Could not load metadata for conversation: {continue_conversation_id}[/red]")
                    raise typer.Exit(1)

                agent_name = metadata.agent
                console.print(f"[cyan]Auto-detected agent from conversation: {agent_name}[/cyan]")
                agent_refs = [f"+{agent_name}"]

            primary_agent_path, delegation_agents = parse_agent_references(agent_refs, with_agents, base_dir)

            if not primary_agent_path.exists():
                console.print(f"[red]Agent file not found: {primary_agent_path}[/red]")
                raise typer.Exit(1)

            if primary_agent_path.suffix != ".md":
                console.print(f"[red]Agent file must be a .md file: {primary_agent_path}[/red]")
                raise typer.Exit(1)

            agent_file = primary_agent_path.resolve()

        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        use_plain_output = plain or should_use_plain_output()

        stderr_console = Console(file=sys.stderr, no_color=no_color)

        agent_info = get_agent_info(agent_file)
        instruction_label = "runtime + agent" if agent_info.get("instructions") else "runtime default"

        agent_attachments = inject_auto_context_if_enabled(
            agent_info.get("attachments"),
            agent_info.get("auto_context"),
            cli_override=auto_context,
        )

        prompt, resolved_attachments = assemble_prompt_with_attachments(
            prompt=prompt,
            agent_attachments=agent_attachments,
            cli_attachments=attachment,
            base_dir=base_dir,
            refresh_cache=refresh_cache,
            console=console,
            stdin_attachment=stdin_attachment,
        )

        if not headless:
            if not use_plain_output:
                stderr_console.print(get_logo(stderr_console), style="cyan")
                stderr_console.print()

            info_items = {
                "Agent": agent_file.name,
                "Task": prompt,
                "Directory": str(Path.cwd()),
                "Model": model or agent_info.get("model", "unknown"),
                "Instructions": instruction_label,
                "Tools": ", ".join(agent_info.get("tools", [])),
            }

            if agent_attachments:
                info_items["Agent Attachments"] = ", ".join(agent_attachments)

            if attachment:
                info_items["CLI Attachments"] = ", ".join(attachment)

            if resolved_attachments:
                info_items["Attachments"] = f"{len(resolved_attachments)} file(s)"

            if use_plain_output:
                print_plain_info(stderr_console, "Tsugite Agent Runner", info_items, style="cyan")

        is_valid, error_msg = validate_agent_execution(agent_file)
        if not is_valid:
            get_error_console(headless, console).print(f"[red]Agent validation failed: {error_msg}[/red]")
            raise typer.Exit(1)

        from tsugite.agent_runner import preview_multistep_agent, run_multistep_agent
        from tsugite.md_agents import has_step_directives

        agent_text = agent_file.read_text()
        is_multistep = has_step_directives(agent_text)

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

        executor = run_multistep_agent if is_multistep else run_agent

        should_save_history = not no_history

        if not headless and not final_only:
            execution_type = "multi-step agent" if is_multistep else "agent"
            stderr_console.print()
            stderr_console.rule(f"[bold cyan]🚀 Starting {execution_type.title()} Execution[/bold cyan]")
            stderr_console.print()

        try:
            # Choose execution mode based on flags
            if headless or final_only:
                # Headless/final-only mode: stderr for progress (if verbose and not final_only), stdout for result
                stderr_console = get_error_console(True, console)

                # In final_only mode, suppress all progress; in headless, respect verbose flag
                show_progress_items = verbose and not final_only

                with custom_agent_ui(
                    console=stderr_console,
                    show_code=show_progress_items,
                    show_observations=show_progress_items,
                    show_progress=False,
                    show_llm_messages=show_progress_items,
                    show_execution_results=show_progress_items,
                    show_execution_logs=show_progress_items,
                    show_panels=False,
                    show_debug_messages=verbose,
                ) as custom_logger:
                    executor_kwargs = _build_executor_kwargs(
                        agent_file,
                        prompt,
                        model,
                        debug,
                        custom_logger,
                        trust_mcp_code,
                        delegation_agents,
                        stream,
                        continue_conversation_id,
                        resolved_attachments,
                        should_save_history,
                        executor,
                    )
                    result = executor(**executor_kwargs)
            elif live_ui:
                custom_logger = create_live_template_logger(interactive=not non_interactive)
                with custom_logger.ui_handler.progress_context():
                    executor_kwargs = _build_executor_kwargs(
                        agent_file,
                        prompt,
                        model,
                        debug,
                        custom_logger,
                        trust_mcp_code,
                        delegation_agents,
                        stream,
                        continue_conversation_id,
                        resolved_attachments,
                        should_save_history,
                        executor,
                    )
                    result = executor(**executor_kwargs)
            else:
                if use_plain_output:
                    custom_logger = create_plain_logger()
                    with custom_logger.ui_handler.progress_context():
                        executor_kwargs = _build_executor_kwargs(
                            agent_file,
                            prompt,
                            model,
                            debug,
                            custom_logger,
                            trust_mcp_code,
                            delegation_agents,
                            stream,
                            continue_conversation_id,
                            resolved_attachments,
                            should_save_history,
                            executor,
                        )
                        result = executor(**executor_kwargs)
                else:
                    default_console = Console(file=sys.stderr, force_terminal=True, no_color=no_color)
                    with custom_agent_ui(
                        console=default_console,
                        show_code=not non_interactive,
                        show_observations=not non_interactive,
                        show_progress=not no_color,
                        show_llm_messages=show_reasoning,
                        show_execution_results=True,
                        show_execution_logs=verbose,
                        show_panels=False,
                        show_debug_messages=verbose,
                    ) as custom_logger:
                        executor_kwargs = _build_executor_kwargs(
                            agent_file,
                            prompt,
                            model,
                            debug,
                            custom_logger,
                            trust_mcp_code,
                            delegation_agents,
                            stream,
                            continue_conversation_id,
                            resolved_attachments,
                            should_save_history,
                            executor,
                        )
                        result = executor(**executor_kwargs)

            # Unpack result if history was enabled (run_agent returns tuple with metadata)
            if should_save_history and executor == run_agent and isinstance(result, tuple):
                result_str, token_count, cost, step_count, execution_steps, system_prompt, attachments = result
            else:
                result_str = result if not isinstance(result, tuple) else result[0]
                token_count = None
                cost = None
                execution_steps = None
                system_prompt = None
                attachments = None

            # Save to history (unless disabled)
            if should_save_history:
                try:
                    from tsugite.agent_runner.history_integration import save_run_to_history

                    agent_info = get_agent_info(agent_file)
                    save_run_to_history(
                        agent_path=agent_file,
                        agent_name=agent_info["name"],
                        prompt=prompt,
                        result=result_str,
                        model=model or agent_info.get("model", "default"),
                        token_count=token_count,
                        cost=cost,
                        execution_steps=execution_steps,
                        continue_conversation_id=continue_conversation_id,
                        system_prompt=system_prompt,
                        attachments=attachments,
                    )
                except Exception as e:
                    # Don't fail the run if history save fails
                    print(f"Warning: Failed to save to history: {e}", file=sys.stderr)

            # Display result
            if headless or final_only:
                # Headless/final-only: render markdown to stdout
                from rich.markdown import Markdown

                from tsugite.console import get_stdout_console

                get_stdout_console(no_color=no_color, force_terminal=True).print(Markdown(result_str))
            else:
                # Show completion banner to stderr
                stderr_console.print()
                stderr_console.rule("[bold green]Agent Execution Complete[/bold green]")
                # Output final result to stdout (for piping) with markdown rendering
                from rich.markdown import Markdown

                from tsugite.console import get_stdout_console

                get_stdout_console(no_color=no_color, force_terminal=True).print(Markdown(result_str))

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
    agent_path: Optional[str] = typer.Argument(
        None, help="Path to agent markdown file or builtin agent name (optional when using --continue)"
    ),
    prompt: Optional[str] = typer.Argument(default="", help="Prompt/task for the agent (optional)"),
    root: Optional[str] = typer.Option(None, "--root", help="Working directory"),
    no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI colors"),
    verbose: bool = typer.Option(False, "--verbose", help="Show full attachment content (default: truncated)"),
    raw: bool = typer.Option(False, "--raw", help="Show raw Jinja templates in instructions without rendering"),
    attachment: Optional[List[str]] = typer.Option(
        None, "-f", "--attachment", help="Attachment(s) to include (repeatable)"
    ),
    refresh_cache: bool = typer.Option(False, "--refresh-cache", help="Force refresh cached attachment content"),
    auto_context: Optional[bool] = typer.Option(
        None,
        "--auto-context/--no-auto-context",
        help="Enable/disable auto-context attachments (overrides config/agent)",
    ),
    continue_conversation: bool = typer.Option(
        False, "--continue", "-c", help="Show prompt for continuing conversation"
    ),
    conversation_id: Optional[str] = typer.Option(
        None, "--conversation-id", help="Specific conversation ID (use with --continue)"
    ),
):
    """Render an agent template without executing it.

    Examples:
        tsu render agent.md "prompt"
        tsu render +builtin-default "prompt"
        tsu render builtin-default "prompt"
        tsu render --continue "prompt"  # Auto-detects agent
        tsu render agent.md "prompt" --continue
        tsu render --continue --conversation-id CONV_ID "prompt"
    """
    # Lazy imports
    from tsugite.agent_preparation import AgentPreparer

    if no_color:
        console.no_color = True

    # Handle conversation continuation
    continue_conversation_id = None
    if continue_conversation:
        from tsugite.ui.chat_history import get_latest_conversation

        if conversation_id:
            continue_conversation_id = conversation_id
            console.print(f"[cyan]Rendering for conversation: {continue_conversation_id}[/cyan]")
        else:
            continue_conversation_id = get_latest_conversation()
            if not continue_conversation_id:
                console.print("[red]No conversations found[/red]")
                raise typer.Exit(1)
            console.print(f"[cyan]Rendering for latest conversation: {continue_conversation_id}[/cyan]")

    # Auto-detect agent from conversation if not specified
    if continue_conversation_id and not agent_path:
        from tsugite.history import get_conversation_metadata

        metadata = get_conversation_metadata(continue_conversation_id)
        if not metadata:
            console.print(f"[red]Could not load metadata for conversation: {continue_conversation_id}[/red]")
            raise typer.Exit(1)

        agent_path = f"+{metadata.agent}"
        console.print(f"[cyan]Auto-detected agent from conversation: {metadata.agent}[/cyan]")

    # Validate agent_path is provided
    if not agent_path:
        console.print("[red]Error: AGENT_PATH is required (or use --continue to auto-detect)[/red]")
        raise typer.Exit(1)

    with change_to_root_directory(root, console):
        try:
            # Load and validate agent (handles both builtin and file-based agents)
            agent, agent_file_path, agent_display_name = load_and_validate_agent(agent_path, console)

            base_dir = Path.cwd()

            # Inject auto-context if enabled
            agent_attachments = inject_auto_context_if_enabled(
                agent.config.attachments,
                agent.config.auto_context,
                cli_override=auto_context,
            )

            prompt_updated, resolved_attachments = assemble_prompt_with_attachments(
                prompt=prompt,
                agent_attachments=agent_attachments,
                cli_attachments=attachment,
                base_dir=base_dir,
                refresh_cache=refresh_cache,
                console=console,
            )

            # Build context
            # Note: Conversation history is no longer loaded here because chat_history
            # template blocks have been removed from modern agents. History is now
            # handled via previous_messages in agent execution, not template rendering.
            context = {}
            if continue_conversation_id:
                # Enable text_mode when continuing (matches run behavior)
                agent.config.text_mode = True

            # Prepare agent (all rendering + tool building logic)
            preparer = AgentPreparer()
            prepared = preparer.prepare(
                agent=agent,
                prompt=prompt_updated,
                skip_tool_directives=True,  # Render doesn't execute tool directives
                context=context,
                attachments=resolved_attachments,  # Pass attachments separately
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

            # Show message structure
            console.print()
            console.rule(
                "[bold yellow]MESSAGE STRUCTURE[/bold yellow] [dim](sent to LLM as separate content blocks)[/dim]",
                style="yellow",
            )

            # Message 1: System (role: system)
            console.print()
            console.rule("[bold cyan]Message 1: System Role[/bold cyan]", style="cyan", align="left")

            # Content Block 1: System Instructions
            console.print()
            console.rule("[dim]Content Block 1: System Instructions[/dim]", style="dim", align="left")
            console.print(prepared.system_message)

            # Display attachments if they exist (as additional content blocks in system message)
            if prepared.attachments:
                for idx, (name, content) in enumerate(prepared.attachments, start=2):
                    console.print()
                    console.rule(
                        f"[dim]Content Block {idx}: Attachment - {name}[/dim]",
                        style="dim",
                        align="left",
                    )
                    console.print(f"[yellow]<Attachment: {name}>[/yellow]")

                    # Truncate unless verbose
                    if verbose:
                        console.print(content)
                    else:
                        # Show first 10 lines and last 5 lines
                        lines = content.split("\n")
                        if len(lines) > 20:
                            preview = "\n".join(lines[:10])
                            preview += (
                                f"\n[dim]... ({len(lines) - 15} lines truncated, use --verbose to see all) ...[/dim]\n"
                            )
                            preview += "\n".join(lines[-5:])
                            console.print(preview)
                        else:
                            console.print(content)

                    console.print(f"[yellow]</Attachment: {name}>[/yellow]")

            # Message 2: User (role: user)
            console.print()
            console.rule("[bold cyan]Message 2: User Role[/bold cyan]", style="cyan", align="left")
            console.print()
            console.rule("[dim]Content: User Task/Prompt[/dim]", style="dim", align="left")
            console.print(prepared.user_message)
            console.print()
            console.rule(style="dim")

        except Exception as e:
            console.print(f"[red]Render error: {e}[/red]")
            raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    from tsugite import __version__

    console.print(f"Tsugite version {__version__}")


@app.command()
def chat(
    agent: Optional[str] = typer.Argument(None, help="Agent name or path (optional, uses default if not provided)"),
    model: Optional[str] = typer.Option(None, "--model", help="Override agent model"),
    max_history: int = typer.Option(DEFAULT_MAX_CHAT_HISTORY, "--max-history", help="Maximum turns to keep in context"),
    stream: bool = typer.Option(False, "--stream", help="Stream LLM responses in real-time"),
    no_history: bool = typer.Option(False, "--no-history", help="Disable conversation history persistence"),
    continue_: Optional[str] = typer.Option(
        None, "--continue", "-c", help="Resume conversation by ID, or latest if no ID given"
    ),
    root: Optional[str] = typer.Option(None, "--root", help="Working directory"),
):
    """Start an interactive chat session with an agent."""
    from tsugite.agent_composition import parse_agent_references
    from tsugite.ui.textual_chat import run_textual_chat

    with change_to_root_directory(root, console):
        # Handle conversation resume
        resume_conversation_id = None
        resume_turns = None

        if continue_ is not None:
            from tsugite.ui.chat_history import get_latest_conversation, load_conversation_history

            # Determine conversation ID to resume
            if continue_ == "" or continue_.lower() == "latest":
                resume_conversation_id = get_latest_conversation()
                if not resume_conversation_id:
                    console.print("[red]No conversations found to resume[/red]")
                    raise typer.Exit(1)
                console.print(f"[cyan]Resuming latest conversation: {resume_conversation_id}[/cyan]")
            else:
                resume_conversation_id = continue_
                console.print(f"[cyan]Resuming conversation: {resume_conversation_id}[/cyan]")

            # Load conversation history
            try:
                resume_turns = load_conversation_history(resume_conversation_id)
                console.print(f"[cyan]Loaded {len(resume_turns)} previous turns[/cyan]")
            except FileNotFoundError:
                console.print(f"[red]Conversation not found: {resume_conversation_id}[/red]")
                raise typer.Exit(1)
            except Exception as e:
                console.print(f"[red]Failed to load conversation: {e}[/red]")
                raise typer.Exit(1)

        # Resolve agent path
        if agent:
            # Parse agent reference
            base_dir = Path.cwd()
            agent_refs = [agent]
            primary_agent_path, _ = parse_agent_references(agent_refs, None, base_dir)
        else:
            # Use package-provided chat assistant by default
            # Users can override by creating .tsugite/chat_assistant.md or agents/chat_assistant.md
            base_dir = Path.cwd()
            agent_refs = ["chat-assistant"]
            primary_agent_path, _ = parse_agent_references(agent_refs, None, base_dir)

        # Validate agent file exists
        if not primary_agent_path.exists():
            console.print(f"[red]Agent file not found: {primary_agent_path}[/red]")
            raise typer.Exit(1)

        # Run chat with Textual UI
        run_textual_chat(
            agent_path=primary_agent_path,
            model_override=model,
            max_history=max_history,
            stream=stream,
            disable_history=no_history,
            resume_conversation_id=resume_conversation_id,
            resume_turns=resume_turns,
        )


# Register subcommands from separate modules
app.add_typer(mcp_app, name="mcp")
app.add_typer(agents_app, name="agents")
app.add_typer(config_app, name="config")
app.add_typer(attachments_app, name="attachments")
app.add_typer(cache_app, name="cache")
app.add_typer(tools_app, name="tools")
app.add_typer(history_app, name="history")
app.command("benchmark")(benchmark_command)
app.command("init")(init)
