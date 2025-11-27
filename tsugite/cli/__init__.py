"""Tsugite CLI application - main entry point."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from rich.console import Console
from rich.panel import Panel
from rich.traceback import install

from .helpers import (
    assemble_prompt_with_attachments,
    change_to_root_directory,
    get_error_console,
    get_logo,
    inject_auto_context_if_enabled,
    load_and_validate_agent,
    print_plain_info,
)

# Chat history limit - keeps last N turns to balance context retention vs memory usage
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
        "stream": stream,
        "continue_conversation_id": continue_conversation_id,
        "attachments": resolved_attachments,
    }
    if should_save_history and executor == run_agent:
        kwargs["return_token_usage"] = True
    return kwargs


def _handle_docker_execution(
    args: List[str],
    network: str,
    keep: bool,
    container: Optional[str],
    model: Optional[str],
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
) -> None:
    """Handle Docker container execution and exit.

    Args:
        All run() command parameters

    Raises:
        typer.Exit: Always exits after Docker execution
    """
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


def _resolve_conversation_continuation(continue_conversation: bool, conversation_id: Optional[str]) -> Optional[str]:
    """Resolve which conversation to continue.

    Args:
        continue_conversation: Whether to continue a conversation
        conversation_id: Specific conversation ID or None for latest

    Returns:
        Conversation ID to continue, or None if not continuing

    Raises:
        typer.Exit: If no conversations found
    """
    if not continue_conversation:
        return None

    from tsugite.ui.chat_history import get_latest_conversation

    if conversation_id:
        console.print(f"[cyan]Continuing conversation: {conversation_id}[/cyan]")
        return conversation_id

    continue_conversation_id = get_latest_conversation()
    if not continue_conversation_id:
        console.print("[red]No conversations found to resume[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Continuing latest conversation: {continue_conversation_id}[/cyan]")
    return continue_conversation_id


def _execute_agent_with_ui(
    executor,
    executor_kwargs: Dict[str, Any],
    headless: bool,
    final_only: bool,
    verbose: bool,
    live_ui: bool,
    use_plain_output: bool,
    non_interactive: bool,
    no_color: bool,
    show_reasoning: bool,
):
    """Execute agent with appropriate UI mode."""
    from tsugite.ui import create_live_template_logger, create_plain_logger, custom_agent_ui

    if headless or final_only:
        from .helpers import get_error_console

        stderr_console = get_error_console(True, console)
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
            executor_kwargs["custom_logger"] = custom_logger
            return executor(**executor_kwargs)

    if live_ui:
        custom_logger = create_live_template_logger(interactive=not non_interactive)
        with custom_logger.ui_handler.progress_context():
            executor_kwargs["custom_logger"] = custom_logger
            return executor(**executor_kwargs)

    if use_plain_output:
        custom_logger = create_plain_logger()
        with custom_logger.ui_handler.progress_context():
            executor_kwargs["custom_logger"] = custom_logger
            return executor(**executor_kwargs)

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
        executor_kwargs["custom_logger"] = custom_logger
        return executor(**executor_kwargs)


def _unpack_execution_result(result, should_save_history: bool, executor):
    """Unpack execution result based on whether history was enabled."""
    from tsugite.agent_runner import run_agent
    from tsugite.agent_runner.models import AgentExecutionResult

    # Handle AgentExecutionResult object (new format when return_token_usage=True)
    if isinstance(result, AgentExecutionResult):
        return (
            result.response,
            result.token_count,
            result.cost,
            result.execution_steps,
            result.system_message,
            result.attachments,
        )

    # Handle old tuple format (for backward compatibility)
    if should_save_history and executor == run_agent and isinstance(result, tuple):
        result_str, token_count, cost, step_count, execution_steps, system_prompt, attachments = result
        return result_str, token_count, cost, execution_steps, system_prompt, attachments

    # Handle plain string result
    result_str = result if not isinstance(result, tuple) else result[0]
    return result_str, None, None, None, None, None


def _display_result(result_str: str, headless: bool, final_only: bool, no_color: bool, stderr_console: Console):
    """Display the final result to the user."""
    from rich.markdown import Markdown

    from tsugite.console import get_stdout_console

    if headless or final_only:
        get_stdout_console(no_color=no_color, force_terminal=True).print(Markdown(result_str))
    else:
        stderr_console.print()
        stderr_console.rule("[bold green]Agent Execution Complete[/bold green]")
        get_stdout_console(no_color=no_color, force_terminal=True).print(Markdown(result_str))


@app.command()
def run(
    args: List[str] = typer.Argument(
        ..., help="Agent and optional prompt (e.g., +assistant 'task' or +assistant create ticket)"
    ),
    root: Optional[str] = typer.Option(None, "--root", help="Working directory"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override agent model"),
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
        tsu run +assistant create a ticket for bug 123
        tsu run --continue "prompt"  # Continue latest conversation (auto-detect agent)
        tsu run +assistant --continue "prompt"  # Continue latest with specific agent
        tsu run --continue --conversation-id CONV_ID "prompt"  # Continue specific conversation
    """
    # Lazy imports - only load heavy dependencies when actually running agents
    from tsugite.agent_runner import get_agent_info, run_agent
    from tsugite.md_agents import validate_agent_execution
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
        _handle_docker_execution(
            args,
            network,
            keep,
            container,
            model,
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

    # Handle conversation continuation - check before parsing args
    continue_conversation_id = _resolve_conversation_continuation(continue_conversation, conversation_id)

    # Parse CLI arguments into agents and prompt (allow empty agents when continuing)
    try:
        from tsugite.cli.helpers import parse_cli_arguments

        agent_refs, prompt, stdin_attachment = parse_cli_arguments(
            args, allow_empty_agents=continue_conversation, check_stdin=not continue_conversation
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

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

            # Handle multi-agent mode: first agent is primary, rest are allowed to spawn
            if not agent_refs:
                console.print("[red]Error: No agent specified[/red]")
                raise typer.Exit(1)

            from tsugite.agent_runner.helpers import set_allowed_agents
            from tsugite.md_agents import parse_agent_file

            primary_agent_ref = agent_refs[0]

            if len(agent_refs) > 1:
                # Multi-agent mode: validate all agents and extract names
                allowed_agent_names = []
                for allowed_ref in agent_refs[1:]:
                    _, allowed_file, _ = load_and_validate_agent(allowed_ref, console)
                    allowed_agent = parse_agent_file(allowed_file)
                    allowed_agent_names.append(allowed_agent.config.name)

                set_allowed_agents(allowed_agent_names)
                console.print(f"[cyan]Allowed agents to spawn: {', '.join(allowed_agent_names)}[/cyan]")
            else:
                # Single agent mode: unrestricted spawning
                set_allowed_agents(None)

            # Load and validate primary agent using shared helper
            _, agent_file, _ = load_and_validate_agent(primary_agent_ref, console)

        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        use_plain_output = plain or should_use_plain_output()

        from tsugite.console import get_stderr_console

        stderr_console = get_stderr_console(no_color=no_color)

        # Set up event bus in context for attachment loading
        from tsugite.events import EventBus
        from tsugite.ui_context import set_ui_context

        event_bus = EventBus()
        set_ui_context(event_bus=event_bus)

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
            executor_kwargs = _build_executor_kwargs(
                agent_file,
                prompt,
                model,
                debug,
                None,
                trust_mcp_code,
                stream,
                continue_conversation_id,
                resolved_attachments,
                should_save_history,
                executor,
            )

            result = _execute_agent_with_ui(
                executor,
                executor_kwargs,
                headless,
                final_only,
                verbose,
                live_ui,
                use_plain_output,
                non_interactive,
                no_color,
                show_reasoning,
            )

            result_str, token_count, cost, execution_steps, system_prompt, attachments = _unpack_execution_result(
                result, should_save_history, executor
            )

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
                except Exception:
                    pass

            _display_result(result_str, headless, final_only, no_color, stderr_console)

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
                from tsugite.attachments.base import AttachmentContentType

                for idx, attachment in enumerate(prepared.attachments, start=2):
                    console.print()
                    console.rule(
                        f"[dim]Content Block {idx}: Attachment - {attachment.name}[/dim]",
                        style="dim",
                        align="left",
                    )
                    console.print(f"[yellow]<Attachment: {attachment.name}>[/yellow]")

                    # Handle different content types
                    if attachment.content_type == AttachmentContentType.TEXT and attachment.content:
                        # Truncate text unless verbose
                        if verbose:
                            console.print(attachment.content)
                        else:
                            # Show first 10 lines and last 5 lines
                            lines = attachment.content.split("\n")
                            if len(lines) > 20:
                                preview = "\n".join(lines[:10])
                                preview += (
                                    f"\n[dim]... ({len(lines) - 15} lines truncated, use --verbose to see all) ...[/dim]\n"
                                )
                                preview += "\n".join(lines[-5:])
                                console.print(preview)
                            else:
                                console.print(attachment.content)
                    elif attachment.source_url:
                        console.print(f"[dim][{attachment.content_type.value}: {attachment.source_url}][/dim]")
                    else:
                        console.print(f"[dim][{attachment.content_type.value} file: {attachment.mime_type}][/dim]")

                    console.print(f"[yellow]</Attachment: {attachment.name}>[/yellow]")

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
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override agent model"),
    max_history: int = typer.Option(DEFAULT_MAX_CHAT_HISTORY, "--max-history", help="Maximum turns to keep in context"),
    stream: bool = typer.Option(False, "--stream", help="Stream LLM responses in real-time"),
    no_history: bool = typer.Option(False, "--no-history", help="Disable conversation history persistence"),
    continue_: Optional[str] = typer.Option(
        None, "--continue", "-c", help="Resume conversation by ID, or latest if no ID given"
    ),
    ui: str = typer.Option("repl", "--ui", help="UI mode: 'repl' (default) or 'tui'"),
    root: Optional[str] = typer.Option(None, "--root", help="Working directory"),
):
    """Start an interactive chat session with an agent."""
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

        # Resolve agent path using shared helper
        # Default to chat-assistant if no agent specified
        agent_to_load = agent if agent else "chat-assistant"
        _, primary_agent_path, _ = load_and_validate_agent(agent_to_load, console)

        # Run chat with selected UI
        if ui.lower() == "tui":
            from tsugite.ui.textual_chat import run_textual_chat

            run_textual_chat(
                agent_path=primary_agent_path,
                model_override=model,
                max_history=max_history,
                stream=stream,
                disable_history=no_history,
                resume_conversation_id=resume_conversation_id,
                resume_turns=resume_turns,
            )
        elif ui.lower() == "repl":
            from tsugite.ui.repl_chat import run_repl_chat

            # REPL defaults to streaming for better UX
            stream_mode = stream or True

            run_repl_chat(
                agent_path=primary_agent_path,
                model_override=model,
                max_history=max_history,
                stream=stream_mode,
                disable_history=no_history,
                resume_conversation_id=resume_conversation_id,
                resume_turns=resume_turns,
            )
        else:
            console.print(f"[red]Unknown UI mode: {ui}. Use 'repl' or 'tui'.[/red]")
            raise typer.Exit(1)


# Register subcommands from separate modules
# These imports are fast now because each module uses lazy loading internally
from .agents import agents_app  # noqa: E402
from .attachments import attachments_app  # noqa: E402
from .benchmark import benchmark_command  # noqa: E402
from .cache import cache_app  # noqa: E402
from .config import config_app  # noqa: E402
from .history import history_app  # noqa: E402
from .init import init  # noqa: E402
from .mcp import mcp_app  # noqa: E402
from .tools import tools_app  # noqa: E402
from .validate import validate_command  # noqa: E402

app.add_typer(mcp_app, name="mcp")
app.add_typer(agents_app, name="agents")
app.add_typer(config_app, name="config")
app.add_typer(attachments_app, name="attachments")
app.add_typer(cache_app, name="cache")
app.add_typer(tools_app, name="tools")
app.add_typer(history_app, name="history")
app.command("benchmark")(benchmark_command)
app.command("init")(init)
app.command("validate")(validate_command)
