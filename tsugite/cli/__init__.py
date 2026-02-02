"""Tsugite CLI application - main entry point."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from rich.console import Console
from rich.panel import Panel
from rich.traceback import install

from tsugite.console import get_error_console
from tsugite.options import (
    AttachmentOptions,
    DockerOptions,
    ExecutionOptions,
    HistoryOptions,
    UIOptions,
)

from .helpers import (
    PathContext,
    assemble_prompt_with_attachments,
    change_to_root_directory,
    get_logo,
    inject_auto_context_if_enabled,
    load_and_validate_agent,
    print_plain_info,
    workspace_directory_context,
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


def _resolve_workspace(workspace_ref: Optional[str]) -> Optional[Any]:
    """Resolve workspace from name or path.

    Args:
        workspace_ref: Workspace name or path

    Returns:
        Workspace object or None
    """
    if not workspace_ref:
        return None

    from tsugite.workspace import Workspace, WorkspaceManager, WorkspaceNotFoundError

    # Check if it's a path (contains "/" or starts with ".")
    if "/" in workspace_ref or workspace_ref.startswith(".") or workspace_ref.startswith("~"):
        workspace_path = Path(workspace_ref).expanduser().resolve()
        try:
            return Workspace.load(workspace_path)
        except WorkspaceNotFoundError:
            return None

    # Lookup by name
    manager = WorkspaceManager()
    try:
        return manager.load_workspace(workspace_ref)
    except WorkspaceNotFoundError:
        return None


def _build_workspace_attachments(workspace) -> List[str]:
    """Build list of workspace attachment paths (convention-based).

    Args:
        workspace: Workspace object

    Returns:
        List of attachment file paths that exist
    """
    from tsugite.workspace.context import build_workspace_attachments

    attachment_objects = build_workspace_attachments(workspace)
    return [str(att.source) for att in attachment_objects]


def _build_docker_command(
    args: List[str],
    docker_opts: DockerOptions,
    exec_opts: ExecutionOptions,
    ui_opts: UIOptions,
    attach_opts: AttachmentOptions,
    history_opts: HistoryOptions,
    root: Optional[str],
    ui_mode: Optional[str],
) -> List[str]:
    """Build Docker wrapper command with all flags."""
    cmd = ["tsugite-docker"]

    if docker_opts.network != "host":
        cmd.extend(["--network", docker_opts.network])
    if docker_opts.keep:
        cmd.append("--keep")
    if docker_opts.container:
        cmd.extend(["--container", docker_opts.container])

    cmd.append("run")
    cmd.extend(args)

    if exec_opts.model_override:
        cmd.extend(["--model", exec_opts.model_override])
    if root:
        cmd.extend(["--root", str(root)])
    if history_opts.storage_dir:
        cmd.extend(["--history-dir", str(history_opts.storage_dir)])
    if ui_mode:
        cmd.extend(["--ui", ui_mode])
    if exec_opts.debug:
        cmd.append("--debug")
    if ui_opts.verbose:
        cmd.append("--verbose")
    if ui_opts.headless:
        cmd.append("--headless")
    if ui_opts.plain:
        cmd.append("--plain")
    if ui_opts.show_reasoning:
        cmd.append("--show-reasoning")
    if ui_opts.no_color:
        cmd.append("--no-color")
    if ui_opts.final_only:
        cmd.append("--final-only")
    if ui_opts.log_json:
        cmd.append("--log-json")
    if ui_opts.non_interactive:
        cmd.append("--non-interactive")
    if exec_opts.trust_mcp_code:
        cmd.append("--trust-mcp-code")
    if attach_opts.sources:
        for att in attach_opts.sources:
            cmd.extend(["--attachment", att])
    if attach_opts.refresh_cache:
        cmd.append("--refresh-cache")

    return cmd


def _resolve_ui_mode(ui_mode: Optional[str], ui_opts: UIOptions, console: Console) -> UIOptions:
    """Resolve UI mode flag and return updated UIOptions.

    Args:
        ui_mode: UI mode string (plain, headless)
        ui_opts: Current UI options
        console: Console for error output

    Returns:
        Updated UIOptions

    Raises:
        typer.Exit: If invalid UI mode or conflicting flags
    """
    if not ui_mode:
        return ui_opts

    if ui_opts.plain or ui_opts.headless:
        console.print("[red]Error: --ui cannot be used with --plain or --headless[/red]")
        raise typer.Exit(1)

    ui_modes = {"plain", "headless"}
    ui_lower = ui_mode.lower()
    if ui_lower not in ui_modes:
        console.print(f"[red]Error: Invalid UI mode '{ui_mode}'. Choose from: {', '.join(ui_modes)}[/red]")
        raise typer.Exit(1)

    if ui_lower == "plain":
        ui_opts.plain = True
    elif ui_lower == "headless":
        ui_opts.headless = True

    return ui_opts


def _build_executor_kwargs(
    agent_file: Path,
    prompt: str,
    exec_opts: ExecutionOptions,
    history_opts: HistoryOptions,
    resolved_attachments: List[Tuple[str, str]],
    executor: Any,
    path_context: Optional["PathContext"] = None,
) -> Dict[str, Any]:
    """Build executor kwargs dict for run_agent/run_multistep_agent."""
    from tsugite.agent_runner import run_agent

    kwargs = {
        "agent_path": agent_file,
        "prompt": prompt,
        "exec_options": exec_opts,
        "continue_conversation_id": history_opts.continue_id,
        "attachments": resolved_attachments,
        "path_context": path_context,
    }
    if history_opts.enabled and executor == run_agent:
        kwargs["exec_options"] = ExecutionOptions(
            model_override=exec_opts.model_override,
            debug=exec_opts.debug,
            stream=exec_opts.stream,
            trust_mcp_code=exec_opts.trust_mcp_code,
            dry_run=exec_opts.dry_run,
            return_token_usage=True,
        )
    return kwargs


def _handle_docker_execution(
    args: List[str],
    docker_opts: DockerOptions,
    exec_opts: ExecutionOptions,
    ui_opts: UIOptions,
    attach_opts: AttachmentOptions,
    history_opts: HistoryOptions,
    root: Optional[str],
    ui_mode: Optional[str],
) -> None:
    """Handle Docker container execution and exit."""
    import shutil
    import subprocess

    wrapper_path = shutil.which("tsugite-docker")
    if not wrapper_path:
        console.print("[red]Error: tsugite-docker wrapper not found in PATH[/red]")
        console.print("[yellow]Install it by adding tsugite/bin/ to your PATH[/yellow]")
        console.print("[dim]See bin/README.md for installation instructions[/dim]")
        raise typer.Exit(1)

    cmd = _build_docker_command(args, docker_opts, exec_opts, ui_opts, attach_opts, history_opts, root, ui_mode)
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

    from tsugite.agent_runner.history_integration import get_latest_conversation

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
    ui_opts: UIOptions,
    use_plain_output: bool,
):
    """Execute agent with appropriate UI mode."""
    from tsugite.ui import create_plain_logger, custom_agent_ui

    if ui_opts.headless or ui_opts.final_only:
        stderr_console = get_error_console(True, console)
        show_progress_items = ui_opts.verbose and not ui_opts.final_only

        with custom_agent_ui(
            console=stderr_console,
            show_code=show_progress_items,
            show_observations=show_progress_items,
            show_progress=False,
            show_llm_messages=show_progress_items,
            show_execution_results=show_progress_items,
            show_execution_logs=show_progress_items,
            show_panels=False,
            show_debug_messages=ui_opts.verbose,
        ) as custom_logger:
            executor_kwargs["custom_logger"] = custom_logger
            return executor(**executor_kwargs)

    if use_plain_output:
        custom_logger = create_plain_logger()
        with custom_logger.ui_handler.progress_context():
            executor_kwargs["custom_logger"] = custom_logger
            return executor(**executor_kwargs)

    default_console = Console(file=sys.stderr, force_terminal=True, no_color=ui_opts.no_color)
    with custom_agent_ui(
        console=default_console,
        show_code=not ui_opts.non_interactive,
        show_observations=not ui_opts.non_interactive,
        show_progress=not ui_opts.no_color,
        show_llm_messages=ui_opts.show_reasoning,
        show_execution_results=True,
        show_execution_logs=ui_opts.verbose,
        show_panels=False,
        show_debug_messages=ui_opts.verbose,
    ) as custom_logger:
        executor_kwargs["custom_logger"] = custom_logger
        return executor(**executor_kwargs)


def _unpack_execution_result(result, should_save_history: bool, executor):
    """Unpack execution result based on whether history was enabled."""
    from tsugite.agent_runner.models import AgentExecutionResult

    # Handle AgentExecutionResult object (when return_token_usage=True)
    if isinstance(result, AgentExecutionResult):
        return (
            result.response,
            result.token_count,
            result.cost,
            result.execution_steps,
            result.system_message,
            result.attachments,
        )

    # Handle plain string result
    return result, None, None, None, None, None


def _display_result(result_str: str, ui_opts: UIOptions, stderr_console: Console):
    """Display the final result to the user."""
    from rich.markdown import Markdown

    from tsugite.console import get_stdout_console

    if ui_opts.headless or ui_opts.final_only:
        get_stdout_console(no_color=ui_opts.no_color, force_terminal=True).print(Markdown(result_str))
    else:
        stderr_console.print()
        stderr_console.rule("[bold green]Agent Execution Complete[/bold green]")
        get_stdout_console(no_color=ui_opts.no_color, force_terminal=True).print(Markdown(result_str))


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
    workspace: Optional[str] = typer.Option(
        None, "--workspace", "-w", help="Workspace directory (auto-loads PERSONA.md, USER.md, MEMORY.md)"
    ),
    no_workspace: bool = typer.Option(False, "--no-workspace", help="Disable workspace (ignore default workspace)"),
    new_session: bool = typer.Option(False, "--new-session", help="Start fresh session (ignore workspace session)"),
    daemon_agent: Optional[str] = typer.Option(
        None, "--daemon", "-d", help="Join active daemon session for specified agent"
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
        tsu run --daemon odyn "follow up message"  # Join daemon session for agent 'odyn'
    """
    # Lazy imports - only load heavy dependencies when actually running agents
    from tsugite.agent_runner import get_agent_info, run_agent
    from tsugite.md_agents import validate_agent_execution
    from tsugite.utils import should_use_plain_output

    # Validate flag combinations
    if new_session and continue_conversation:
        console.print("[red]Error: Cannot use --new-session with --continue[/red]")
        raise typer.Exit(1)

    if workspace and no_workspace:
        console.print("[red]Error: Cannot use --workspace with --no-workspace[/red]")
        raise typer.Exit(1)

    # Build UI options first (needed for status messages)
    ui_opts = UIOptions(
        plain=plain,
        headless=headless,
        no_color=no_color,
        final_only=final_only,
        verbose=verbose,
        show_reasoning=show_reasoning,
        non_interactive=non_interactive,
        log_json=log_json,
    )

    # Load config to check for default workspace
    from tsugite.config import load_config

    config = load_config()

    # Determine which workspace to use: explicit > no-workspace > default from config
    workspace_to_use = workspace
    if not workspace and not no_workspace and config.default_workspace:
        workspace_to_use = config.default_workspace

    if new_session and not workspace_to_use:
        console.print("[yellow]Warning: --new-session has no effect without a workspace[/yellow]")
    exec_opts = ExecutionOptions(
        model_override=model,
        debug=debug,
        stream=stream,
        trust_mcp_code=trust_mcp_code,
        dry_run=dry_run,
    )
    history_opts = HistoryOptions(
        enabled=not no_history,
        continue_id=conversation_id if continue_conversation else None,
        storage_dir=Path(history_dir) if history_dir else None,
    )
    # Resolve workspace (name or path)
    resolved_workspace = _resolve_workspace(workspace_to_use)
    if workspace_to_use and not resolved_workspace:
        console.print(f"[yellow]Warning: Workspace '{workspace_to_use}' not found[/yellow]")

    # Auto-continue workspace session unless explicitly overridden
    workspace_session_continued = False
    if resolved_workspace and not continue_conversation and not new_session:
        from tsugite.workspace import WorkspaceSession

        session = WorkspaceSession(resolved_workspace)
        session_id = session.get_conversation_id()

        if not session_id:
            session_id = session.start_new()

        history_opts.continue_id = session_id
        workspace_session_continued = True

    # Add workspace files to attachment list
    workspace_attachments = []
    if resolved_workspace:
        workspace_attachments = _build_workspace_attachments(resolved_workspace)

    # Combine workspace attachments with CLI attachments
    all_attachments = workspace_attachments + (list(attachment) if attachment else [])

    attach_opts = AttachmentOptions(
        sources=all_attachments,
        refresh_cache=refresh_cache,
        auto_context=auto_context,
    )
    docker_opts = DockerOptions(
        enabled=docker,
        keep=keep,
        container=container,
        network=network,
    )

    if history_opts.storage_dir:
        history_opts.storage_dir.mkdir(parents=True, exist_ok=True)

    if ui_opts.no_color:
        console.no_color = True

    # Resolve UI mode to update ui_opts
    ui_opts = _resolve_ui_mode(ui, ui_opts, console)

    # Handle subagent mode - override incompatible settings
    if subagent_mode:
        import os

        if ui_opts.plain or ui_opts.headless:
            console.print("[red]Error: --subagent-mode cannot be combined with --plain or --headless[/red]")
            raise typer.Exit(1)

        ui_opts.non_interactive = True
        history_opts.enabled = False
        os.environ["TSUGITE_SUBAGENT_MODE"] = "1"

    if docker_opts.enabled or docker_opts.container:
        _handle_docker_execution(args, docker_opts, exec_opts, ui_opts, attach_opts, history_opts, root, ui)

    # Handle daemon mode
    daemon_metadata = None
    if daemon_agent:
        from tsugite.history import SessionStorage, get_history_dir, list_session_files

        try:
            from tsugite.daemon.config import load_daemon_config

            daemon_config = load_daemon_config()
            if daemon_agent not in daemon_config.agents:
                console.print(f"[red]Agent '{daemon_agent}' not found in daemon config[/red]")
                raise typer.Exit(1)

            agent_config = daemon_config.agents[daemon_agent]

        except ValueError as e:
            console.print(f"[red]Daemon config not found: {e}[/red]")
            console.print("[dim]Run 'tsugite daemon' to start daemon first[/dim]")
            raise typer.Exit(1)

        # Find latest session for this agent
        user_id = os.environ.get("USER", "cli-user")

        # Search for daemon-managed sessions for this agent
        latest_conv_id = None
        for session_file in list_session_files():
            try:
                storage = SessionStorage.load(session_file)
                if storage.agent == daemon_agent:
                    latest_conv_id = storage.session_id
                    break
            except Exception:
                continue

        if latest_conv_id:
            console.print(f"[cyan]Joining daemon session: {latest_conv_id}[/cyan]")
            history_opts.continue_id = latest_conv_id
        else:
            console.print(f"[yellow]No active daemon session found for '{daemon_agent}'[/yellow]")
            console.print("[dim]Creating new daemon-managed session...[/dim]")

        # Override agent with daemon agent
        args = [f"+{daemon_agent}"] + args

        # Build CLI metadata for history
        from datetime import datetime, timezone

        daemon_metadata = {
            "source": "cli",
            "channel_id": None,
            "user_id": user_id,
            "reply_to": "cli",
            "is_daemon_managed": True,
            "daemon_agent": daemon_agent,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Load workspace attachments from daemon config
        daemon_workspace = _resolve_workspace(str(agent_config.workspace_dir)) or agent_config.workspace_dir
        workspace_attachments = _build_workspace_attachments(daemon_workspace)
        all_attachments = workspace_attachments + (list(attachment) if attachment else [])
        attach_opts = AttachmentOptions(
            sources=all_attachments,
            refresh_cache=refresh_cache,
            auto_context=auto_context,
        )

    # Handle conversation continuation - check before parsing args
    if continue_conversation and not history_opts.continue_id:
        # User passed --continue without --conversation-id, resolve to latest
        history_opts.continue_id = _resolve_conversation_continuation(True, None)
    elif history_opts.continue_id:
        # User passed explicit conversation ID
        console.print(f"[cyan]Continuing conversation: {history_opts.continue_id}[/cyan]")

    # Parse CLI arguments into agents and prompt (allow empty agents when continuing)
    try:
        from tsugite.cli.helpers import parse_cli_arguments

        agent_refs, prompt, stdin_attachment = parse_cli_arguments(
            args, allow_empty_agents=continue_conversation, check_stdin=not continue_conversation
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    with workspace_directory_context(resolved_workspace, root, console) as path_context:
        try:
            base_dir = Path.cwd()

            if not agent_refs and continue_conversation:
                from tsugite.history import SessionStorage, get_history_dir

                session_path = get_history_dir() / f"{history_opts.continue_id}.jsonl"
                try:
                    storage = SessionStorage.load(session_path)
                    agent_name = storage.agent
                except Exception:
                    console.print(f"[red]Could not load metadata for conversation: {history_opts.continue_id}[/red]")
                    raise typer.Exit(1)

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

        use_plain_output = ui_opts.plain or should_use_plain_output()

        from tsugite.console import get_stderr_console

        stderr_console = get_stderr_console(no_color=ui_opts.no_color)

        # Print deferred workspace status messages
        if not ui_opts.headless and not ui_opts.final_only:
            if workspace_to_use and config.default_workspace == workspace_to_use and not workspace:
                stderr_console.print(f"[dim]Using default workspace: {workspace_to_use}[/dim]")
            if workspace_session_continued:
                stderr_console.print(f"[dim]Continuing workspace session: {history_opts.continue_id[:8]}...[/dim]")

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
            cli_override=attach_opts.auto_context,
        )

        prompt, resolved_attachments = assemble_prompt_with_attachments(
            prompt=prompt,
            agent_attachments=agent_attachments,
            cli_attachments=attach_opts.sources,
            base_dir=base_dir,
            refresh_cache=attach_opts.refresh_cache,
            console=console,
            stdin_attachment=stdin_attachment,
        )

        if not ui_opts.headless:
            if not use_plain_output:
                stderr_console.print(get_logo(stderr_console), style="cyan")
                stderr_console.print()

            info_items = {
                "Agent": agent_file.name,
                "Task": prompt,
                "Directory": str(Path.cwd()),
                "Model": exec_opts.model_override or agent_info.get("model", "unknown"),
                "Instructions": instruction_label,
                "Tools": ", ".join(agent_info.get("tools", [])),
            }

            if agent_attachments:
                info_items["Agent Attachments"] = ", ".join(agent_attachments)

            if attach_opts.sources:
                info_items["CLI Attachments"] = ", ".join(attach_opts.sources)

            if resolved_attachments:
                info_items["Attachments"] = f"{len(resolved_attachments)} file(s)"

            if use_plain_output:
                print_plain_info(stderr_console, "Tsugite Agent Runner", info_items, style="cyan")

        is_valid, error_msg = validate_agent_execution(agent_file)
        if not is_valid:
            get_error_console(ui_opts.headless, console).print(f"[red]Agent validation failed: {error_msg}[/red]")
            raise typer.Exit(1)

        from tsugite.agent_runner import preview_multistep_agent, run_multistep_agent
        from tsugite.md_agents import has_step_directives

        agent_text = agent_file.read_text()
        is_multistep = has_step_directives(agent_text)

        if exec_opts.dry_run:
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

        if not ui_opts.headless and not ui_opts.final_only:
            execution_type = "multi-step agent" if is_multistep else "agent"
            stderr_console.print()
            stderr_console.rule(f"[bold cyan]🚀 Starting {execution_type.title()} Execution[/bold cyan]")
            stderr_console.print()

        try:
            executor_kwargs = _build_executor_kwargs(
                agent_file,
                prompt,
                exec_opts,
                history_opts,
                resolved_attachments,
                executor,
                path_context,
            )

            result = _execute_agent_with_ui(
                executor,
                executor_kwargs,
                ui_opts,
                use_plain_output,
            )

            result_str, token_count, cost, execution_steps, system_prompt, attachments = _unpack_execution_result(
                result, history_opts.enabled, executor
            )

            if history_opts.enabled:
                try:
                    from tsugite.agent_runner.history_integration import save_run_to_history

                    agent_info = get_agent_info(agent_file)
                    save_run_to_history(
                        agent_path=agent_file,
                        agent_name=agent_info["name"],
                        prompt=prompt,
                        result=result_str,
                        model=exec_opts.model_override or agent_info.get("model", "default"),
                        token_count=token_count,
                        cost=cost,
                        execution_steps=execution_steps,
                        continue_conversation_id=history_opts.continue_id,
                        system_prompt=system_prompt,
                        attachments=attachments,
                        channel_metadata=daemon_metadata,
                    )
                except Exception:
                    pass

            _display_result(result_str, ui_opts, stderr_console)

        except ValueError as e:
            get_error_console(ui_opts.headless, console).print(f"[red]Configuration error: {e}[/red]")
            raise typer.Exit(1)
        except RuntimeError as e:
            get_error_console(ui_opts.headless, console).print(f"[red]Execution error: {e}[/red]")
            raise typer.Exit(1)
        except KeyboardInterrupt:
            get_error_console(ui_opts.headless, console).print("\n[yellow]Agent execution interrupted by user[/yellow]")
            raise typer.Exit(130)
        except Exception as e:
            err_console = get_error_console(ui_opts.headless, console)
            err_console.print(f"[red]Unexpected error: {e}[/red]")
            if not ui_opts.log_json:
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
        from tsugite.agent_runner.history_integration import get_latest_conversation

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
        from tsugite.history import SessionStorage, get_history_dir

        session_path = get_history_dir() / f"{continue_conversation_id}.jsonl"
        try:
            storage = SessionStorage.load(session_path)
            agent_name = storage.agent
        except Exception:
            console.print(f"[red]Could not load metadata for conversation: {continue_conversation_id}[/red]")
            raise typer.Exit(1)

        agent_path = f"+{agent_name}"
        console.print(f"[cyan]Auto-detected agent from conversation: {agent_name}[/cyan]")

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
                                preview += f"\n[dim]... ({len(lines) - 15} lines truncated, use --verbose to see all) ...[/dim]\n"
                                preview += "\n".join(lines[-5:])
                                console.print(preview)
                            else:
                                console.print(attachment.content)
                    elif attachment.source_url:
                        console.print(f"[dim][{attachment.content_type.value}: {attachment.source_url}][/dim]")
                    else:
                        console.print(f"[dim][{attachment.content_type.value} file: {attachment.mime_type}][/dim]")

                    console.print(f"[yellow]</Attachment: {attachment.name}>[/yellow]")

            # Display loaded skills (as additional content blocks in system message)
            if prepared.skills:
                next_idx = len(prepared.attachments) + 2 if prepared.attachments else 2
                for idx, skill in enumerate(prepared.skills, start=next_idx):
                    console.print()
                    console.rule(
                        f"[dim]Content Block {idx}: Skill - {skill.name}[/dim]",
                        style="dim",
                        align="left",
                    )
                    console.print(f"[magenta]<Skill: {skill.name}>[/magenta]")
                    # Truncate skill unless verbose
                    if verbose:
                        console.print(skill.content)
                    else:
                        lines = skill.content.split("\n")
                        if len(lines) > 20:
                            preview = "\n".join(lines[:10])
                            preview += (
                                f"\n[dim]... ({len(lines) - 15} lines truncated, use --verbose to see all) ...[/dim]\n"
                            )
                            preview += "\n".join(lines[-5:])
                            console.print(preview)
                        else:
                            console.print(skill.content)
                    console.print(f"[magenta]</Skill: {skill.name}>[/magenta]")

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
    workspace: Optional[str] = typer.Option(
        None, "--workspace", "-w", help="Workspace directory (auto-loads PERSONA.md, USER.md, MEMORY.md)"
    ),
    no_workspace: bool = typer.Option(False, "--no-workspace", help="Disable workspace (ignore default workspace)"),
):
    """Start an interactive chat session with an agent."""
    # Validate flag combinations
    if workspace and no_workspace:
        console.print("[red]Error: Cannot use --workspace with --no-workspace[/red]")
        raise typer.Exit(1)

    # Build option dataclasses
    exec_opts = ExecutionOptions(model_override=model, stream=stream)
    history_opts = HistoryOptions(enabled=not no_history, max_turns=max_history)

    # Load config to check for default workspace
    from tsugite.config import load_config

    config = load_config()

    # Determine which workspace to use: explicit > no-workspace > default from config
    workspace_to_use = workspace
    if not workspace and not no_workspace and config.default_workspace:
        workspace_to_use = config.default_workspace

    # Resolve workspace (name or path)
    resolved_workspace = _resolve_workspace(workspace_to_use)
    if workspace_to_use and not resolved_workspace:
        console.print(f"[yellow]Warning: Workspace '{workspace_to_use}' not found[/yellow]")

    # Build workspace attachments
    workspace_attachments = []
    if resolved_workspace:
        workspace_attachments = _build_workspace_attachments(resolved_workspace)

    with workspace_directory_context(resolved_workspace, root, console) as path_context:
        # Handle conversation resume
        resume_turns = None

        if continue_ is not None:
            from tsugite.agent_runner.history_integration import get_latest_conversation
            from tsugite.history import SessionStorage, get_history_dir, get_turns

            if continue_ == "" or continue_.lower() == "latest":
                history_opts.continue_id = get_latest_conversation()
                if not history_opts.continue_id:
                    console.print("[red]No conversations found to resume[/red]")
                    raise typer.Exit(1)
                console.print(f"[cyan]Resuming latest conversation: {history_opts.continue_id}[/cyan]")
            else:
                history_opts.continue_id = continue_
                console.print(f"[cyan]Resuming conversation: {history_opts.continue_id}[/cyan]")

            try:
                session_path = get_history_dir() / f"{history_opts.continue_id}.jsonl"
                resume_turns = get_turns(session_path)
                console.print(f"[cyan]Loaded {len(resume_turns)} previous turns[/cyan]")
            except FileNotFoundError:
                console.print(f"[red]Conversation not found: {history_opts.continue_id}[/red]")
                raise typer.Exit(1)
            except Exception as e:
                console.print(f"[red]Failed to load conversation: {e}[/red]")
                raise typer.Exit(1)

        agent_to_load = agent if agent else "default"
        _, primary_agent_path, _ = load_and_validate_agent(agent_to_load, console)

        if ui.lower() == "tui":
            from tsugite.ui.textual_chat import run_textual_chat

            run_textual_chat(
                agent_path=primary_agent_path,
                exec_options=exec_opts,
                history_options=history_opts,
                resume_turns=resume_turns,
                path_context=path_context,
                workspace_attachments=workspace_attachments,
            )
        elif ui.lower() == "repl":
            from tsugite.ui.repl_chat import run_repl_chat

            # REPL defaults to streaming for better UX
            exec_opts.stream = exec_opts.stream or True

            run_repl_chat(
                agent_path=primary_agent_path,
                exec_options=exec_opts,
                history_options=history_opts,
                resume_turns=resume_turns,
                path_context=path_context,
                workspace_attachments=workspace_attachments,
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
from .daemon import daemon_app  # noqa: E402
from .history import history_app  # noqa: E402
from .init import init  # noqa: E402
from .mcp import mcp_app  # noqa: E402
from .serve import serve_app  # noqa: E402
from .tools import tools_app  # noqa: E402
from .validate import validate_command  # noqa: E402
from .workspace import workspace_app  # noqa: E402

app.add_typer(mcp_app, name="mcp")
app.add_typer(serve_app, name="serve")
app.add_typer(agents_app, name="agents")
app.add_typer(config_app, name="config")
app.add_typer(daemon_app, name="daemon")
app.add_typer(attachments_app, name="attachments")
app.add_typer(cache_app, name="cache")
app.add_typer(tools_app, name="tools")
app.add_typer(history_app, name="history")
app.add_typer(workspace_app, name="workspace")
app.command("benchmark")(benchmark_command)
app.command("init")(init)
app.command("validate")(validate_command)
