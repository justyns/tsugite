"""CLI run command - execute agents."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from rich.console import Console

from tsugite.console import get_error_console
from tsugite.options import (
    AttachmentOptions,
    ExecutionOptions,
    HistoryOptions,
    UIOptions,
)

from .helpers import (
    PathContext,
    assemble_prompt_with_attachments,
    get_logo,
    inject_auto_context_if_enabled,
    load_and_validate_agent,
    print_plain_info,
    workspace_directory_context,
)


def _resolve_effective_workspace(workspace: Optional[str], no_workspace: bool) -> tuple[Optional[str], Optional[Any]]:
    """Determine and resolve which workspace to use (explicit > no-workspace > config default).

    Returns (workspace_ref, workspace_object) — either may be None.
    """
    from tsugite.config import load_config

    config = load_config()
    workspace_to_use = workspace
    if not workspace and not no_workspace and config.default_workspace:
        workspace_to_use = config.default_workspace
    return workspace_to_use, _resolve_workspace(workspace_to_use)


def _resolve_workspace(workspace_ref: Optional[str]) -> Optional[Any]:
    """Resolve workspace from name or path."""
    if not workspace_ref:
        return None

    from tsugite.workspace import Workspace, WorkspaceNotFoundError

    # Check if it's a path (contains "/" or starts with ".")
    if "/" in workspace_ref or workspace_ref.startswith(".") or workspace_ref.startswith("~"):
        workspace_path = Path(workspace_ref).expanduser().resolve()
        try:
            return Workspace.load(workspace_path)
        except WorkspaceNotFoundError:
            return None

    # Lookup by name
    try:
        return Workspace.load_by_name(workspace_ref)
    except WorkspaceNotFoundError:
        return None


def _build_workspace_attachments(workspace) -> List[str]:
    """Build list of workspace attachment paths (convention-based)."""
    from tsugite.workspace.context import build_workspace_attachments

    attachment_objects = build_workspace_attachments(workspace)
    return [str(att.source) for att in attachment_objects]


def _check_and_run_onboarding(
    workspace, workspace_ref: str, model: Optional[str], interactive: bool = True
) -> Optional[Any]:
    """Check if workspace needs onboarding and run it if user confirms."""
    if not workspace.needs_onboarding() or not interactive:
        return workspace

    from rich.prompt import Confirm

    if Confirm.ask("This looks like a new workspace. Run setup?", default=True):
        from tsugite.cli.workspace import onboard_workspace

        onboard_workspace(workspace.name, model=model)
        return _resolve_workspace(workspace_ref)

    return workspace


def _resolve_ui_mode(ui_mode: Optional[str], ui_opts: UIOptions, console: Console) -> UIOptions:
    """Resolve UI mode flag and return updated UIOptions."""
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
        from dataclasses import replace

        kwargs["exec_options"] = replace(exec_opts, return_token_usage=True)
    return kwargs


def _resolve_conversation_continuation(
    continue_conversation: bool, conversation_id: Optional[str], console: Console
) -> Optional[str]:
    """Resolve which conversation to continue."""
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
    console: Console,
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
        show_panels=False,
        show_debug_messages=ui_opts.verbose,
    ) as custom_logger:
        executor_kwargs["custom_logger"] = custom_logger
        return executor(**executor_kwargs)


def _unpack_execution_result(result):
    """Unpack execution result into a consistent tuple."""
    from tsugite.agent_runner.models import AgentExecutionResult

    if isinstance(result, AgentExecutionResult):
        return (
            result.response,
            result.token_count,
            result.cost,
            result.execution_steps,
            result.system_message,
            result.attachments,
            result.session_id,
        )

    return result, None, None, None, None, None, None


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
    attachment: Optional[List[str]] = typer.Option(
        None, "-f", "--attachment", help="Attachment(s) to include (repeatable)"
    ),
    refresh_cache: bool = typer.Option(False, "--refresh-cache", help="Force refresh cached attachment content"),
    auto_context: Optional[bool] = typer.Option(
        None,
        "--auto-context/--no-auto-context",
        help="Enable/disable auto-context attachments (overrides config/agent)",
    ),
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
    sandbox: bool = typer.Option(False, "--sandbox", help="Run agent code in bubblewrap sandbox"),
    no_sandbox: bool = typer.Option(False, "--no-sandbox", help="Disable sandbox (overrides config)"),
    allow_domain: Optional[List[str]] = typer.Option(
        None,
        "--allow-domain",
        help="Domain(s) allowed in sandbox, with optional port (e.g. github.com, *.example.com:8080). Default ports: 80, 443",
    ),
    no_network: bool = typer.Option(False, "--no-network", help="Sandbox with no network access at all"),
    no_secrets: bool = typer.Option(False, "--no-secrets", help="Skip secrets backend initialization"),
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
    from tsugite.agent_runner import get_agent_info, run_agent
    from tsugite.md_agents import validate_agent_execution
    from tsugite.secrets import init_cli as init_secrets
    from tsugite.utils import should_use_plain_output

    from . import console

    init_secrets(no_secrets)

    # Validate flag combinations
    if new_session and continue_conversation:
        console.print("[red]Error: Cannot use --new-session with --continue[/red]")
        raise typer.Exit(1)

    if workspace and no_workspace:
        console.print("[red]Error: Cannot use --workspace with --no-workspace[/red]")
        raise typer.Exit(1)

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

    workspace_to_use, resolved_workspace = _resolve_effective_workspace(workspace, no_workspace)

    if new_session and not workspace_to_use:
        console.print("[yellow]Warning: --new-session has no effect without a workspace[/yellow]")
    exec_opts = ExecutionOptions.from_cli(
        model=model,
        debug=debug,
        stream=stream,
        dry_run=dry_run,
        sandbox=sandbox,
        no_sandbox=no_sandbox,
        allow_domain=allow_domain,
        no_network=no_network,
    )
    history_opts = HistoryOptions.from_cli(
        no_history=no_history,
        continue_conversation=continue_conversation,
        conversation_id=conversation_id,
        history_dir=history_dir,
    )
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
        resolved_workspace = _check_and_run_onboarding(
            resolved_workspace, workspace_to_use, model, interactive=not ui_opts.non_interactive
        )
        workspace_attachments = _build_workspace_attachments(resolved_workspace)

    # Combine workspace attachments with CLI attachments
    all_attachments = workspace_attachments + (list(attachment) if attachment else [])

    attach_opts = AttachmentOptions(
        sources=all_attachments,
        refresh_cache=refresh_cache,
        auto_context=auto_context,
    )

    if history_opts.storage_dir:
        history_opts.storage_dir.mkdir(parents=True, exist_ok=True)

    if ui_opts.no_color:
        console.no_color = True

    # Resolve UI mode to update ui_opts
    ui_opts = _resolve_ui_mode(ui, ui_opts, console)

    # Handle subagent mode and daemon mode (both need os)
    import os

    if subagent_mode:
        if ui_opts.plain or ui_opts.headless:
            console.print("[red]Error: --subagent-mode cannot be combined with --plain or --headless[/red]")
            raise typer.Exit(1)

        ui_opts.non_interactive = True
        history_opts.enabled = False
        os.environ["TSUGITE_SUBAGENT_MODE"] = "1"

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
                meta = SessionStorage.load_meta_fast(session_file)
                if meta and meta.data.get("agent") == daemon_agent:
                    latest_conv_id = session_file.stem
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
        history_opts.continue_id = _resolve_conversation_continuation(True, None, console)
    elif history_opts.continue_id:
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
                    meta = SessionStorage.load_meta_fast(session_path)
                    agent_name = meta.data.get("agent") if meta else None
                    if not agent_name:
                        raise ValueError("agent name missing from session_start")
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
                allowed_agent_names = []
                for allowed_ref in agent_refs[1:]:
                    _, allowed_file, _ = load_and_validate_agent(allowed_ref, console)
                    allowed_agent = parse_agent_file(allowed_file)
                    allowed_agent_names.append(allowed_agent.config.name)

                set_allowed_agents(allowed_agent_names)
                console.print(f"[cyan]Allowed agents to spawn: {', '.join(allowed_agent_names)}[/cyan]")
            else:
                set_allowed_agents(None)

            _, agent_file, _ = load_and_validate_agent(primary_agent_ref, console)

        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        use_plain_output = ui_opts.plain or should_use_plain_output()

        from tsugite.console import get_stderr_console

        stderr_console = get_stderr_console(no_color=ui_opts.no_color)

        # Print deferred workspace status messages
        if not ui_opts.headless and not ui_opts.final_only:
            if workspace_to_use and not workspace:
                stderr_console.print(f"[dim]Using default workspace: {workspace_to_use}[/dim]")
            if workspace_session_continued:
                stderr_console.print(f"[dim]Continuing workspace session: {history_opts.continue_id[:8]}...[/dim]")

        # Set up event bus in context for attachment loading
        from tsugite.events import EventBus
        from tsugite.models import resolve_effective_model
        from tsugite.ui_context import set_ui_context

        event_bus = EventBus()
        set_ui_context(event_bus=event_bus)

        agent_info = get_agent_info(agent_file)
        instruction_label = "runtime + agent" if agent_info.get("instructions") else "runtime default"

        # Agent config attachments (USER.md, MEMORY.md, Jinja templates) are resolved
        # by AgentPreparer as workspace-relative files — not through CLI attachment resolution.
        # Only CLI-level attachments (-f flag) and auto-context are resolved here.
        cli_only_attachments = inject_auto_context_if_enabled(
            None,
            agent_info.get("auto_context"),
            cli_override=attach_opts.auto_context,
        )

        prompt, resolved_attachments = assemble_prompt_with_attachments(
            prompt=prompt,
            agent_attachments=cli_only_attachments,
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
                "Model": resolve_effective_model(exec_opts.model_override, agent_info.get("model_raw")) or "unknown",
                "Instructions": instruction_label,
                "Tools": ", ".join(agent_info.get("tools", [])),
            }

            agent_attachments = agent_info.get("attachments")
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

        def _save_history(
            result_str="",
            token_count=None,
            cost=None,
            execution_steps=None,
            system_prompt=None,
            attachments=None,
            status="success",
            error_message=None,
            session_id=None,
        ):
            if not history_opts.enabled:
                return
            try:
                from tsugite.agent_runner.history_integration import save_run_to_history

                # If the agent loop already created a session and recorded
                # events live, target that same session. Otherwise honor the
                # user's --continue or create a fresh one.
                target_session = session_id or history_opts.continue_id
                save_run_to_history(
                    agent_path=agent_file,
                    agent_name=agent_info["name"],
                    prompt=prompt,
                    result=result_str,
                    model=resolve_effective_model(exec_opts.model_override, agent_info.get("model_raw")) or "unknown",
                    token_count=token_count,
                    cost=cost,
                    execution_steps=execution_steps,
                    continue_conversation_id=target_session,
                    system_prompt=system_prompt,
                    attachments=attachments,
                    channel_metadata=daemon_metadata,
                    status=status,
                    error_message=error_message,
                )
            except Exception:
                pass

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
                console,
            )

            (
                result_str,
                token_count,
                cost,
                execution_steps,
                system_prompt,
                attachments,
                session_id,
            ) = _unpack_execution_result(result)

            _save_history(
                result_str,
                token_count,
                cost,
                execution_steps,
                system_prompt,
                attachments,
                session_id=session_id,
            )
            _display_result(result_str, ui_opts, stderr_console)

        except ValueError as e:
            _save_history(status="error", error_message=str(e))
            get_error_console(ui_opts.headless, console).print(f"[red]Configuration error: {e}[/red]")
            raise typer.Exit(1)
        except RuntimeError as e:
            _save_history(status="error", error_message=str(e))
            get_error_console(ui_opts.headless, console).print(f"[red]Execution error: {e}[/red]")
            raise typer.Exit(1)
        except KeyboardInterrupt:
            _save_history(status="interrupted")
            get_error_console(ui_opts.headless, console).print("\n[yellow]Agent execution interrupted by user[/yellow]")
            raise typer.Exit(130)
        except Exception as e:
            _save_history(status="error", error_message=str(e))
            err_console = get_error_console(ui_opts.headless, console)
            err_console.print(f"[red]Unexpected error: {e}[/red]")
            if not ui_opts.log_json:
                err_console.print("\n[dim]Use --log-json for machine-readable output[/dim]")
            raise typer.Exit(1)
