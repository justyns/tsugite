"""Tsugite CLI application - main entry point."""

import os
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel

from tsugite.agent_runner import get_agent_info, run_agent, validate_agent_execution
from tsugite.animation import loading_animation
from tsugite.custom_ui import create_silent_logger, custom_agent_ui
from tsugite.utils import expand_file_references

from .agents import agents_app
from .attachments import attachments_app
from .benchmark import benchmark_command
from .cache import cache_app
from .config import config_app
from .helpers import agent_context, get_error_console, get_logo, get_output_console, parse_cli_arguments
from .mcp import mcp_app
from .tools import tools_app

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

    if history_dir:
        Path(history_dir).mkdir(parents=True, exist_ok=True)

    if no_color:
        console.no_color = True

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
        if debug:
            cmd.append("--debug")
        if verbose:
            cmd.append("--verbose")
        if headless:
            cmd.append("--headless")
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

    try:
        # Change to root directory first if specified
        original_cwd = None
        if root:
            root_path = Path(root)
            if not root_path.exists():
                console.print(f"[red]Working directory not found: {root}[/red]")
                raise typer.Exit(1)
            original_cwd = os.getcwd()
            os.chdir(str(root_path))

        base_dir = Path.cwd()

        primary_agent_path, delegation_agents = parse_agent_references(agent_refs, with_agents, base_dir)

        # Validate primary agent
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

    try:
        # Get agent info for display
        agent_info = get_agent_info(agent_file)
        instruction_label = "runtime + agent" if agent_info.get("instructions") else "runtime default"

        # Resolve agent attachments (from agent definition)
        agent_attachment_contents = []
        agent_attachments = agent_info.get("attachments", [])
        if agent_attachments:
            from tsugite.utils import resolve_attachments

            try:
                agent_attachment_contents = resolve_attachments(agent_attachments, base_dir, refresh_cache)
            except ValueError as e:
                console.print(f"[red]Agent attachment error: {e}[/red]")
                raise typer.Exit(1)

        # Resolve CLI attachments (-f flag)
        cli_attachment_contents = []
        if attachment:
            from tsugite.utils import resolve_attachments

            try:
                cli_attachment_contents = resolve_attachments(attachment, base_dir, refresh_cache)
            except ValueError as e:
                console.print(f"[red]Attachment error: {e}[/red]")
                raise typer.Exit(1)

        # Expand @filename references in prompt
        expanded_files = []
        try:
            prompt, expanded_files = expand_file_references(prompt, base_dir)
        except ValueError as e:
            console.print(f"[red]File reference error: {e}[/red]")
            raise typer.Exit(1)

        # Assemble prompt with proper order: agent attachments -> CLI attachments -> file refs -> prompt
        all_attachments = []
        if agent_attachment_contents:
            all_attachments.extend(agent_attachment_contents)
        if cli_attachment_contents:
            all_attachments.extend(cli_attachment_contents)

        if all_attachments:
            attachment_sections = [
                f"<Attachment: {name}>\n{content}\n</Attachment: {name}>" for name, content in all_attachments
            ]
            prompt = "\n\n".join(attachment_sections) + "\n\n" + prompt

        # Skip initial panel in headless mode
        if not headless:
            # Display ASCII logo (adaptive based on terminal width)
            console.print(get_logo(console), style="cyan")
            console.print()  # blank line for spacing

            # Build panel content
            panel_content = (
                f"[cyan]Agent:[/cyan] {agent_file.name}\n"
                f"[cyan]Task:[/cyan] {prompt}\n"
                f"[cyan]Directory:[/cyan] {Path.cwd()}\n"
                f"[cyan]Model:[/cyan] {model or agent_info.get('model', 'unknown')}\n"
                f"[cyan]Instructions:[/cyan] {instruction_label}\n"
                f"[cyan]Tools:[/cyan] {', '.join(agent_info.get('tools', []))}"
            )

            # Add agent attachments if any
            if agent_attachment_contents:
                agent_attachment_names = [name for name, _ in agent_attachment_contents]
                panel_content += f"\n[cyan]Agent Attachments:[/cyan] {', '.join(agent_attachment_names)}"

            # Add CLI attachments if any
            if cli_attachment_contents:
                cli_attachment_names = [name for name, _ in cli_attachment_contents]
                panel_content += f"\n[cyan]CLI Attachments:[/cyan] {', '.join(cli_attachment_names)}"

            # Add expanded files if any
            if expanded_files:
                panel_content += f"\n[cyan]Expanded Files:[/cyan] {', '.join(expanded_files)}"

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
        from tsugite.agent_runner import run_multistep_agent
        from tsugite.md_agents import has_step_directives

        agent_text = agent_file.read_text()
        is_multistep = has_step_directives(agent_text)

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
                    )
            elif native_ui:
                # Use minimal output with loading animation
                with loading_animation(
                    console=console, message="Waiting for LLM response", enabled=not non_interactive and not no_color
                ):
                    result = executor(
                        agent_path=agent_file,
                        prompt=prompt,
                        model_override=model,
                        debug=debug,
                        trust_mcp_code=trust_mcp_code,
                        delegation_agents=delegation_agents,
                    )
            else:
                # Default: Use custom UI
                with custom_agent_ui(
                    console=console,
                    show_code=not non_interactive,
                    show_observations=not non_interactive,
                    show_progress=not no_color,
                    show_llm_messages=show_reasoning,
                    show_execution_results=True,
                    show_execution_logs=verbose,
                ) as custom_logger:
                    result = executor(
                        agent_path=agent_file,
                        prompt=prompt,
                        model_override=model,
                        debug=debug,
                        custom_logger=custom_logger,
                        trust_mcp_code=trust_mcp_code,
                        delegation_agents=delegation_agents,
                    )

            # Display result
            if headless:
                # Headless: plain result to stdout
                get_output_console().print(result)
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

    finally:
        # Restore original working directory
        if original_cwd:
            os.chdir(original_cwd)


@app.command()
def render(
    agent_path: str = typer.Argument(help="Path to agent markdown file"),
    prompt: Optional[str] = typer.Argument(default="", help="Prompt/task for the agent (optional)"),
    root: Optional[str] = typer.Option(None, "--root", help="Working directory"),
    no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI colors"),
    attachment: Optional[List[str]] = typer.Option(
        None, "-f", "--attachment", help="Attachment(s) to include (repeatable)"
    ),
    refresh_cache: bool = typer.Option(False, "--refresh-cache", help="Force refresh cached attachment content"),
):
    """Render an agent template without executing it."""
    from tsugite.md_agents import parse_agent
    from tsugite.renderer import AgentRenderer
    from tsugite.utils import is_interactive

    if no_color:
        console.no_color = True

    with agent_context(agent_path, root, console) as agent_file:
        try:
            base_dir = Path.cwd()

            # Parse agent first to get its configuration
            agent_text = agent_file.read_text()
            agent = parse_agent(agent_text, agent_file)

            # Resolve agent attachments (from agent definition)
            agent_attachment_contents = []
            if agent.config.attachments:
                from tsugite.utils import resolve_attachments

                try:
                    agent_attachment_contents = resolve_attachments(agent.config.attachments, base_dir, refresh_cache)
                except ValueError as e:
                    console.print(f"[red]Agent attachment error: {e}[/red]")
                    raise typer.Exit(1)

            # Resolve CLI attachments (-f flag)
            cli_attachment_contents = []
            if attachment:
                from tsugite.utils import resolve_attachments

                try:
                    cli_attachment_contents = resolve_attachments(attachment, base_dir, refresh_cache)
                except ValueError as e:
                    console.print(f"[red]Attachment error: {e}[/red]")
                    raise typer.Exit(1)

            # Expand file references in prompt
            try:
                prompt_expanded, _ = expand_file_references(prompt, base_dir)
            except ValueError as e:
                console.print(f"[red]File reference error: {e}[/red]")
                raise typer.Exit(1)

            # Assemble prompt with proper order: agent attachments -> CLI attachments -> file refs -> prompt
            all_attachments = []
            if agent_attachment_contents:
                all_attachments.extend(agent_attachment_contents)
            if cli_attachment_contents:
                all_attachments.extend(cli_attachment_contents)

            if all_attachments:
                attachment_sections = [
                    f"<Attachment: {name}>\n{content}\n</Attachment: {name}>" for name, content in all_attachments
                ]
                prompt_expanded = "\n\n".join(attachment_sections) + "\n\n" + prompt_expanded

            # Execute prefetch tools if any
            from tsugite.agent_runner import execute_prefetch

            prefetch_context = {}
            if agent.config.prefetch:
                try:
                    prefetch_context = execute_prefetch(agent.config.prefetch)
                except Exception as e:
                    console.print(f"[yellow]Warning: Prefetch execution failed: {e}[/yellow]")

            # Prepare context
            context = {
                **prefetch_context,
                "user_prompt": prompt_expanded,
                "is_interactive": is_interactive(),
                "task_summary": "## Current Tasks\nNo tasks yet.",
            }

            # Render template
            renderer = AgentRenderer()
            rendered_content = renderer.render(agent.content, context)

            console.print(
                Panel(
                    f"[cyan]Agent:[/cyan] {agent_file.name}\n"
                    f"[cyan]Prompt:[/cyan] {prompt}\n"
                    f"[cyan]Directory:[/cyan] {Path.cwd()}",
                    title="Tsugite Template Renderer",
                    border_style="green",
                )
            )

            console.print("\n" + "=" * 50)
            console.print("[bold green]Rendered Template[/bold green]")
            console.print("=" * 50)
            console.print(rendered_content)

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
    root: Optional[str] = typer.Option(None, "--root", help="Working directory"),
):
    """Start an interactive chat session with an agent."""
    from tsugite.agent_composition import parse_agent_references
    from tsugite.chat_ui import run_chat_cli

    # Change to root directory if specified
    original_cwd = None
    if root:
        root_path = Path(root)
        if not root_path.exists():
            console.print(f"[red]Working directory not found: {root}[/red]")
            raise typer.Exit(1)
        original_cwd = os.getcwd()
        os.chdir(str(root_path))

    try:
        # Resolve agent path
        if agent:
            # Parse agent reference
            base_dir = Path.cwd()
            agent_refs = [agent]
            primary_agent_path, _ = parse_agent_references(agent_refs, None, base_dir)
        else:
            # Use default agent
            default_agent = Path.cwd() / ".tsugite" / "default.md"
            if not default_agent.exists():
                # Fallback to assistant
                default_agent = Path.cwd() / "agents" / "assistant.md"
                if not default_agent.exists():
                    console.print("[red]No default agent found[/red]")
                    console.print("Specify an agent: [cyan]tsugite chat +assistant[/cyan]")
                    raise typer.Exit(1)
            primary_agent_path = default_agent

        if not primary_agent_path.exists():
            console.print(f"[red]Agent file not found: {primary_agent_path}[/red]")
            raise typer.Exit(1)

        # Run chat
        run_chat_cli(
            agent_path=primary_agent_path,
            model_override=model,
            max_history=max_history,
        )

    finally:
        # Restore original directory
        if original_cwd:
            os.chdir(original_cwd)


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to"),
    port: int = typer.Option(8080, "--port", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload on code changes"),
):
    """Start the web UI server."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]Web UI dependencies not installed![/red]")
        console.print("\nInstall with: [cyan]uv add fastapi uvicorn python-multipart[/cyan]")
        raise typer.Exit(1)

    console.print("[cyan]Starting Tsugite Web UI...[/cyan]")
    console.print(f"[dim]Server: http://{host}:{port}[/dim]")
    console.print(f"[dim]Chat UI: http://{host}:{port}/chat[/dim]\n")

    try:
        uvicorn.run(
            "tsugite.web.server:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Server error: {e}[/red]")
        raise typer.Exit(1)


# Register subcommands from separate modules
app.add_typer(mcp_app, name="mcp")
app.add_typer(agents_app, name="agents")
app.add_typer(config_app, name="config")
app.add_typer(attachments_app, name="attachments")
app.add_typer(cache_app, name="cache")
app.add_typer(tools_app, name="tools")
app.command("benchmark")(benchmark_command)
