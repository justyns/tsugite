#!/usr/bin/env python3
"""Tsugite CLI application."""

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel

from tsugite.agent_runner import get_agent_info, run_agent, validate_agent_execution
from tsugite.animation import loading_animation
from tsugite.constants import TSUGITE_LOGO_NARROW, TSUGITE_LOGO_WIDE
from tsugite.custom_ui import create_silent_logger, custom_agent_ui
from tsugite.utils import expand_file_references

app = typer.Typer(
    name="tsugite",
    help="Micro-agent runner for task automation using markdown definitions",
    no_args_is_help=True,
)

console = Console()


def _get_logo(console: Console) -> str:
    return TSUGITE_LOGO_NARROW if console.width < 80 else TSUGITE_LOGO_WIDE


def _get_error_console(headless: bool, console: Console) -> Console:
    return Console(file=sys.stderr, no_color=True) if headless else console


def _get_output_console() -> Console:
    return Console(file=sys.stdout, no_color=True)


def parse_cli_arguments(args: List[str]) -> tuple[List[str], str]:
    """Parse CLI arguments into agent references and prompt.

    Args:
        args: List of positional arguments from CLI

    Returns:
        Tuple of (agent_refs, prompt)

    Examples:
        ["+a", "+b", "task"] -> (["+a", "+b"], "task")
        ["+a", "create", "ticket"] -> (["+a"], "create ticket")
        ["agent.md", "helper.md", "do", "work"] -> (["agent.md", "helper.md"], "do work")
    """
    if not args:
        raise ValueError("No arguments provided")

    agents = []
    prompt_parts = []

    for i, arg in enumerate(args):
        # Check if this looks like an agent reference
        # Exclude arguments containing @ (file references) from being treated as agents
        # Also exclude arguments with spaces unless they're file paths (contain /)
        has_file_reference = "@" in arg
        has_path_separator = "/" in arg
        has_spaces = " " in arg
        is_agent = (
            (arg.startswith("+") or arg.endswith(".md") or has_path_separator)
            and not has_file_reference
            and not (has_spaces and not has_path_separator)
        )

        if is_agent and not prompt_parts:
            # Still collecting agents
            agents.append(arg)
        else:
            # First non-agent arg or after we started collecting prompt
            prompt_parts.append(arg)

    if not agents:
        raise ValueError("No agent specified")

    prompt = " ".join(prompt_parts)
    return agents, prompt


@contextmanager
def _agent_context(agent_path: str, root: Optional[str]):
    """Validate agent path and optionally change working directory."""

    original_cwd = None

    try:
        if root:
            root_path = Path(root)
            if not root_path.exists():
                console.print(f"[red]Working directory not found: {root}[/red]")
                raise typer.Exit(1)
            original_cwd = os.getcwd()
            os.chdir(str(root_path))

        agent_file = Path(agent_path)
        if not agent_file.exists():
            console.print(f"[red]Agent file not found: {agent_path}[/red]")
            raise typer.Exit(1)

        if agent_file.suffix != ".md":
            console.print(f"[red]Agent file must be a .md file: {agent_path}[/red]")
            raise typer.Exit(1)

        yield agent_file.resolve()

    finally:
        if original_cwd:
            os.chdir(original_cwd)


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
    native_ui: bool = typer.Option(False, "--native-ui", help="Use native smolagents output instead of custom UI"),
    silent: bool = typer.Option(False, "--silent", help="Suppress all agent output"),
    show_reasoning: bool = typer.Option(False, "--show-reasoning", help="Show LLM reasoning messages"),
    verbose: bool = typer.Option(False, "--verbose", help="Show all execution details"),
    headless: bool = typer.Option(
        False, "--headless", help="Headless mode for CI/scripts: result to stdout, optional progress to stderr"
    ),
    trust_mcp_code: bool = typer.Option(False, "--trust-mcp-code", help="Trust remote code from MCP servers"),
    attachment: Optional[List[str]] = typer.Option(
        None, "-f", "--attachment", help="Attachment(s) to include (repeatable)"
    ),
    refresh_cache: bool = typer.Option(False, "--refresh-cache", help="Force refresh cached attachment content"),
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
            console.print(_get_logo(console), style="cyan")
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
            _get_error_console(headless, console).print(f"[red]Agent validation failed: {error_msg}[/red]")
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
                stderr_console = _get_error_console(True, console)

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
                # Use native smolagents output with loading animation
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
                _get_output_console().print(result)
            elif not silent:
                console.print("\n" + "=" * 50)
                console.print("[bold green]Agent Execution Complete[/bold green]")
                console.print("=" * 50)
                console.print(result)

        except ValueError as e:
            _get_error_console(headless, console).print(f"[red]Configuration error: {e}[/red]")
            raise typer.Exit(1)
        except RuntimeError as e:
            _get_error_console(headless, console).print(f"[red]Execution error: {e}[/red]")
            raise typer.Exit(1)
        except KeyboardInterrupt:
            _get_error_console(headless, console).print("\n[yellow]Agent execution interrupted by user[/yellow]")
            raise typer.Exit(130)
        except Exception as e:
            err_console = _get_error_console(headless, console)
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

    with _agent_context(agent_path, root) as agent_file:
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
    since: Optional[str] = typer.Option(None, "--since", help="Show history since date/time"),
):
    """View or manage execution history."""
    console.print(f"[yellow]History {action} not yet implemented[/yellow]")


@app.command("benchmark")
def benchmark_command(
    action: str = typer.Argument(help="Action: run, compare, report"),
    models: Optional[str] = typer.Option(None, "--models", help="Comma-separated list of models to test"),
    categories: Optional[str] = typer.Option(None, "--categories", help="Comma-separated list of categories to test"),
    agent_path: Optional[str] = typer.Option(None, "--agent", help="Path to specific agent file to test"),
    baseline: Optional[str] = typer.Option(None, "--baseline", help="Baseline model for comparison"),
    output: Optional[str] = typer.Option(None, "--output", help="Output file for report"),
    format: Optional[str] = typer.Option("markdown", "--format", help="Report format: json, markdown, html, csv"),
    test_filter: Optional[str] = typer.Option(None, "--filter", help="Filter tests by name/ID"),
    parallel: bool = typer.Option(True, "--parallel/--sequential", help="Run tests in parallel"),
    repeat: int = typer.Option(1, "--repeat", help="Number of times to repeat each test"),
):
    """Run benchmarks and generate reports."""
    import asyncio
    from pathlib import Path

    from .benchmark import BenchmarkConfig, BenchmarkRunner
    from .benchmark.reports import ReportGenerator

    if action == "run":
        # Parse models list
        model_list = []
        if models:
            model_list = [m.strip() for m in models.split(",")]
        else:
            console.print("[red]Error: --models is required for run action[/red]")
            raise typer.Exit(1)

        # Parse categories list
        category_list = None
        if categories:
            category_list = [c.strip() for c in categories.split(",")]

        # Create config
        config = BenchmarkConfig(
            models=model_list,
            categories=category_list or ["basic"],
            parallel=parallel,
            repeat_count=repeat,
            output_dir=Path("benchmark_results"),
        )

        console.print("[cyan]Running benchmarks...[/cyan]")
        console.print(f"Models: {', '.join(model_list)}")

        if agent_path:
            console.print(f"Agent: {agent_path}")
        else:
            console.print(f"Categories: {', '.join(config.categories)}")

        # Run benchmark
        runner = BenchmarkRunner(config)
        try:
            result = asyncio.run(
                runner.run_benchmark(
                    models=model_list,
                    categories=category_list,
                    test_filter=test_filter,
                    agent_path=Path(agent_path) if agent_path else None,
                )
            )

            # Generate default reports
            output_dir = Path("benchmark_results")
            output_dir.mkdir(exist_ok=True)

            timestamp = result.start_time.strftime("%Y%m%d_%H%M%S")

            # Generate reports
            report_gen = ReportGenerator(result)

            # JSON report (always generated for data)
            json_path = output_dir / f"benchmark_{timestamp}.json"
            report_gen.generate_json_report(json_path)
            console.print(f"[green]JSON report saved: {json_path}[/green]")

            # Main report in requested format
            if output:
                output_path = Path(output)
            else:
                output_path = output_dir / f"benchmark_{timestamp}.{format}"

            if format == "json":
                report_gen.generate_json_report(output_path)
            elif format == "markdown":
                report_gen.generate_markdown_report(output_path)
            elif format == "html":
                report_gen.generate_html_report(output_path)
            elif format == "csv":
                report_gen.generate_csv_summary(output_path)
            else:
                console.print(f"[red]Unknown format: {format}[/red]")
                raise typer.Exit(1)

            console.print(f"[green]Report saved: {output_path}[/green]")

            # Print summary
            console.print("\n" + "=" * 50)
            console.print("[bold green]Benchmark Complete[/bold green]")
            console.print("=" * 50)

            summary = result.summary
            console.print(f"Duration: {result.total_duration:.2f}s")
            console.print(f"Models: {len(result.model_performances)}")
            console.print(f"Tests: {summary.get('total_tests', 0)}")
            console.print(f"Average Accuracy: {summary.get('average_accuracy', 0):.1%}")

            if summary.get("best_model"):
                console.print(f"Best Model: {summary['best_model']}")

            # Print detailed per-model performance
            if len(result.model_performances) > 0:
                console.print("\n[bold]Model Performance:[/bold]")
                for model_name, perf in result.model_performances.items():
                    console.print(f"  [cyan]{model_name}[/cyan]:")
                    console.print(f"    Accuracy: {perf.accuracy:.1%} ({perf.passed_tests}/{perf.total_tests} passed)")
                    console.print(f"    Avg Duration: {perf.average_duration:.2f}s")
                    if perf.total_cost > 0:
                        console.print(f"    Total Cost: ${perf.total_cost:.4f}")

            if result.errors:
                console.print(f"\n[yellow]Errors encountered: {len(result.errors)}[/yellow]")

        except Exception as e:
            console.print(f"[red]Benchmark failed: {e}[/red]")
            raise typer.Exit(1)

    elif action == "compare":
        console.print("[yellow]Compare functionality not yet implemented[/yellow]")

    elif action == "report":
        console.print("[yellow]Report generation from existing data not yet implemented[/yellow]")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available actions: run, compare, report")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    from tsugite import __version__

    console.print(f"Tsugite version {__version__}")


# MCP subcommands
mcp_app = typer.Typer(help="Manage MCP server configurations")
app.add_typer(mcp_app, name="mcp")


@mcp_app.command("list")
def mcp_list():
    """List all configured MCP servers."""
    from tsugite.mcp_config import get_default_config_path, load_mcp_config

    config_path = get_default_config_path()
    servers = load_mcp_config()

    if not servers:
        console.print("[yellow]No MCP servers configured[/yellow]")
        console.print(f"\nConfig file will be created at: {config_path}")
        return

    console.print(f"[cyan]Found {len(servers)} MCP server(s)[/cyan] in [dim]{config_path}[/dim]\n")

    for name, config in servers.items():
        server_type = "stdio" if config.is_stdio() else "HTTP"
        console.print(f"[bold]{name}[/bold] ({server_type})")

        if config.is_stdio():
            console.print(f"  Command: {config.command}")
            if config.args:
                console.print(f"  Args: {' '.join(config.args)}")
            if config.env:
                env_keys = list(config.env.keys())
                console.print(f"  Env vars: {', '.join(env_keys)}")
        else:
            console.print(f"  URL: {config.url}")

        console.print()


@mcp_app.command("show")
def mcp_show(server_name: str = typer.Argument(help="Name of the MCP server to show")):
    """Show detailed configuration for a specific MCP server."""
    from tsugite.mcp_config import load_mcp_config

    servers = load_mcp_config()

    if server_name not in servers:
        console.print(f"[red]MCP server '{server_name}' not found[/red]")
        console.print(f"\nAvailable servers: {', '.join(servers.keys())}")
        raise typer.Exit(1)

    config = servers[server_name]
    server_type = "stdio" if config.is_stdio() else "HTTP"

    console.print(Panel(f"[cyan]Server:[/cyan] {server_name}\n[cyan]Type:[/cyan] {server_type}", border_style="blue"))

    if config.is_stdio():
        console.print(f"\n[bold]Command:[/bold] {config.command}")
        if config.args:
            console.print("[bold]Arguments:[/bold]")
            for arg in config.args:
                console.print(f"  - {arg}")
        if config.env:
            console.print("\n[bold]Environment Variables:[/bold]")
            for key, value in config.env.items():
                if "token" in key.lower() or "key" in key.lower() or "secret" in key.lower():
                    console.print(f"  {key}: [dim]<redacted>[/dim]")
                else:
                    console.print(f"  {key}: {value}")
    else:
        console.print(f"\n[bold]URL:[/bold] {config.url}")


@mcp_app.command("test")
def mcp_test(
    server_name: str = typer.Argument(help="Name of the MCP server to test"),
    trust_code: bool = typer.Option(False, "--trust-code", help="Trust remote code from this server"),
):
    """Test connection to an MCP server and list available tools."""
    from tsugite.mcp_config import load_mcp_config
    from tsugite.mcp_integration import load_mcp_tools

    servers = load_mcp_config()

    if server_name not in servers:
        console.print(f"[red]MCP server '{server_name}' not found[/red]")
        console.print(f"\nAvailable servers: {', '.join(servers.keys())}")
        raise typer.Exit(1)

    config = servers[server_name]

    console.print(f"[cyan]Testing connection to '{server_name}'...[/cyan]")

    try:
        tools = load_mcp_tools(server_name, config, allowed_tools=None, trust_remote_code=trust_code)

        console.print(f"[green]✓ Successfully connected to '{server_name}'[/green]")
        console.print(f"\n[bold]Available tools ({len(tools)}):[/bold]")

        for tool in tools:
            console.print(f"  - {tool.name}: {tool.description if hasattr(tool, 'description') else 'No description'}")

    except RuntimeError as e:
        console.print(f"[red]✗ Connection failed: {e}[/red]")
        raise typer.Exit(1)


@mcp_app.command("add")
def mcp_add(
    name: str = typer.Argument(help="Name for the MCP server"),
    url: Optional[str] = typer.Option(None, "--url", help="URL for HTTP server"),
    command: Optional[str] = typer.Option(None, "--command", help="Command for stdio server"),
    args: Optional[list[str]] = typer.Option(None, "--args", help="Argument for stdio server (repeatable)"),
    env: Optional[list[str]] = typer.Option(None, "--env", help="Environment variable as KEY=value (repeatable)"),
    server_type: Optional[str] = typer.Option(None, "--type", help="Server type: stdio or http"),
    force: bool = typer.Option(False, "--force", help="Overwrite if server already exists"),
):
    """Add a new MCP server to the configuration."""
    from tsugite.mcp_config import MCPServerConfig, add_server_to_config

    # Validate that either url or command is provided
    if not url and not command:
        console.print("[red]Error: Must specify either --url (for HTTP) or --command (for stdio)[/red]")
        raise typer.Exit(1)

    if url and command:
        console.print("[red]Error: Cannot specify both --url and --command[/red]")
        raise typer.Exit(1)

    # Parse environment variables
    env_dict = None
    if env:
        env_dict = {}
        for env_var in env:
            if "=" not in env_var:
                console.print(f"[red]Error: Invalid env var format '{env_var}'. Must be KEY=value[/red]")
                raise typer.Exit(1)
            key, value = env_var.split("=", 1)
            env_dict[key] = value

    # Create server config
    try:
        server = MCPServerConfig(
            name=name,
            url=url,
            command=command,
            args=args if args else None,
            env=env_dict,
            type=server_type,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    # Add to config
    try:
        from tsugite.mcp_config import get_config_path_for_write

        config_path = get_config_path_for_write()
        add_server_to_config(server, overwrite=force)

        action = "Updated" if force else "Added"
        server_type_name = "HTTP" if server.is_http() else "stdio"

        console.print(f"[green]✓ {action} MCP server '{name}' ({server_type_name})[/green]")

        # Show config summary
        if server.is_http():
            console.print(f"  URL: {server.url}")
        else:
            console.print(f"  Command: {server.command}")
            if server.args:
                console.print(f"  Args: {' '.join(server.args)}")
            if server.env:
                console.print(f"  Env vars: {', '.join(server.env.keys())}")

        console.print(f"\nServer saved to: {config_path}")

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to add server: {e}[/red]")
        raise typer.Exit(1)


# Config subcommands
agents_app = typer.Typer(help="Manage agents and agent inheritance")
app.add_typer(agents_app, name="agents")


@agents_app.command("list")
def agents_list(
    global_only: bool = typer.Option(False, "--global", help="List only global agents"),
    local_only: bool = typer.Option(False, "--local", help="List only local agents"),
):
    """List available agents."""
    from tsugite.agent_inheritance import get_global_agents_paths
    from tsugite.agent_utils import list_local_agents

    # Show global agents
    if global_only or not local_only:
        console.print("[cyan]Global Agents:[/cyan]\n")

        found_any = False
        for global_path in get_global_agents_paths():
            if not global_path.exists():
                continue

            agent_files = sorted(global_path.glob("*.md"))
            if agent_files:
                found_any = True
                console.print(f"[dim]{global_path}[/dim]")
                for agent_file in agent_files:
                    console.print(f"  • {agent_file.stem}")
                console.print()

        if not found_any:
            console.print("[yellow]No global agents found[/yellow]")
            console.print("\nGlobal agent locations:")
            for path in get_global_agents_paths():
                console.print(f"  {path}")
            console.print()

    # Show local agents
    if local_only or not global_only:
        console.print("[cyan]Local Agents:[/cyan]\n")

        local_agents = list_local_agents()

        if not local_agents:
            console.print("[yellow]No local agents found[/yellow]")
            console.print("\nLocal agent locations:")
            console.print("  • Current directory (*.md)")
            console.print("  • .tsugite/")
            console.print("  • ./agents/")
        else:
            for location, agent_files in local_agents.items():
                console.print(f"[dim]{location}[/dim]")
                for agent_file in agent_files:
                    console.print(f"  • {agent_file.stem}")
                console.print()


@agents_app.command("show")
def agents_show(
    agent_path: str = typer.Argument(help="Agent name or path to agent file"),
    show_inheritance: bool = typer.Option(False, "--inheritance", help="Show inheritance chain"),
):
    """Show agent information.

    Can be either a file path (e.g., 'examples/my_agent.md') or
    an agent name to search globally (e.g., 'default', 'builtin-default').
    """
    from tsugite.agent_inheritance import find_agent_file
    from tsugite.builtin_agents import get_builtin_default_agent, is_builtin_agent
    from tsugite.md_agents import parse_agent_file

    try:
        # Check if it's a built-in agent
        if is_builtin_agent(agent_path):
            agent = get_builtin_default_agent()
            config = agent.config
            console.print("[dim]Built-in agent[/dim]\n")
        else:
            agent_file = Path(agent_path)

            # If path doesn't exist, try to find it as an agent name
            if not agent_file.exists():
                found_path = find_agent_file(agent_path, Path.cwd())
                if found_path:
                    agent_file = found_path
                    console.print(f"[dim]Found: {agent_file}[/dim]\n")
                else:
                    console.print(f"[red]Agent not found: {agent_path}[/red]")
                    console.print("\nSearched in:")
                    console.print("  • Built-in agents")
                    console.print("  • Current directory")
                    console.print("  • .tsugite/")
                    console.print("  • ./agents/")
                    console.print("  • Global locations (use 'agents list --global' to see)")
                    raise typer.Exit(1)

            agent = parse_agent_file(agent_file)
            config = agent.config

        console.print(f"[cyan]Agent:[/cyan] [bold]{config.name}[/bold]\n")

        if config.description:
            console.print(f"[bold]Description:[/bold] {config.description}\n")

        if config.model:
            console.print(f"[bold]Model:[/bold] {config.model}")
        else:
            console.print("[bold]Model:[/bold] [dim](uses default)[/dim]")

        console.print(f"[bold]Max Steps:[/bold] {config.max_steps}")

        if config.tools:
            console.print(f"\n[bold]Tools ({len(config.tools)}):[/bold]")
            for tool in config.tools:
                console.print(f"  • {tool}")

        if config.extends:
            console.print(f"\n[bold]Extends:[/bold] {config.extends}")

        if show_inheritance:
            from tsugite.agent_utils import build_inheritance_chain

            try:
                chain = build_inheritance_chain(agent_file)
                console.print("\n[bold cyan]Inheritance Chain:[/bold cyan]")

                if len(chain) == 1:
                    console.print("  [dim](no inheritance)[/dim]")
                else:
                    for i, (agent_name, agent_path) in enumerate(chain):
                        is_current = i == len(chain) - 1
                        prefix = "  └─" if is_current else "  ├─"

                        if is_current:
                            console.print(f"{prefix} [bold]{agent_name}[/bold] [dim](current)[/dim]")
                        else:
                            console.print(f"{prefix} {agent_name} [dim]({agent_path.name})[/dim]")
                            if i < len(chain) - 1:
                                console.print("  │")
            except Exception as e:
                console.print(f"\n[yellow]Could not build inheritance chain: {e}[/yellow]")

        console.print()

    except Exception as e:
        console.print(f"[red]Failed to load agent: {e}[/red]")
        raise typer.Exit(1)


config_app = typer.Typer(help="Manage Tsugite configuration")
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show():
    """Show current configuration."""
    from tsugite.config import get_config_path, load_config

    config_path = get_config_path()
    config = load_config()

    console.print(f"[cyan]Configuration file:[/cyan] [dim]{config_path}[/dim]\n")

    if config.default_model:
        console.print(f"[bold]Default Model:[/bold] {config.default_model}\n")
    else:
        console.print("[yellow]No default model set[/yellow]\n")

    if config.default_base_agent is not None:
        console.print(f"[bold]Default Base Agent:[/bold] {config.default_base_agent}\n")
    else:
        console.print("[bold]Default Base Agent:[/bold] default (fallback)\n")

    if config.model_aliases:
        console.print(f"[bold]Model Aliases ({len(config.model_aliases)}):[/bold]")
        for alias, model in config.model_aliases.items():
            console.print(f"  {alias} → {model}")
    else:
        console.print("[dim]No model aliases defined[/dim]")


@config_app.command("set-default")
def config_set_default(
    model: str = typer.Argument(help="Model string (e.g., 'ollama:qwen2.5-coder:7b')"),
):
    """Set the default model."""
    from tsugite.config import get_config_path, set_default_model

    try:
        set_default_model(model)
        config_path = get_config_path()
        console.print(f"[green]✓ Default model set to:[/green] {model}")
        console.print(f"[dim]Saved to: {config_path}[/dim]")
    except Exception as e:
        console.print(f"[red]Failed to set default model: {e}[/red]")
        raise typer.Exit(1)


@config_app.command("alias")
def config_alias(
    name: str = typer.Argument(help="Alias name (e.g., 'cheap')"),
    model: str = typer.Argument(help="Model string (e.g., 'openai:gpt-4o-mini')"),
):
    """Create or update a model alias."""
    from tsugite.config import add_model_alias, get_config_path

    try:
        add_model_alias(name, model)
        config_path = get_config_path()
        console.print(f"[green]✓ Alias created:[/green] {name} → {model}")
        console.print(f"[dim]Saved to: {config_path}[/dim]")
    except Exception as e:
        console.print(f"[red]Failed to create alias: {e}[/red]")
        raise typer.Exit(1)


@config_app.command("alias-remove")
def config_alias_remove(
    name: str = typer.Argument(help="Alias name to remove"),
):
    """Remove a model alias."""
    from tsugite.config import get_config_path, remove_model_alias

    try:
        config_path = get_config_path()
        if remove_model_alias(name):
            console.print(f"[green]✓ Alias removed:[/green] {name}")
            console.print(f"[dim]Saved to: {config_path}[/dim]")
        else:
            console.print(f"[yellow]Alias '{name}' not found[/yellow]")
    except Exception as e:
        console.print(f"[red]Failed to remove alias: {e}[/red]")
        raise typer.Exit(1)


@config_app.command("list-aliases")
def config_list_aliases():
    """List all model aliases."""
    from tsugite.config import load_config

    config = load_config()

    if not config.model_aliases:
        console.print("[yellow]No model aliases defined[/yellow]")
        console.print("\nCreate an alias with: [cyan]tsugite config alias NAME MODEL[/cyan]")
        return

    console.print(f"[cyan]Model Aliases ({len(config.model_aliases)}):[/cyan]\n")
    for alias, model in config.model_aliases.items():
        console.print(f"  [bold]{alias}[/bold] → {model}")


@config_app.command("set-default-base")
def config_set_default_base(
    base_agent: str = typer.Argument(help="Base agent name (e.g., 'default') or 'none' to disable"),
):
    """Set the default base agent for inheritance."""
    from tsugite.config import get_config_path, set_default_base_agent

    try:
        # Handle "none" as None
        agent_value = None if base_agent.lower() == "none" else base_agent

        set_default_base_agent(agent_value)
        config_path = get_config_path()

        if agent_value is None:
            console.print("[green]✓ Default base agent disabled[/green]")
        else:
            console.print(f"[green]✓ Default base agent set to:[/green] {base_agent}")

        console.print(f"[dim]Saved to: {config_path}[/dim]")
    except Exception as e:
        console.print(f"[red]Failed to set default base agent: {e}[/red]")
        raise typer.Exit(1)


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


# Attachments subcommands
attachments_app = typer.Typer(help="Manage reusable text attachments")
app.add_typer(attachments_app, name="attachments")


@attachments_app.command("add")
def attachments_add(
    alias: str = typer.Argument(help="Unique name for the attachment"),
    source: str = typer.Argument(help="File path, URL, or '-' for stdin"),
):
    """Add or update an attachment.

    For stdin input ('-'), content is stored inline in attachments.json.
    For files and URLs, only the reference is stored (content fetched on demand).
    """
    from tsugite.attachments import add_attachment

    try:
        if source == "-":
            # Read from stdin - store inline
            import sys

            content = sys.stdin.read()
            add_attachment(alias, source="inline", content=content)

            console.print(f"[green]✓ Inline attachment '{alias}' saved[/green]")
            console.print("  Type: Inline text")
            console.print(f"  Size: {len(content)} characters")
        else:
            # File or URL reference - validate but don't fetch
            if source.startswith("http://") or source.startswith("https://"):
                # URL reference
                add_attachment(alias, source=source)
                console.print(f"[green]✓ URL attachment '{alias}' saved[/green]")
                console.print(f"  Source: {source}")
                console.print("  Type: URL (fetched on demand)")
            else:
                # File reference - validate it exists
                file_path = Path(source).expanduser()
                if not file_path.exists():
                    console.print(f"[red]File not found: {source}[/red]")
                    raise typer.Exit(1)

                # Store absolute path for reliability
                absolute_path = str(file_path.resolve())
                add_attachment(alias, source=absolute_path)

                console.print(f"[green]✓ File attachment '{alias}' saved[/green]")
                console.print(f"  Source: {absolute_path}")
                console.print("  Type: File (read on demand)")

    except Exception as e:
        console.print(f"[red]Failed to add attachment: {e}[/red]")
        raise typer.Exit(1)


@attachments_app.command("list")
def attachments_list():
    """List all attachments."""
    from rich.table import Table

    from tsugite.attachments import list_attachments

    attachments = list_attachments()

    if not attachments:
        console.print("[yellow]No attachments found[/yellow]")
        console.print("\nAdd an attachment with: [cyan]tsugite attachments add NAME SOURCE[/cyan]")
        return

    table = Table(title=f"Attachments ({len(attachments)} total)")
    table.add_column("Alias", style="cyan")
    table.add_column("Source", style="dim")
    table.add_column("Size", justify="right")
    table.add_column("Updated", style="dim")

    for alias, data in sorted(attachments.items()):
        size = len(data.get("content", ""))
        updated = data.get("updated", "unknown")[:10]  # Just date part
        table.add_row(alias, data.get("source", "unknown"), f"{size:,}", updated)

    console.print(table)


@attachments_app.command("show")
def attachments_show(
    alias: str = typer.Argument(help="Attachment alias to show"),
    content: bool = typer.Option(False, "--content", help="Show full content"),
):
    """Show details of an attachment."""
    from rich.panel import Panel

    from tsugite.attachments import get_attachment
    from tsugite.cache import get_cache_key, list_cache

    result = get_attachment(alias)

    if result is None:
        console.print(f"[red]Attachment '{alias}' not found[/red]")
        raise typer.Exit(1)

    source, stored_content = result

    # Determine attachment type
    is_inline = source.lower() in ("inline", "text")

    # Build panel content
    panel_content = f"[cyan]Alias:[/cyan] {alias}\n"
    panel_content += f"[cyan]Type:[/cyan] {'Inline' if is_inline else 'Reference'}\n"
    panel_content += f"[cyan]Source:[/cyan] {source}\n"

    # Check cache status for references
    if not is_inline:
        cache_entries = list_cache()
        cache_key = get_cache_key(source)
        if cache_key in cache_entries:
            cache_info = cache_entries[cache_key]
            panel_content += f"[cyan]Cached:[/cyan] Yes (size: {cache_info['size']:,} bytes)\n"
            panel_content += f"[cyan]Cached at:[/cyan] {cache_info['cached_at']}\n"
        else:
            panel_content += "[cyan]Cached:[/cyan] No\n"

    # Show content or preview
    if is_inline and stored_content:
        panel_content += f"[cyan]Size:[/cyan] {len(stored_content):,} characters\n"
        if content:
            panel_content += f"\n[cyan]Content:[/cyan]\n{stored_content}"
        else:
            preview = stored_content[:200]
            if len(stored_content) > 200:
                preview += "..."
            panel_content += f"\n[cyan]Preview:[/cyan]\n{preview}"
            panel_content += "\n\n[dim]Use --content to show full content[/dim]"
    elif not is_inline and content:
        # For references, fetch and show content if requested
        from tsugite.utils import resolve_attachments

        resolved = resolve_attachments([alias], Path.cwd())
        if resolved:
            _, resolved_content = resolved[0]
            panel_content += f"[cyan]Size:[/cyan] {len(resolved_content):,} characters\n"
            panel_content += f"\n[cyan]Content:[/cyan]\n{resolved_content}"
    elif not is_inline:
        panel_content += "\n[dim]Use --content to fetch and show full content[/dim]"

    console.print(Panel(panel_content, title=f"Attachment: {alias}", border_style="blue"))


@attachments_app.command("remove")
def attachments_remove(
    alias: str = typer.Argument(help="Attachment alias to remove"),
):
    """Remove an attachment."""
    from tsugite.attachments import remove_attachment

    if remove_attachment(alias):
        console.print(f"[green]✓ Attachment '{alias}' removed[/green]")
    else:
        console.print(f"[yellow]Attachment '{alias}' not found[/yellow]")


@attachments_app.command("search")
def attachments_search(
    query: str = typer.Argument(help="Search term"),
):
    """Search attachments by alias or source."""
    from rich.table import Table

    from tsugite.attachments import search_attachments

    results = search_attachments(query)

    if not results:
        console.print(f"[yellow]No attachments found matching '{query}'[/yellow]")
        return

    table = Table(title=f"Search Results for '{query}' ({len(results)} found)")
    table.add_column("Alias", style="cyan")
    table.add_column("Source", style="dim")
    table.add_column("Size", justify="right")

    for alias, data in sorted(results.items()):
        size = len(data.get("content", ""))
        table.add_row(alias, data.get("source", "unknown"), f"{size:,}")

    console.print(table)


# Cache subcommands
cache_app = typer.Typer(help="Manage attachment cache")
app.add_typer(cache_app, name="cache")


@cache_app.command("clear")
def cache_clear(
    alias: Optional[str] = typer.Argument(None, help="Attachment alias to clear cache for (or all if omitted)"),
):
    """Clear cache for an attachment or entire cache."""
    from tsugite.attachments import get_attachment
    from tsugite.cache import clear_cache

    try:
        if alias:
            # Get attachment source
            result = get_attachment(alias)
            if result is None:
                console.print(f"[red]Attachment '{alias}' not found[/red]")
                raise typer.Exit(1)

            source, content = result
            if content is not None:
                console.print(f"[yellow]Attachment '{alias}' is inline (no cache)[/yellow]")
                return

            # Clear cache for this source
            count = clear_cache(source)
            if count > 0:
                console.print(f"[green]✓ Cache cleared for '{alias}'[/green]")
            else:
                console.print(f"[yellow]No cache found for '{alias}'[/yellow]")
        else:
            # Clear all cache
            count = clear_cache()
            console.print(f"[green]✓ Cleared {count} cache entries[/green]")

    except Exception as e:
        console.print(f"[red]Failed to clear cache: {e}[/red]")
        raise typer.Exit(1)


@cache_app.command("list")
def cache_list():
    """List all cached attachments."""
    from rich.table import Table

    from tsugite.cache import list_cache

    cache_entries = list_cache()

    if not cache_entries:
        console.print("[yellow]No cached entries found[/yellow]")
        return

    table = Table(title=f"Cached Attachments ({len(cache_entries)} total)")
    table.add_column("Source", style="cyan")
    table.add_column("Cached At", style="dim")
    table.add_column("Size", justify="right")

    for key, metadata in sorted(cache_entries.items(), key=lambda x: x[1].get("cached_at", "")):
        source = metadata.get("source", "unknown")
        cached_at = metadata.get("cached_at", "unknown")[:19]  # Just date + time
        size = metadata.get("size", 0)
        table.add_row(source, cached_at, f"{size:,} bytes")

    console.print(table)


@cache_app.command("info")
def cache_info(
    alias: str = typer.Argument(help="Attachment alias"),
):
    """Show cache info for an attachment."""
    from rich.panel import Panel

    from tsugite.attachments import get_attachment
    from tsugite.cache import get_cache_info

    result = get_attachment(alias)
    if result is None:
        console.print(f"[red]Attachment '{alias}' not found[/red]")
        raise typer.Exit(1)

    source, content = result

    if content is not None:
        console.print(f"[yellow]Attachment '{alias}' is inline (no cache)[/yellow]")
        return

    cache_metadata = get_cache_info(source)

    if cache_metadata:
        cached_at = cache_metadata.get("cached_at", "unknown")
        size = cache_metadata.get("size", 0)

        panel_content = (
            f"[cyan]Source:[/cyan] {source}\n"
            f"[cyan]Cached:[/cyan] Yes\n"
            f"[cyan]Cached At:[/cyan] {cached_at}\n"
            f"[cyan]Size:[/cyan] {size:,} bytes"
        )
        console.print(Panel(panel_content, title=f"Cache Info: {alias}", border_style="green"))
    else:
        panel_content = f"[cyan]Source:[/cyan] {source}\n[yellow]Cached:[/yellow] No (will be fetched on first use)"
        console.print(Panel(panel_content, title=f"Cache Info: {alias}", border_style="yellow"))


if __name__ == "__main__":
    app()
