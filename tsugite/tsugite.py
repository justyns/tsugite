#!/usr/bin/env python3
"""Tsugite CLI application."""

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from tsugite.agent_runner import get_agent_info, run_agent, validate_agent_execution
from tsugite.animation import loading_animation
from tsugite.custom_ui import create_silent_logger, custom_agent_ui

app = typer.Typer(
    name="tsugite",
    help="Micro-agent runner for task automation using markdown definitions",
    no_args_is_help=True,
)

console = Console()


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
    agent_path: str = typer.Argument(help="Path to agent markdown file"),
    prompt: str = typer.Argument(default="", help="Prompt/task for the agent (optional)"),
    root: Optional[str] = typer.Option(None, "--root", help="Working directory"),
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
):
    """Run an agent with the given prompt."""

    if history_dir:
        Path(history_dir).mkdir(parents=True, exist_ok=True)

    if no_color:
        console.no_color = True

    with _agent_context(agent_path, root) as agent_file:
        # Get agent info for display
        agent_info = get_agent_info(agent_file)
        instruction_label = "runtime + agent" if agent_info.get("instructions") else "runtime default"

        # Skip initial panel in headless mode
        if not headless:
            console.print(
                Panel(
                    f"[cyan]Agent:[/cyan] {agent_file.name}\n"
                    f"[cyan]Task:[/cyan] {prompt}\n"
                    f"[cyan]Directory:[/cyan] {Path.cwd()}\n"
                    f"[cyan]Model:[/cyan] {agent_info.get('model', 'unknown')}\n"
                    f"[cyan]Instructions:[/cyan] {instruction_label}\n"
                    f"[cyan]Tools:[/cyan] {', '.join(agent_info.get('tools', []))}",
                    title="Tsugite Agent Runner",
                    border_style="blue",
                )
            )

        # Validate agent before execution
        is_valid, error_msg = validate_agent_execution(agent_file)
        if not is_valid:
            error_console = Console(file=sys.stderr, no_color=True) if headless else console
            error_console.print(f"[red]Agent validation failed: {error_msg}[/red]")
            raise typer.Exit(1)

        # Skip "Starting agent execution" in headless mode
        if not headless:
            console.print("[green]Starting agent execution...[/green]")

        try:
            # Choose execution mode based on flags
            if silent:
                # Completely silent execution
                result = run_agent(
                    agent_path=agent_file,
                    prompt=prompt,
                    model_override=model,
                    debug=debug,
                    custom_logger=create_silent_logger(),
                    trust_mcp_code=trust_mcp_code,
                )
            elif headless:
                # Headless mode: stderr for progress (if verbose), stdout for result
                stderr_console = Console(file=sys.stderr, no_color=True)
                stdout_console = Console(file=sys.stdout, no_color=True)

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
                    result = run_agent(
                        agent_path=agent_file,
                        prompt=prompt,
                        model_override=model,
                        debug=debug,
                        custom_logger=custom_logger,
                        trust_mcp_code=trust_mcp_code,
                    )
            elif native_ui:
                # Use native smolagents output with loading animation
                with loading_animation(
                    console=console, message="Waiting for LLM response", enabled=not non_interactive and not no_color
                ):
                    result = run_agent(
                        agent_path=agent_file,
                        prompt=prompt,
                        model_override=model,
                        debug=debug,
                        trust_mcp_code=trust_mcp_code,
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
                    result = run_agent(
                        agent_path=agent_file,
                        prompt=prompt,
                        model_override=model,
                        debug=debug,
                        custom_logger=custom_logger,
                        trust_mcp_code=trust_mcp_code,
                    )

            # Display result
            if headless:
                # Headless: plain result to stdout
                stdout_console = Console(file=sys.stdout, no_color=True)
                stdout_console.print(result)
            elif not silent:
                console.print("\n" + "=" * 50)
                console.print("[bold green]Agent Execution Complete[/bold green]")
                console.print("=" * 50)
                console.print(result)

        except ValueError as e:
            error_console = Console(file=sys.stderr, no_color=True) if headless else console
            error_console.print(f"[red]Configuration error: {e}[/red]")
            raise typer.Exit(1)
        except RuntimeError as e:
            error_console = Console(file=sys.stderr, no_color=True) if headless else console
            error_console.print(f"[red]Execution error: {e}[/red]")
            raise typer.Exit(1)
        except KeyboardInterrupt:
            error_console = Console(file=sys.stderr, no_color=True) if headless else console
            error_console.print("\n[yellow]Agent execution interrupted by user[/yellow]")
            raise typer.Exit(130)
        except Exception as e:
            error_console = Console(file=sys.stderr, no_color=True) if headless else console
            error_console.print(f"[red]Unexpected error: {e}[/red]")
            if not log_json:
                error_console.print("\n[dim]Use --log-json for machine-readable output[/dim]")
            raise typer.Exit(1)


@app.command()
def render(
    agent_path: str = typer.Argument(help="Path to agent markdown file"),
    prompt: Optional[str] = typer.Argument(default="", help="Prompt/task for the agent (optional)"),
    root: Optional[str] = typer.Option(None, "--root", help="Working directory"),
    no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI colors"),
):
    """Render an agent template without executing it."""
    from tsugite.md_agents import parse_agent
    from tsugite.renderer import AgentRenderer

    if no_color:
        console.no_color = True

    with _agent_context(agent_path, root) as agent_file:
        try:
            # Parse agent
            agent_text = agent_file.read_text()
            agent = parse_agent(agent_text, agent_file)

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
                "user_prompt": prompt,
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
    from tsugite.mcp_config import load_mcp_config

    servers = load_mcp_config()

    if not servers:
        console.print("[yellow]No MCP servers configured[/yellow]")
        console.print(f"Create a config file at: {Path.home() / '.tsugite' / 'mcp.json'}")
        return

    console.print(f"[cyan]Found {len(servers)} MCP server(s):[/cyan]\n")

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
            console.print(f"[bold]Arguments:[/bold]")
            for arg in config.args:
                console.print(f"  - {arg}")
        if config.env:
            console.print(f"\n[bold]Environment Variables:[/bold]")
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

        console.print(f"\nServer added to: {Path.home() / '.tsugite' / 'mcp.json'}")

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to add server: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
