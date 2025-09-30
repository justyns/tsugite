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


if __name__ == "__main__":
    app()
