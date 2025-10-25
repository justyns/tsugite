"""Benchmark CLI commands."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def benchmark_command(
    action: str = typer.Argument(help="Action: run, view, list"),
    models: Optional[str] = typer.Option(None, "--models", help="Comma-separated list of models to test"),
    categories: Optional[str] = typer.Option(None, "--categories", help="Comma-separated list of categories to test"),
    agent_path: Optional[str] = typer.Option(None, "--agent", help="Path to specific agent file to test"),
    output: Optional[str] = typer.Option(None, "--output", help="Output file for report"),
    format: Optional[str] = typer.Option("markdown", "--format", help="Report format: json, markdown, html, csv"),
    test_filter: Optional[str] = typer.Option(None, "--filter", help="Filter tests by name/ID"),
    parallel: bool = typer.Option(True, "--parallel/--sequential", help="Run tests in parallel"),
    repeat: int = typer.Option(1, "--repeat", help="Number of times to repeat each test"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed test outputs and case breakdowns"),
):
    """Run benchmarks and generate reports."""
    if action == "run":
        run_benchmark_action(
            console=console,
            models=models,
            categories=categories,
            agent_path=agent_path,
            output=output,
            format=format,
            test_filter=test_filter,
            parallel=parallel,
            repeat=repeat,
        )
    elif action == "view":
        view_benchmark_action(console=console, output=output, verbose=verbose)
    elif action == "list":
        list_benchmark_action(console=console)
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available actions: run, view, list")
        raise typer.Exit(1)


def parse_benchmark_run_args(
    models: Optional[str], categories: Optional[str], console: Console
) -> tuple[List[str], List[str]]:
    """Parse and validate models and categories arguments.

    Args:
        models: Comma-separated list of models
        categories: Comma-separated list of categories (optional)
        console: Rich console for error output

    Returns:
        Tuple of (model_list, category_list)

    Raises:
        typer.Exit: If models argument is missing
    """
    import typer

    # Parse models list (required)
    if not models:
        console.print("[red]Error: --models is required for run action[/red]")
        raise typer.Exit(1)

    model_list = [m.strip() for m in models.split(",")]

    # Parse categories list (optional, defaults to ["basic"])
    category_list = ["basic"]
    if categories:
        category_list = [c.strip() for c in categories.split(",")]

    return model_list, category_list


def print_benchmark_summary(
    console: Console,
    duration: float,
    model_count: int,
    total_tests: int,
    avg_accuracy: float,
    best_model: Optional[str] = None,
) -> None:
    """Print benchmark summary section.

    Args:
        console: Rich console for output
        duration: Total duration in seconds
        model_count: Number of models tested
        total_tests: Total number of tests
        avg_accuracy: Average accuracy (0.0-1.0)
        best_model: Name of best performing model (optional)
    """
    console.print("\n" + "=" * 50)
    console.print("[bold green]Benchmark Complete[/bold green]")
    console.print("=" * 50)
    console.print(f"Duration: {duration:.2f}s")
    console.print(f"Models: {model_count}")
    console.print(f"Tests: {total_tests}")
    console.print(f"Average Accuracy: {avg_accuracy:.1%}")
    if best_model:
        console.print(f"Best Model: {best_model}")


def print_model_performance(console: Console, model_performances: Dict[str, Any]) -> None:
    """Print per-model performance metrics.

    Args:
        console: Rich console for output
        model_performances: Dict mapping model names to performance data
    """
    if not model_performances:
        return

    console.print("\n[bold]Model Performance:[/bold]")
    for model_name, perf in model_performances.items():
        # Handle both dict (from JSON) and object (from result) formats
        if isinstance(perf, dict):
            accuracy = perf["accuracy"]
            passed = perf["passed_tests"]
            total = perf["total_tests"]
            avg_dur = perf["average_duration"]
            avg_steps = perf.get("average_steps", 0)
            total_cost = perf.get("total_cost", 0)
        else:
            accuracy = perf.accuracy
            passed = perf.passed_tests
            total = perf.total_tests
            avg_dur = perf.average_duration
            avg_steps = perf.average_steps
            total_cost = perf.total_cost

        console.print(f"  [cyan]{model_name}[/cyan]:")
        console.print(f"    Accuracy: {accuracy:.1%} ({passed}/{total} passed)")
        console.print(f"    Avg Duration: {avg_dur:.2f}s")
        console.print(f"    Avg Steps: {avg_steps:.1f}")
        if total_cost > 0:
            console.print(f"    Total Cost: ${total_cost:.4f}")


def print_test_results_table(
    console: Console,
    model_name: str,
    test_results: Dict[str, Any],
) -> None:
    """Print test results table for a single model.

    Args:
        console: Rich console for output
        model_name: Name of the model
        test_results: Dict mapping test IDs to test result data
    """
    console.print(f"\n[bold]Test Results - {model_name}:[/bold]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Test ID", style="dim", width=30)
    table.add_column("Category", width=12)
    table.add_column("Status", justify="center", width=10)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Duration", justify="right", width=10)
    table.add_column("Steps", justify="right", width=8)
    table.add_column("Cost", justify="right", width=10)

    test_items = list(test_results.items())

    # Show all tests if <=20, otherwise show first 15 and last 5
    if len(test_items) <= 20:
        display_tests = test_items
        truncated = False
    else:
        display_tests = test_items[:15] + test_items[-5:]
        truncated = True
        num_hidden = len(test_items) - 20

    for i, (test_id, test_result) in enumerate(display_tests):
        # Add separator row if we truncated
        if truncated and i == 15:
            table.add_row(f"... {num_hidden} more tests ...", "", "", "", "", "", "", style="dim italic")

        # Handle both dict (from JSON) and object (from result) formats
        if isinstance(test_result, dict):
            category = test_result.get("category", "unknown")
            passed = test_result.get("passed", False)
            score = test_result.get("score", 0)
            duration = test_result.get("duration", 0)
            steps = test_result.get("steps_taken", 0)
            cost_val = test_result.get("cost", 0)
        else:
            category = test_result.category
            passed = test_result.passed
            score = test_result.score
            duration = test_result.duration
            steps = test_result.steps_taken
            cost_val = test_result.cost

        status = "✅ PASS" if passed else "❌ FAIL"
        status_style = "green" if passed else "red"
        score_str = f"{score:.2f}"
        duration_str = f"{duration:.2f}s"
        steps_str = str(steps)
        cost_str = f"${cost_val:.4f}" if cost_val > 0 else "$0.00"

        table.add_row(
            test_id,
            category,
            f"[{status_style}]{status}[/{status_style}]",
            score_str,
            duration_str,
            steps_str,
            cost_str,
        )

    console.print(table)


def print_detailed_test_results(console: Console, detailed_results: Dict[str, Dict[str, Any]]) -> None:
    """Print detailed test results with outputs and errors (verbose mode).

    Args:
        console: Rich console for output
        detailed_results: Dict mapping model names to test results
    """
    console.print("\n")
    console.rule("[bold cyan]Detailed Test Results[/bold cyan]")

    for model_name, tests in detailed_results.items():
        for test_id, test_result in tests.items():
            # Test header
            passed = test_result.get("passed", False)
            status_icon = "✅" if passed else "❌"
            status_text = "PASS" if passed else "FAIL"
            status_color = "green" if passed else "red"

            console.print(f"\n📝 [bold]{test_id}[/bold]")
            cost_val = test_result.get("cost", 0)
            cost_str = f"${cost_val:.4f}" if cost_val > 0 else "$0.00"
            console.print(
                f"   Status: [{status_color}]{status_icon} {status_text}[/{status_color}] | "
                f"Score: {test_result.get('score', 0):.2f} | "
                f"Duration: {test_result.get('duration', 0):.2f}s | "
                f"Steps: {test_result.get('steps_taken', 0)} | "
                f"Cost: {cost_str}"
            )

            # Individual test cases (if available)
            metrics = test_result.get("metrics", {})
            case_results = metrics.get("case_results", [])
            if case_results:
                total_cases = metrics.get("total_cases", len(case_results))
                console.print(f"\n   Test Cases ({total_cases}):")
                for case in case_results:
                    case_passed = case.get("passed", False)
                    case_icon = "✅" if case_passed else "❌"
                    case_score = case.get("score", 0)
                    case_name = case.get("test_case", "Unknown")
                    console.print(f"   • {case_name} {case_icon} (score: {case_score:.2f})")

            # Actual vs Expected Output
            actual = test_result.get("output", "")
            expected = test_result.get("expected_output", "")

            if actual or expected:
                console.print("\n   [bold]Actual Output:[/bold]")
                if actual:
                    # Truncate if too long
                    display_actual = actual if len(actual) <= 500 else actual[:500] + "\n... (truncated)"
                    panel = Panel(display_actual, border_style="cyan", padding=(0, 1))
                    console.print(panel)
                else:
                    console.print("   [dim](empty)[/dim]")

                if expected:
                    console.print("\n   [bold]Expected Output:[/bold]")
                    display_expected = expected if len(expected) <= 200 else expected[:200] + "..."
                    console.print(f"   [yellow]{display_expected}[/yellow]")

            # Error details
            error = test_result.get("error")
            if error:
                console.print("\n   [bold red]Error:[/bold red]")
                error_panel = Panel(error, border_style="red", padding=(0, 1))
                console.print(error_panel)

            console.print()  # Blank line between tests


def print_error_summary(console: Console, errors: List[str], verbose: bool = False) -> None:
    """Print error summary.

    Args:
        console: Rich console for output
        errors: List of error messages
        verbose: If True, show all errors; otherwise show first 3
    """
    if not errors:
        return

    error_count = len(errors)
    console.print(f"\n[bold red]Errors ({error_count}):[/bold red]")

    # In verbose mode, show all errors; otherwise show first 3
    display_errors = errors if verbose else errors[:3]

    for error in display_errors:
        console.print(f"  [red]•[/red] {error}")

    if not verbose and len(errors) > 3:
        remaining = len(errors) - 3
        console.print(f"  [dim]... and {remaining} more errors (use --verbose to see all)[/dim]")


def load_benchmark_file(file_path: Path, console: Console) -> Dict[str, Any]:
    """Load benchmark JSON file.

    Args:
        file_path: Path to benchmark JSON file
        console: Rich console for error output

    Returns:
        Loaded JSON data as dict

    Raises:
        typer.Exit: If file doesn't exist or can't be loaded
    """
    import json

    import typer

    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        raise typer.Exit(1)

    try:
        with open(file_path) as f:
            return json.load(f)
    except Exception as e:
        console.print(f"[red]Failed to load benchmark data: {e}[/red]")
        raise typer.Exit(1)


def find_latest_benchmark() -> Optional[Path]:
    """Find the most recent benchmark JSON file.

    Returns:
        Path to latest benchmark file, or None if no files found
    """
    results_dir = Path("benchmark_results")
    if not results_dir.exists():
        return None

    # Find all benchmark JSON files
    json_files = list(results_dir.glob("benchmark_*.json"))
    if not json_files:
        return None

    # Sort by modification time, most recent first
    json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return json_files[0]


def resolve_benchmark_file(output: Optional[str], console: Console) -> Path:
    """Resolve which benchmark file to view.

    Args:
        output: User-specified file path or "latest"
        console: Rich console for error output

    Returns:
        Path to benchmark file

    Raises:
        typer.Exit: If no files found
    """
    import typer

    if output:
        # User specified a specific file
        if output == "latest":
            file_path = find_latest_benchmark()
            if not file_path:
                console.print("[red]No benchmark results found in benchmark_results/[/red]")
                raise typer.Exit(1)
        else:
            file_path = Path(output)
    else:
        # Default to latest
        file_path = find_latest_benchmark()
        if not file_path:
            console.print("[red]No benchmark results found in benchmark_results/[/red]")
            console.print("[dim]Run benchmarks first with: tsugite benchmark run --models <model>[/dim]")
            raise typer.Exit(1)

    return file_path


def run_benchmark_action(
    console: Console,
    models: Optional[str],
    categories: Optional[str],
    agent_path: Optional[str],
    output: Optional[str],
    format: str,
    test_filter: Optional[str],
    parallel: bool,
    repeat: int,
) -> None:
    """Handle the 'run' benchmark action.

    Args:
        console: Rich console for output
        models: Comma-separated list of models
        categories: Comma-separated list of categories
        agent_path: Path to specific agent file
        output: Output file path
        format: Report format (json, markdown, html, csv)
        test_filter: Filter tests by name/ID
        parallel: Run tests in parallel
        repeat: Number of times to repeat each test
    """
    import asyncio

    import typer

    from tsugite.benchmark import BenchmarkConfig, BenchmarkRunner
    from tsugite.benchmark.reports import ReportGenerator

    # Parse and validate arguments
    model_list, category_list = parse_benchmark_run_args(models, categories, console)

    # Create config
    config = BenchmarkConfig(
        models=model_list,
        categories=category_list,
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

        # Generate reports
        output_dir = Path("benchmark_results")
        output_dir.mkdir(exist_ok=True)
        timestamp = result.start_time.strftime("%Y%m%d_%H%M%S")
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
        summary = result.summary
        print_benchmark_summary(
            console,
            duration=result.total_duration,
            model_count=len(result.model_performances),
            total_tests=summary.get("total_tests", 0),
            avg_accuracy=summary.get("average_accuracy", 0),
            best_model=summary.get("best_model"),
        )

        # Print model performance
        print_model_performance(console, result.model_performances)

        # Print test results tables
        if result.model_performances:
            for model_name, _ in result.model_performances.items():
                model_tests = result.test_results.get(model_name, {})
                print_test_results_table(console, model_name, model_tests)

        # Print errors
        print_error_summary(console, result.errors, verbose=False)

    except Exception as e:
        console.print(f"[red]Benchmark failed: {e}[/red]")
        raise typer.Exit(1)


def view_benchmark_action(
    console: Console,
    output: Optional[str],
    verbose: bool,
) -> None:
    """Handle the 'view' benchmark action.

    Args:
        console: Rich console for output
        output: File path or "latest"
        verbose: Show detailed test outputs
    """
    # Resolve and load file
    file_path = resolve_benchmark_file(output, console)
    data = load_benchmark_file(file_path, console)

    # Display results
    console.print("\n[bold cyan]Benchmark Results[/bold cyan]")
    console.print(f"[dim]File: {file_path}[/dim]")
    console.print(f"[dim]Generated: {data.get('generated_at', 'Unknown')}[/dim]\n")

    console.print("=" * 50)
    console.print("[bold green]Summary[/bold green]")
    console.print("=" * 50)

    # Print summary
    benchmark_info = data.get("benchmark_info", {})
    summary = data.get("summary", {})

    print_benchmark_summary(
        console,
        duration=benchmark_info.get("total_duration", 0),
        model_count=len(benchmark_info.get("models_tested", [])),
        total_tests=benchmark_info.get("total_tests", 0),
        avg_accuracy=summary.get("average_accuracy", 0),
        best_model=summary.get("best_model"),
    )

    # Print model performance
    model_performances = data.get("model_performances", {})
    print_model_performance(console, model_performances)

    # Print test results tables
    detailed_results = data.get("detailed_results", {})
    for model_name, tests in detailed_results.items():
        print_test_results_table(console, model_name, tests)

    # Detailed test breakdown (verbose mode)
    if verbose:
        print_detailed_test_results(console, detailed_results)

    # Print errors
    errors = data.get("errors", [])
    print_error_summary(console, errors, verbose)


def list_benchmark_action(console: Console) -> None:
    """Handle the 'list' benchmark action.

    Args:
        console: Rich console for output
    """
    import json
    from datetime import datetime

    import typer

    results_dir = Path("benchmark_results")
    if not results_dir.exists():
        console.print("[red]No benchmark results directory found[/red]")
        console.print("[dim]Run benchmarks first with: tsugite benchmark run --models <model>[/dim]")
        raise typer.Exit(1)

    # Find all benchmark JSON files
    json_files = list(results_dir.glob("benchmark_*.json"))
    if not json_files:
        console.print("[red]No benchmark results found in benchmark_results/[/red]")
        console.print("[dim]Run benchmarks first with: tsugite benchmark run --models <model>[/dim]")
        raise typer.Exit(1)

    # Sort by modification time, most recent first
    json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    console.print("\n[bold cyan]Benchmark Results[/bold cyan]")
    console.print(f"Found {len(json_files)} result file(s) in {results_dir}/\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("", width=3)  # Marker for latest
    table.add_column("Timestamp", style="cyan", width=20)
    table.add_column("Models", width=15)
    table.add_column("Tests", justify="right", width=7)
    table.add_column("Accuracy", justify="right", width=10)
    table.add_column("Duration", justify="right", width=10)
    table.add_column("File", style="dim", width=30)

    for i, file_path in enumerate(json_files):
        try:
            with open(file_path) as f:
                data = json.load(f)

            benchmark_info = data.get("benchmark_info", {})
            summary = data.get("summary", {})

            # Extract info
            timestamp_str = benchmark_info.get("start_time", "")
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str).strftime("%Y-%m-%d %H:%M:%S")
            else:
                timestamp = "Unknown"

            models = benchmark_info.get("models_tested", [])
            model_str = ", ".join(models) if models else "Unknown"
            if len(model_str) > 15:
                model_str = model_str[:12] + "..."

            tests = benchmark_info.get("total_tests", 0)
            accuracy = summary.get("average_accuracy", 0)
            duration = benchmark_info.get("total_duration", 0)

            # Mark most recent
            marker = "→" if i == 0 else ""

            table.add_row(
                f"[green]{marker}[/green]" if marker else "",
                timestamp,
                model_str,
                str(tests),
                f"{accuracy:.1%}",
                f"{duration:.1f}s",
                file_path.name,
            )

        except Exception:
            # Skip files that can't be loaded
            table.add_row("", "Error", "-", "-", "-", "-", f"[red]{file_path.name}[/red]")

    console.print(table)
    console.print("\n[dim]View details: tsugite benchmark view [file|latest][/dim]")
    console.print("[dim]Most recent run is marked with →[/dim]")
