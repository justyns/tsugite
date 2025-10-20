"""Benchmark CLI commands."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

console = Console()


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


def benchmark_command(
    action: str = typer.Argument(help="Action: run, view, list, compare"),
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
    import asyncio
    from pathlib import Path

    from tsugite.benchmark import BenchmarkConfig, BenchmarkRunner
    from tsugite.benchmark.reports import ReportGenerator

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
                    console.print(f"    Avg Steps: {perf.average_steps:.1f}")
                    if perf.total_cost > 0:
                        console.print(f"    Total Cost: ${perf.total_cost:.4f}")

            # Print test results table
            if len(result.model_performances) > 0:
                from rich.table import Table

                for model_name in result.model_performances.keys():
                    console.print(f"\n[bold]Test Results - {model_name}:[/bold]")

                    table = Table(show_header=True, header_style="bold magenta")
                    table.add_column("Test ID", style="dim", width=30)
                    table.add_column("Category", width=12)
                    table.add_column("Status", justify="center", width=10)
                    table.add_column("Score", justify="right", width=8)
                    table.add_column("Duration", justify="right", width=10)
                    table.add_column("Steps", justify="right", width=8)
                    table.add_column("Cost", justify="right", width=10)

                    model_tests = result.test_results.get(model_name, {})
                    test_items = list(model_tests.items())

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
                            table.add_row(
                                f"... {num_hidden} more tests ...", "", "", "", "", "", "", style="dim italic"
                            )

                        category = test_result.category
                        status = "✅ PASS" if test_result.passed else "❌ FAIL"
                        status_style = "green" if test_result.passed else "red"
                        score = f"{test_result.score:.2f}"
                        duration = f"{test_result.duration:.2f}s"
                        steps = str(test_result.steps_taken)
                        cost = f"${test_result.cost:.4f}" if test_result.cost > 0 else "$0.00"

                        table.add_row(
                            test_id,
                            category,
                            f"[{status_style}]{status}[/{status_style}]",
                            score,
                            duration,
                            steps,
                            cost,
                        )

                    console.print(table)

            # Print top errors
            if result.errors:
                console.print(f"\n[bold red]Errors ({len(result.errors)}):[/bold red]")
                # Show first 3 errors inline
                for error in result.errors[:3]:
                    console.print(f"  [red]•[/red] {error}")
                if len(result.errors) > 3:
                    console.print(f"  [dim]... and {len(result.errors) - 3} more errors (see markdown report)[/dim]")

        except Exception as e:
            console.print(f"[red]Benchmark failed: {e}[/red]")
            raise typer.Exit(1)

    elif action == "compare":
        console.print("[yellow]Compare functionality not yet implemented[/yellow]")

    elif action == "view":
        # Determine which file to view
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

        if not file_path.exists():
            console.print(f"[red]File not found: {file_path}[/red]")
            raise typer.Exit(1)

        # Load the JSON data
        try:
            with open(file_path) as f:
                data = json.load(f)
        except Exception as e:
            console.print(f"[red]Failed to load benchmark data: {e}[/red]")
            raise typer.Exit(1)

        # Display results
        console.print("\n[bold cyan]Benchmark Results[/bold cyan]")
        console.print(f"[dim]File: {file_path}[/dim]")
        console.print(f"[dim]Generated: {data.get('generated_at', 'Unknown')}[/dim]\n")

        console.print("=" * 50)
        console.print("[bold green]Summary[/bold green]")
        console.print("=" * 50)

        benchmark_info = data.get("benchmark_info", {})
        summary = data.get("summary", {})

        console.print(f"Duration: {benchmark_info.get('total_duration', 0):.2f}s")
        console.print(f"Models: {len(benchmark_info.get('models_tested', []))}")
        console.print(f"Tests: {benchmark_info.get('total_tests', 0)}")
        console.print(f"Average Accuracy: {summary.get('average_accuracy', 0):.1%}")

        if summary.get("best_model"):
            console.print(f"Best Model: {summary['best_model']}")

        # Model performance
        model_performances = data.get("model_performances", {})
        if model_performances:
            console.print("\n[bold]Model Performance:[/bold]")
            for model_name, perf in model_performances.items():
                console.print(f"  [cyan]{model_name}[/cyan]:")
                console.print(
                    f"    Accuracy: {perf['accuracy']:.1%} ({perf['passed_tests']}/{perf['total_tests']} passed)"
                )
                console.print(f"    Avg Duration: {perf['average_duration']:.2f}s")
                console.print(f"    Avg Steps: {perf.get('average_steps', 0):.1f}")
                if perf.get("total_cost", 0) > 0:
                    console.print(f"    Total Cost: ${perf['total_cost']:.4f}")

        # Test results table
        from rich.table import Table

        detailed_results = data.get("detailed_results", {})
        for model_name, tests in detailed_results.items():
            console.print(f"\n[bold]Test Results - {model_name}:[/bold]")

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Test ID", style="dim", width=30)
            table.add_column("Category", width=12)
            table.add_column("Status", justify="center", width=10)
            table.add_column("Score", justify="right", width=8)
            table.add_column("Duration", justify="right", width=10)
            table.add_column("Steps", justify="right", width=8)
            table.add_column("Cost", justify="right", width=10)

            test_items = list(tests.items())

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

                category = test_result.get("category", "unknown")
                passed = test_result.get("passed", False)
                status = "✅ PASS" if passed else "❌ FAIL"
                status_style = "green" if passed else "red"
                score = f"{test_result.get('score', 0):.2f}"
                duration = f"{test_result.get('duration', 0):.2f}s"
                steps = str(test_result.get("steps_taken", 0))
                cost_val = test_result.get("cost", 0)
                cost = f"${cost_val:.4f}" if cost_val > 0 else "$0.00"

                table.add_row(
                    test_id, category, f"[{status_style}]{status}[/{status_style}]", score, duration, steps, cost
                )

            console.print(table)

        # Detailed test breakdown (verbose mode)
        if verbose:
            from rich.panel import Panel

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
                            panel = Panel(
                                display_actual,
                                border_style="cyan",
                                padding=(0, 1),
                            )
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
                        error_panel = Panel(
                            error,
                            border_style="red",
                            padding=(0, 1),
                        )
                        console.print(error_panel)

                    console.print()  # Blank line between tests

        # Errors summary
        errors = data.get("errors", [])
        if errors:
            error_count = len(errors)
            console.print(f"\n[bold red]Errors ({error_count}):[/bold red]")

            # In verbose mode, show all errors; otherwise show first 3
            display_errors = errors if verbose else errors[:3]

            for error in display_errors:
                console.print(f"  [red]•[/red] {error}")

            if not verbose and len(errors) > 3:
                console.print(f"  [dim]... and {len(errors) - 3} more errors (use --verbose to see all)[/dim]")

    elif action == "list":
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

        from datetime import datetime

        from rich.table import Table

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

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available actions: run, view, list, compare")
        raise typer.Exit(1)
