"""Benchmark CLI commands."""

from typing import Optional

import typer
from rich.console import Console

console = Console()


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
