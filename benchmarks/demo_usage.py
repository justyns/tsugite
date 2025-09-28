#!/usr/bin/env python3
"""
Demo script showing how to use the Tsugite benchmark framework.

This script demonstrates:
1. Discovering benchmark tests
2. Running evaluators on sample data
3. Generating reports
"""

import asyncio
from pathlib import Path
from tsugite.benchmark import (
    BenchmarkRunner,
    BenchmarkConfig,
    CorrectnessEvaluator,
    PerformanceEvaluator,
    QualityEvaluator,
    ReportGenerator,
)
from tsugite.benchmark.core import BenchmarkResult
from tsugite.benchmark.metrics import TestResult, ModelPerformance
from datetime import datetime


def demo_test_discovery():
    """Demonstrate test discovery functionality."""
    print("🔍 Demo: Test Discovery")
    print("=" * 50)

    config = BenchmarkConfig(models=["demo-model"])
    runner = BenchmarkRunner(config)

    # Discover tests in all categories
    for category in ["basic", "tools", "scenarios", "performance"]:
        try:
            tests = runner.discover_tests([category])
            print(f"\n📁 {category.title()} Category: {len(tests)} tests")
            for test in tests:
                print(f"  • {test.name} ({test.test_id})")
                if test.expected_output:
                    print(f"    Expected: {test.expected_output}")
                if test.evaluation_criteria:
                    criteria_list = ", ".join(test.evaluation_criteria.keys())
                    print(f"    Criteria: {criteria_list}")
        except Exception as e:
            print(f"  ⚠️  Error discovering {category} tests: {e}")


def demo_evaluators():
    """Demonstrate evaluator functionality."""
    print("\n\n📊 Demo: Evaluators")
    print("=" * 50)

    # Correctness Evaluator
    print("\n1. Correctness Evaluator")
    correctness = CorrectnessEvaluator()

    # Test exact match
    result = correctness.evaluate(output="Hello, World!", expected="Hello, World!", output_type="string")
    print(f"  Exact match: {result['passed']} (score: {result['score']:.2f})")

    # Test number evaluation
    result = correctness.evaluate(output="42", expected="42", output_type="number")
    print(f"  Number match: {result['passed']} (score: {result['score']:.2f})")

    # Test JSON evaluation
    result = correctness.evaluate(
        output='{"name": "test", "value": 42}',
        expected='{"name": "test", "value": 42}',
        output_type="json",
    )
    print(f"  JSON match: {result['passed']} (score: {result['score']:.2f})")

    # Performance Evaluator
    print("\n2. Performance Evaluator")
    performance = PerformanceEvaluator()

    # Test fast execution
    result = performance.evaluate(duration=1.0, timeout=10.0)
    print(f"  Fast execution: {result['efficiency_tier']} (score: {result['speed_score']:.2f})")

    # Test slow execution
    result = performance.evaluate(duration=8.0, timeout=10.0)
    print(f"  Slow execution: {result['efficiency_tier']} (score: {result['speed_score']:.2f})")

    # Quality Evaluator
    print("\n3. Quality Evaluator")
    quality = QualityEvaluator()

    async def test_quality():
        criteria = {
            "completeness": {
                "type": "keyword",
                "keywords": ["complete", "done"],
                "weight": 1.0,
            }
        }

        result = await quality.evaluate(output="The task is complete and done successfully.", criteria=criteria)
        print(f"  Quality check: {result['overall_quality']} (score: {result['score']:.2f})")

    asyncio.run(test_quality())


def demo_report_generation():
    """Demonstrate report generation."""
    print("\n\n📄 Demo: Report Generation")
    print("=" * 50)

    # Create sample benchmark result
    config = BenchmarkConfig(models=["demo-model-1", "demo-model-2"])
    start_time = datetime.now()
    end_time = datetime.now()

    # Sample model performances
    model_performances = {
        "demo-model-1": ModelPerformance(
            model="demo-model-1",
            total_tests=4,
            passed_tests=3,
            accuracy=0.75,
            average_duration=2.0,
            total_duration=8.0,
            total_tokens=400,
            total_cost=0.02,
        ),
        "demo-model-2": ModelPerformance(
            model="demo-model-2",
            total_tests=4,
            passed_tests=4,
            accuracy=1.0,
            average_duration=3.0,
            total_duration=12.0,
            total_tokens=500,
            total_cost=0.03,
        ),
    }

    # Sample test results
    test_results = {
        "demo-model-1": {
            "basic_001": TestResult(
                "basic_001",
                "demo-model-1",
                True,
                0.9,
                1.5,
                "Hello, World!",
                "Hello, World!",
            ),
            "basic_002": TestResult("basic_002", "demo-model-1", False, 0.3, 2.0, "24", "42"),
        },
        "demo-model-2": {
            "basic_001": TestResult(
                "basic_001",
                "demo-model-2",
                True,
                1.0,
                2.0,
                "Hello, World!",
                "Hello, World!",
            ),
            "basic_002": TestResult("basic_002", "demo-model-2", True, 0.9, 3.0, "42", "42"),
        },
    }

    summary = {
        "total_models": 2,
        "total_tests": 2,
        "average_accuracy": 0.875,
        "best_model": "demo-model-2",
        "model_rankings": [
            {
                "model": "demo-model-2",
                "accuracy": 1.0,
                "avg_duration": 3.0,
                "total_cost": 0.03,
            },
            {
                "model": "demo-model-1",
                "accuracy": 0.75,
                "avg_duration": 2.0,
                "total_cost": 0.02,
            },
        ],
    }

    result = BenchmarkResult(
        config=config,
        start_time=start_time,
        end_time=end_time,
        total_duration=20.0,
        model_performances=model_performances,
        test_results=test_results,
        summary=summary,
    )

    # Generate reports
    report_gen = ReportGenerator(result)

    # Create output directory
    output_dir = Path("demo_reports")
    output_dir.mkdir(exist_ok=True)

    # Generate different report formats
    formats = [
        ("json", "demo_report.json"),
        ("markdown", "demo_report.md"),
        ("html", "demo_report.html"),
        ("csv", "demo_report.csv"),
    ]

    for format_name, filename in formats:
        output_path = output_dir / filename
        try:
            if format_name == "json":
                report_gen.generate_json_report(output_path)
            elif format_name == "markdown":
                report_gen.generate_markdown_report(output_path)
            elif format_name == "html":
                report_gen.generate_html_report(output_path)
            elif format_name == "csv":
                report_gen.generate_csv_summary(output_path)

            print(f"  ✅ Generated {format_name.upper()} report: {output_path}")
        except Exception as e:
            print(f"  ❌ Failed to generate {format_name} report: {e}")


def main():
    """Run all demos."""
    print("🚀 Tsugite Benchmark Framework Demo")
    print("=" * 60)

    try:
        demo_test_discovery()
        demo_evaluators()
        demo_report_generation()

        print("\n\n✨ Demo completed successfully!")
        print("\nNext steps:")
        print("1. Run: tsugite benchmark run --models 'your-model' --categories 'basic'")
        print("2. Check the generated reports in benchmark_results/")
        print("3. Create custom benchmark tests in benchmarks/")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
