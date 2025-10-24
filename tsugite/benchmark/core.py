"""Core benchmark framework for evaluating Tsugite agents."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .discovery import BenchmarkTest, TestDiscovery
from .execution import TestExecutor
from .metrics import BenchmarkTestResult, ModelPerformance


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    models: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=lambda: ["basic"])
    timeout: int = 120  # seconds
    parallel: bool = True
    output_dir: Path = field(default_factory=lambda: Path("benchmark_results"))
    repeat_count: int = 1  # Number of times to run each test for averaging
    llm_evaluator_model: str = "openai:gpt-4o-mini"  # Model to use for LLM evaluation


@dataclass
class BenchmarkResult:
    """Results from a complete benchmark run."""

    config: BenchmarkConfig
    start_time: datetime
    end_time: datetime
    total_duration: float
    model_performances: Dict[str, ModelPerformance]
    test_results: Dict[str, Dict[str, BenchmarkTestResult]]  # model -> test_id -> result
    summary: Dict[str, Any]
    errors: List[str] = field(default_factory=list)


class BenchmarkRunner:
    """Main benchmark runner for evaluating models across test suites."""

    benchmark_dir: Path = Path("benchmarks")

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.benchmark_dir = Path(type(self).benchmark_dir)

        # Initialize components
        self.discovery = TestDiscovery(self.benchmark_dir, config.timeout)
        self.executor = TestExecutor(config.output_dir, config.llm_evaluator_model)

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def discover_tests(
        self, categories: Optional[List[str]] = None, agent_path: Optional[Path] = None
    ) -> List[BenchmarkTest]:
        """Discover benchmark tests.

        Args:
            categories: Categories to search (e.g., ["basic", "tools"])
            agent_path: Specific agent file to test

        Returns:
            List of discovered tests
        """
        if categories is None:
            categories = self.config.categories

        return self.discovery.discover_tests(categories, agent_path)

    async def run_benchmark(
        self,
        models: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        test_filter: Optional[str] = None,
        agent_path: Optional[Path] = None,
    ) -> BenchmarkResult:
        """Run benchmark suite against specified models.

        Args:
            models: List of models to test
            categories: Test categories to run
            test_filter: Filter tests by name/ID substring
            agent_path: Specific agent file to test

        Returns:
            Benchmark results

        Raises:
            ValueError: If no models or tests found
        """
        start_time = datetime.now()

        # Validate models
        if models is None:
            models = self.config.models
        if not models:
            raise ValueError("No models specified for benchmarking")

        # Discover tests
        if agent_path:
            tests = self.discovery.discover_tests(agent_path=agent_path)
        else:
            tests = self.discovery.discover_tests(categories)
            if test_filter:
                tests = [
                    t
                    for t in tests
                    if test_filter.lower() in t.name.lower() or test_filter.lower() in t.test_id.lower()
                ]

        if not tests:
            raise ValueError("No tests found matching criteria")

        print(f"Running {len(tests)} tests across {len(models)} models...")

        # Run tests for each model
        model_performances = {}
        test_results = {model: {} for model in models}
        errors = []

        for model_name in models:
            print(f"\nEvaluating model: {model_name}")
            try:
                model_perf, model_test_results, model_errors = await self._run_model_tests(model_name, tests)
                model_performances[model_name] = model_perf
                test_results[model_name] = model_test_results
                errors.extend(model_errors)
            except Exception as e:
                error_msg = f"Failed to evaluate model {model_name}: {e}"
                errors.append(error_msg)
                print(f"Error: {error_msg}")

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        # Generate summary
        summary = self._generate_summary(model_performances, test_results)

        return BenchmarkResult(
            config=self.config,
            start_time=start_time,
            end_time=end_time,
            total_duration=total_duration,
            model_performances=model_performances,
            test_results=test_results,
            summary=summary,
            errors=errors,
        )

    async def _run_model_tests(
        self, model_name: str, tests: List[BenchmarkTest]
    ) -> tuple[ModelPerformance, Dict[str, BenchmarkTestResult], List[str]]:
        """Run all tests for a single model.

        Args:
            model_name: Model to test
            tests: List of tests to run

        Returns:
            Tuple of (model_performance, test_results, errors)
        """
        model_test_results = {}
        model_errors = []

        # Aggregate metrics
        total_tests = len(tests)
        passed_tests = 0
        total_duration = 0.0
        total_tokens = 0
        total_cost = 0.0
        total_steps = 0

        for test in tests:
            print(f"  Running test: {test.test_id}")

            try:
                test_result = await self.executor.run_test(model_name, test)
                model_test_results[test.test_id] = test_result

                if test_result.error:
                    model_errors.append(f"Test {test.test_id} for model {model_name}: {test_result.error}")

                if test_result.passed:
                    passed_tests += 1

                total_duration += test_result.duration
                total_tokens += test_result.token_usage.get("total", 0)
                total_cost += test_result.cost
                total_steps += test_result.steps_taken

            except Exception as e:
                error_msg = f"Test {test.test_id} for model {model_name} failed: {e}"
                model_errors.append(error_msg)
                print(f"    Error: {error_msg}")

                # Create failed test result
                failed_result = BenchmarkTestResult(
                    test_id=test.test_id,
                    model=model_name,
                    passed=False,
                    score=0.0,
                    duration=0.0,
                    output="",
                    expected_output=test.expected_output or "",
                    error=str(e),
                    token_usage={},
                    cost=0.0,
                    metrics={},
                )
                model_test_results[test.test_id] = failed_result

        # Calculate overall metrics
        accuracy = passed_tests / total_tests if total_tests > 0 else 0.0
        avg_duration = total_duration / total_tests if total_tests > 0 else 0.0
        avg_steps = total_steps / total_tests if total_tests > 0 else 0.0

        model_performance = ModelPerformance(
            model=model_name,
            total_tests=total_tests,
            passed_tests=passed_tests,
            accuracy=accuracy,
            average_duration=avg_duration,
            total_duration=total_duration,
            total_tokens=total_tokens,
            total_cost=total_cost,
            scores_by_category={},  # Will be calculated in summary
            average_steps=avg_steps,
        )

        return model_performance, model_test_results, model_errors

    def _generate_summary(
        self,
        model_performances: Dict[str, ModelPerformance],
        test_results: Dict[str, Dict[str, BenchmarkTestResult]],
    ) -> Dict[str, Any]:
        """Generate summary statistics from benchmark results.

        Args:
            model_performances: Performance data per model
            test_results: Detailed test results per model

        Returns:
            Summary dictionary
        """
        summary = {
            "total_models": len(model_performances),
            "model_rankings": [],
            "category_performance": {},
            "best_model": None,
            "worst_model": None,
            "average_accuracy": 0.0,
            "total_tests": 0,
        }

        if not model_performances:
            return summary

        # Calculate model rankings by accuracy
        ranked_models = sorted(model_performances.items(), key=lambda x: x[1].accuracy, reverse=True)

        summary["model_rankings"] = [
            {
                "model": model,
                "accuracy": perf.accuracy,
                "avg_duration": perf.average_duration,
                "total_cost": perf.total_cost,
            }
            for model, perf in ranked_models
        ]

        summary["best_model"] = ranked_models[0][0] if ranked_models else None
        summary["worst_model"] = ranked_models[-1][0] if ranked_models else None

        # Calculate average accuracy
        total_accuracy = sum(perf.accuracy for perf in model_performances.values())
        summary["average_accuracy"] = total_accuracy / len(model_performances)

        # Get total tests from first model
        if model_performances:
            first_model = next(iter(model_performances.values()))
            summary["total_tests"] = first_model.total_tests

        return summary
