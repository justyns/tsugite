"""Metrics and data structures for benchmark results."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkTestResult:
    """Result from running a single test."""

    test_id: str
    model: str
    passed: bool
    score: float  # 0.0 to 1.0
    duration: float  # seconds
    output: str
    expected_output: str
    category: str = "unknown"
    error: Optional[str] = None
    token_usage: Dict[str, int] = field(default_factory=dict)
    cost: float = 0.0
    steps_taken: int = 0  # Number of reasoning steps/attempts
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ModelPerformance:
    """Aggregated performance metrics for a model."""

    model: str
    total_tests: int
    passed_tests: int
    accuracy: float  # passed_tests / total_tests
    average_duration: float
    total_duration: float
    total_tokens: int
    total_cost: float
    scores_by_category: Dict[str, float] = field(default_factory=dict)
    error_rate: float = 0.0
    reliability_score: float = 0.0
    average_steps: float = 0.0


@dataclass
class CategoryMetrics:
    """Performance metrics for a test category."""

    category: str
    total_tests: int
    model_scores: Dict[str, float]  # model -> average score
    average_score: float
    best_model: str
    worst_model: str


class BenchmarkMetrics:
    """Utility class for calculating benchmark metrics."""

    @staticmethod
    def calculate_accuracy(passed: int, total: int) -> float:
        """Calculate accuracy percentage."""
        if total == 0:
            return 0.0
        return passed / total

    @staticmethod
    def calculate_efficiency_score(duration: float, baseline_duration: float) -> float:
        """Calculate efficiency score relative to baseline (higher is better)."""
        if baseline_duration <= 0:
            return 1.0
        return min(baseline_duration / duration, 1.0)

    @staticmethod
    def calculate_cost_efficiency(score: float, cost: float) -> float:
        """Calculate cost efficiency (score per unit cost)."""
        if cost <= 0:
            return score
        return score / cost

    @staticmethod
    def calculate_weighted_score(scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate weighted average of scores."""
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(scores.get(metric, 0.0) * weight for metric, weight in weights.items())
        return weighted_sum / total_weight

    @staticmethod
    def calculate_reliability_score(test_results: List[BenchmarkTestResult]) -> float:
        """Calculate reliability score based on error patterns."""
        if not test_results:
            return 0.0

        total_tests = len(test_results)
        error_count = sum(1 for result in test_results if result.error is not None)
        timeout_count = sum(1 for result in test_results if result.error and "timeout" in result.error.lower())

        # Penalize errors and timeouts more heavily
        error_penalty = (error_count * 0.5 + timeout_count * 0.3) / total_tests
        reliability = max(0.0, 1.0 - error_penalty)

        return reliability

    @staticmethod
    def calculate_category_performance(test_results: Dict[str, BenchmarkTestResult], category: str) -> CategoryMetrics:
        """Calculate performance metrics for a specific category."""
        category_results = [result for result in test_results.values() if result.test_id.startswith(category)]

        if not category_results:
            return CategoryMetrics(
                category=category,
                total_tests=0,
                model_scores={},
                average_score=0.0,
                best_model="",
                worst_model="",
            )

        # Group by model
        model_scores = {}
        for result in category_results:
            if result.model not in model_scores:
                model_scores[result.model] = []
            model_scores[result.model].append(result.score)

        # Calculate average scores per model
        avg_model_scores = {model: round(sum(scores) / len(scores), 4) for model, scores in model_scores.items()}

        # Find best and worst models
        if avg_model_scores:
            best_model = max(avg_model_scores.items(), key=lambda x: x[1])[0]
            worst_model = min(avg_model_scores.items(), key=lambda x: x[1])[0]
            average_score = round(sum(avg_model_scores.values()) / len(avg_model_scores), 4)
        else:
            best_model = worst_model = ""
            average_score = 0.0

        return CategoryMetrics(
            category=category,
            total_tests=len(category_results),
            model_scores=avg_model_scores,
            average_score=average_score,
            best_model=best_model,
            worst_model=worst_model,
        )

    @staticmethod
    def get_performance_tier(accuracy: float) -> str:
        """Classify performance into tiers."""
        if accuracy >= 0.9:
            return "Excellent"
        elif accuracy >= 0.75:
            return "Good"
        elif accuracy >= 0.6:
            return "Fair"
        elif accuracy >= 0.4:
            return "Poor"
        else:
            return "Very Poor"

    @staticmethod
    def calculate_improvement_percentage(baseline_score: float, current_score: float) -> float:
        """Calculate percentage improvement over baseline."""
        if baseline_score <= 0:
            return 0.0

        improvement = ((current_score - baseline_score) / baseline_score) * 100
        return round(improvement, 4)

    @staticmethod
    def aggregate_test_results(results: List[BenchmarkTestResult]) -> Dict[str, Any]:
        """Aggregate multiple test results into summary statistics."""
        if not results:
            return {
                "total_tests": 0,
                "passed_tests": 0,
                "accuracy": 0.0,
                "average_score": 0.0,
                "average_duration": 0.0,
                "total_cost": 0.0,
                "error_rate": 0.0,
            }

        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        total_score = sum(r.score for r in results)
        total_duration = sum(r.duration for r in results)
        total_cost = sum(r.cost for r in results)
        error_count = sum(1 for r in results if r.error is not None)

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "accuracy": passed_tests / total_tests,
            "average_score": total_score / total_tests,
            "average_duration": total_duration / total_tests,
            "total_duration": total_duration,
            "total_cost": total_cost,
            "error_rate": error_count / total_tests,
        }
