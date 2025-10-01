"""Tests for benchmark metrics functionality."""

import pytest
from datetime import datetime
from tsugite.benchmark.metrics import (
    BenchmarkTestResult,
    ModelPerformance,
    CategoryMetrics,
    BenchmarkMetrics,
)


class TestBenchmarkTestResult:
    """Test BenchmarkTestResult data structure."""

    def test_test_result_creation(self):
        """Test creating a test result."""
        result = BenchmarkTestResult(
            test_id="test_001",
            model="test-model:v1",
            passed=True,
            score=0.95,
            duration=1.5,
            output="42",
            expected_output="42",
            token_usage={"total": 100},
            cost=0.001,
        )

        assert result.test_id == "test_001"
        assert result.model == "test-model:v1"
        assert result.passed is True
        assert result.score == 0.95
        assert result.duration == 1.5
        assert result.output == "42"
        assert result.expected_output == "42"
        assert result.token_usage["total"] == 100
        assert result.cost == 0.001
        assert result.error is None
        assert isinstance(result.timestamp, datetime)

    def test_test_result_with_error(self):
        """Test creating a test result with error."""
        result = BenchmarkTestResult(
            test_id="test_002",
            model="test-model:v1",
            passed=False,
            score=0.0,
            duration=0.0,
            output="",
            expected_output="42",
            error="Test failed with exception",
        )

        assert result.passed is False
        assert result.score == 0.0
        assert result.error == "Test failed with exception"


class TestModelPerformance:
    """Test ModelPerformance data structure."""

    def test_model_performance_creation(self):
        """Test creating model performance metrics."""
        performance = ModelPerformance(
            model="test-model:v1",
            total_tests=10,
            passed_tests=8,
            accuracy=0.8,
            average_duration=2.5,
            total_duration=25.0,
            total_tokens=1000,
            total_cost=0.05,
        )

        assert performance.model == "test-model:v1"
        assert performance.total_tests == 10
        assert performance.passed_tests == 8
        assert performance.accuracy == 0.8
        assert performance.average_duration == 2.5
        assert performance.total_duration == 25.0
        assert performance.total_tokens == 1000
        assert performance.total_cost == 0.05
        assert performance.scores_by_category == {}
        assert performance.error_rate == 0.0
        assert performance.reliability_score == 0.0


class TestCategoryMetrics:
    """Test CategoryMetrics data structure."""

    def test_category_metrics_creation(self):
        """Test creating category metrics."""
        metrics = CategoryMetrics(
            category="basic",
            total_tests=5,
            model_scores={"model1": 0.9, "model2": 0.7},
            average_score=0.8,
            best_model="model1",
            worst_model="model2",
        )

        assert metrics.category == "basic"
        assert metrics.total_tests == 5
        assert metrics.model_scores["model1"] == 0.9
        assert metrics.model_scores["model2"] == 0.7
        assert metrics.average_score == 0.8
        assert metrics.best_model == "model1"
        assert metrics.worst_model == "model2"


class TestBenchmarkMetrics:
    """Test BenchmarkMetrics utility functions."""

    def test_calculate_accuracy(self):
        """Test accuracy calculation."""
        assert BenchmarkMetrics.calculate_accuracy(8, 10) == 0.8
        assert BenchmarkMetrics.calculate_accuracy(10, 10) == 1.0
        assert BenchmarkMetrics.calculate_accuracy(0, 10) == 0.0
        assert BenchmarkMetrics.calculate_accuracy(0, 0) == 0.0

    def test_calculate_efficiency_score(self):
        """Test efficiency score calculation."""
        # Faster than baseline should give high score
        assert BenchmarkMetrics.calculate_efficiency_score(1.0, 2.0) == 1.0

        # Same as baseline
        assert BenchmarkMetrics.calculate_efficiency_score(2.0, 2.0) == 1.0

        # Slower than baseline should give lower score
        assert BenchmarkMetrics.calculate_efficiency_score(4.0, 2.0) == 0.5

        # Edge case: zero baseline
        assert BenchmarkMetrics.calculate_efficiency_score(1.0, 0.0) == 1.0

    def test_calculate_cost_efficiency(self):
        """Test cost efficiency calculation."""
        assert BenchmarkMetrics.calculate_cost_efficiency(0.9, 0.01) == 90.0
        assert BenchmarkMetrics.calculate_cost_efficiency(0.5, 0.02) == 25.0

        # Zero cost should return the score itself
        assert BenchmarkMetrics.calculate_cost_efficiency(0.8, 0.0) == 0.8

    def test_calculate_weighted_score(self):
        """Test weighted score calculation."""
        scores = {"accuracy": 0.9, "speed": 0.7, "cost": 0.8}
        weights = {"accuracy": 0.5, "speed": 0.3, "cost": 0.2}

        expected = (0.9 * 0.5 + 0.7 * 0.3 + 0.8 * 0.2) / 1.0
        result = BenchmarkMetrics.calculate_weighted_score(scores, weights)

        assert abs(result - expected) < 0.001

        # Test with zero weights
        assert BenchmarkMetrics.calculate_weighted_score(scores, {}) == 0.0

    def test_calculate_reliability_score(self):
        """Test reliability score calculation."""
        # All successful tests
        successful_results = [
            BenchmarkTestResult("test1", "model", True, 1.0, 1.0, "output", "expected"),
            BenchmarkTestResult("test2", "model", True, 1.0, 1.0, "output", "expected"),
        ]

        reliability = BenchmarkMetrics.calculate_reliability_score(successful_results)
        assert reliability == 1.0

        # Some errors
        mixed_results = [
            BenchmarkTestResult("test1", "model", True, 1.0, 1.0, "output", "expected"),
            BenchmarkTestResult("test2", "model", False, 0.0, 0.0, "", "expected", error="Failed"),
        ]

        reliability = BenchmarkMetrics.calculate_reliability_score(mixed_results)
        assert reliability < 1.0  # Should be penalized for errors

        # Timeout errors (more penalty)
        timeout_results = [
            BenchmarkTestResult("test1", "model", True, 1.0, 1.0, "output", "expected"),
            BenchmarkTestResult("test2", "model", False, 0.0, 0.0, "", "expected", error="timeout"),
        ]

        timeout_reliability = BenchmarkMetrics.calculate_reliability_score(timeout_results)
        error_reliability = BenchmarkMetrics.calculate_reliability_score(mixed_results)
        assert timeout_reliability < error_reliability  # Timeouts penalized more

        # Empty list
        assert BenchmarkMetrics.calculate_reliability_score([]) == 0.0

    def test_calculate_category_performance(self):
        """Test category performance calculation."""
        test_results = {
            "basic_001": BenchmarkTestResult("basic_001", "model1", True, 0.9, 1.0, "out", "exp"),
            "basic_002": BenchmarkTestResult("basic_002", "model1", True, 0.8, 1.0, "out", "exp"),
            "basic_001_m2": BenchmarkTestResult("basic_001", "model2", True, 0.7, 1.0, "out", "exp"),
            "basic_002_m2": BenchmarkTestResult("basic_002", "model2", True, 0.6, 1.0, "out", "exp"),
            "tools_001": BenchmarkTestResult("tools_001", "model1", True, 0.95, 1.0, "out", "exp"),
        }

        # Test basic category
        basic_metrics = BenchmarkMetrics.calculate_category_performance(test_results, "basic")

        assert basic_metrics.category == "basic"
        assert basic_metrics.total_tests == 4  # 2 tests x 2 models
        assert basic_metrics.best_model == "model1"  # Higher average score
        assert basic_metrics.model_scores["model1"] == 0.85  # (0.9 + 0.8) / 2
        assert basic_metrics.model_scores["model2"] == 0.65  # (0.7 + 0.6) / 2

        # Test empty category
        empty_metrics = BenchmarkMetrics.calculate_category_performance(test_results, "nonexistent")
        assert empty_metrics.total_tests == 0
        assert empty_metrics.best_model == ""
        assert empty_metrics.average_score == 0.0

    def test_get_performance_tier(self):
        """Test performance tier classification."""
        assert BenchmarkMetrics.get_performance_tier(0.95) == "Excellent"
        assert BenchmarkMetrics.get_performance_tier(0.80) == "Good"
        assert BenchmarkMetrics.get_performance_tier(0.65) == "Fair"
        assert BenchmarkMetrics.get_performance_tier(0.50) == "Poor"
        assert BenchmarkMetrics.get_performance_tier(0.30) == "Very Poor"

    def test_calculate_improvement_percentage(self):
        """Test improvement percentage calculation."""
        assert BenchmarkMetrics.calculate_improvement_percentage(0.5, 0.6) == 20.0
        assert BenchmarkMetrics.calculate_improvement_percentage(0.8, 0.9) == 12.5
        assert BenchmarkMetrics.calculate_improvement_percentage(0.6, 0.6) == 0.0
        assert BenchmarkMetrics.calculate_improvement_percentage(0.8, 0.7) == -12.5  # Regression

        # Zero baseline
        assert BenchmarkMetrics.calculate_improvement_percentage(0.0, 0.5) == 0.0

    def test_aggregate_test_results(self):
        """Test aggregating multiple test results."""
        results = [
            BenchmarkTestResult("test1", "model", True, 0.9, 1.0, "out", "exp", cost=0.01),
            BenchmarkTestResult("test2", "model", True, 0.8, 2.0, "out", "exp", cost=0.02),
            BenchmarkTestResult(
                "test3",
                "model",
                False,
                0.3,
                1.5,
                "out",
                "exp",
                error="Failed",
                cost=0.01,
            ),
        ]

        aggregated = BenchmarkMetrics.aggregate_test_results(results)

        assert aggregated["total_tests"] == 3
        assert aggregated["passed_tests"] == 2
        assert aggregated["accuracy"] == 2 / 3
        assert aggregated["average_score"] == (0.9 + 0.8 + 0.3) / 3
        assert aggregated["average_duration"] == (1.0 + 2.0 + 1.5) / 3
        assert aggregated["total_duration"] == 4.5
        assert aggregated["total_cost"] == 0.04
        assert aggregated["error_rate"] == 1 / 3

        # Test empty results
        empty_aggregated = BenchmarkMetrics.aggregate_test_results([])
        assert empty_aggregated["total_tests"] == 0
        assert empty_aggregated["accuracy"] == 0.0
