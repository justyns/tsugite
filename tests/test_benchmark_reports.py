"""Tests for benchmark report generation."""

import json
from datetime import datetime

import pytest

from tsugite.benchmark.core import BenchmarkConfig, BenchmarkResult
from tsugite.benchmark.metrics import BenchmarkTestResult, ModelPerformance
from tsugite.benchmark.reports import ReportGenerator


@pytest.fixture
def sample_benchmark_result():
    """Create a sample benchmark result for testing."""
    config = BenchmarkConfig(
        models=["model1", "model2"],
        categories=["basic", "tools"],
        timeout=60,
    )

    start_time = datetime(2023, 1, 1, 12, 0, 0)
    end_time = datetime(2023, 1, 1, 12, 5, 0)

    model_performances = {
        "model1": ModelPerformance(
            model="model1",
            total_tests=4,
            passed_tests=3,
            accuracy=0.75,
            average_duration=2.0,
            total_duration=8.0,
            total_tokens=400,
            total_cost=0.02,
        ),
        "model2": ModelPerformance(
            model="model2",
            total_tests=4,
            passed_tests=4,
            accuracy=1.0,
            average_duration=3.0,
            total_duration=12.0,
            total_tokens=500,
            total_cost=0.03,
        ),
    }

    test_results = {
        "model1": {
            "basic_001": BenchmarkTestResult("basic_001", "model1", True, 0.9, 1.5, "42", "42"),
            "basic_002": BenchmarkTestResult("basic_002", "model1", False, 0.3, 2.0, "24", "42"),
            "tools_001": BenchmarkTestResult("tools_001", "model1", True, 0.8, 2.5, "success", "success"),
            "tools_002": BenchmarkTestResult("tools_002", "model1", True, 0.7, 2.0, "done", "done"),
        },
        "model2": {
            "basic_001": BenchmarkTestResult("basic_001", "model2", True, 1.0, 2.0, "42", "42"),
            "basic_002": BenchmarkTestResult("basic_002", "model2", True, 0.9, 3.0, "42", "42"),
            "tools_001": BenchmarkTestResult("tools_001", "model2", True, 1.0, 4.0, "success", "success"),
            "tools_002": BenchmarkTestResult("tools_002", "model2", True, 0.8, 3.0, "done", "done"),
        },
    }

    summary = {
        "total_models": 2,
        "total_tests": 4,
        "average_accuracy": 0.875,
        "best_model": "model2",
        "worst_model": "model1",
        "model_rankings": [
            {
                "model": "model2",
                "accuracy": 1.0,
                "avg_duration": 3.0,
                "total_cost": 0.03,
            },
            {
                "model": "model1",
                "accuracy": 0.75,
                "avg_duration": 2.0,
                "total_cost": 0.02,
            },
        ],
    }

    return BenchmarkResult(
        config=config,
        start_time=start_time,
        end_time=end_time,
        total_duration=300.0,
        model_performances=model_performances,
        test_results=test_results,
        summary=summary,
        errors=["Warning: model1 had one timeout"],
    )


class TestReportGenerator:
    """Test ReportGenerator functionality."""

    def test_json_report_generation(self, sample_benchmark_result, tmp_path):
        """Test JSON report generation."""
        generator = ReportGenerator(sample_benchmark_result)
        output_path = tmp_path / "test_report.json"

        generator.generate_json_report(output_path)

        assert output_path.exists()

        # Load and verify JSON content
        with open(output_path) as f:
            report_data = json.load(f)

        assert "benchmark_info" in report_data
        assert "config" in report_data
        assert "summary" in report_data
        assert "model_performances" in report_data
        assert "detailed_results" in report_data
        assert "errors" in report_data
        assert "generated_at" in report_data

        # Verify benchmark info
        benchmark_info = report_data["benchmark_info"]
        assert benchmark_info["total_duration"] == 300.0
        assert len(benchmark_info["models_tested"]) == 2
        assert benchmark_info["total_tests"] == 4

        # Verify model performances
        model_perfs = report_data["model_performances"]
        assert "model1" in model_perfs
        assert "model2" in model_perfs
        assert model_perfs["model1"]["accuracy"] == 0.75
        assert model_perfs["model2"]["accuracy"] == 1.0

        # Verify detailed results
        detailed = report_data["detailed_results"]
        assert "model1" in detailed
        assert "basic_001" in detailed["model1"]
        assert detailed["model1"]["basic_001"]["passed"] is True

        # Verify errors
        assert len(report_data["errors"]) == 1
        assert "timeout" in report_data["errors"][0]

    def test_markdown_report_generation(self, sample_benchmark_result, tmp_path):
        """Test Markdown report generation."""
        generator = ReportGenerator(sample_benchmark_result)
        output_path = tmp_path / "test_report.md"

        generator.generate_markdown_report(output_path)

        assert output_path.exists()

        content = output_path.read_text()

        # Verify key sections
        assert "# Tsugite Benchmark Report" in content
        assert "## Summary" in content
        assert "## Model Rankings" in content
        assert "## Detailed Performance" in content
        assert "## Test Results Details" in content
        assert "## Errors" in content

        # Verify summary content
        assert "Duration**: 300.00 seconds" in content
        assert "Models Tested**: 2" in content
        assert "Total Tests**: 4" in content
        assert "Average Accuracy**: 87.5%" in content

        # Verify model rankings table
        assert "| Rank | Model | Accuracy | Avg Duration | Total Cost |" in content
        assert "| 1 | model2 | 100.0% | 3.00s | $0.0300 |" in content
        assert "| 2 | model1 | 75.0% | 2.00s | $0.0200 |" in content

        # Verify detailed performance
        assert "### model1" in content
        assert "**Accuracy**: 75.0% (3/4)" in content
        assert "**Performance Tier**: Good" in content

        # Verify test details table
        assert "| Test ID | Category | Result | Score | Duration |" in content
        assert "| basic_001 | basic | ✅ PASS | 0.90 | 1.50s |" in content
        assert "| basic_002 | basic | ❌ FAIL | 0.30 | 2.00s |" in content

        # Verify errors section
        assert "timeout" in content

    def test_html_report_generation(self, sample_benchmark_result, tmp_path):
        """Test HTML report generation."""
        generator = ReportGenerator(sample_benchmark_result)
        output_path = tmp_path / "test_report.html"

        generator.generate_html_report(output_path)

        assert output_path.exists()

        content = output_path.read_text()

        # Verify HTML structure
        assert "<!DOCTYPE html>" in content
        assert '<html lang="en">' in content
        assert "<head>" in content
        assert "<body>" in content

        # Verify title and styling
        assert "<title>Tsugite Benchmark Report</title>" in content
        assert "<style>" in content
        assert "font-family:" in content

        # Verify main content
        assert "🚀 Tsugite Benchmark Report" in content
        assert "📊 Summary" in content
        assert "🏆 Model Rankings" in content
        assert "📈 Detailed Results" in content

        # Verify metrics
        assert "300.0s" in content  # Total duration
        assert "2" in content  # Models tested
        assert "4" in content  # Total tests
        assert "87.5%" in content  # Average accuracy

        # Verify performance tiers and CSS classes
        assert "tier-good" in content or "tier-excellent" in content
        assert "progress-bar" in content
        assert "progress-fill" in content

    def test_csv_summary_generation(self, sample_benchmark_result, tmp_path):
        """Test CSV summary generation."""
        generator = ReportGenerator(sample_benchmark_result)
        output_path = tmp_path / "test_report.csv"

        generator.generate_csv_summary(output_path)

        assert output_path.exists()

        content = output_path.read_text()
        lines = content.strip().split("\n")

        # Verify header
        header = lines[0]
        expected_cols = [
            "Model",
            "Test_ID",
            "Category",
            "Passed",
            "Score",
            "Duration",
            "Token_Usage",
            "Cost",
            "Error",
        ]
        for col in expected_cols:
            assert col in header

        # Verify data rows (should have 8 rows: 2 models × 4 tests each)
        assert len(lines) == 9  # 1 header + 8 data rows

        # Verify some specific data
        assert "model1,basic_001,basic,True,0.9,1.5,0,0.0," in content
        assert "model1,basic_002,basic,False,0.3,2.0,0,0.0," in content
        assert "model2,basic_001,basic,True,1.0,2.0,0,0.0," in content

    def test_get_test_category(self, sample_benchmark_result):
        """Test test category extraction."""
        generator = ReportGenerator(sample_benchmark_result)

        assert generator._get_test_category("basic_001") == "basic"
        assert generator._get_test_category("tools_002") == "tools"
        assert generator._get_test_category("scenarios_003") == "scenarios"
        assert generator._get_test_category("performance_004") == "performance"
        assert generator._get_test_category("unknown_test") == "unknown"

    def test_get_performance_tier(self, sample_benchmark_result):
        """Test performance tier classification."""
        generator = ReportGenerator(sample_benchmark_result)

        assert generator._get_performance_tier(0.95) == "Excellent"
        assert generator._get_performance_tier(0.80) == "Good"
        assert generator._get_performance_tier(0.65) == "Fair"
        assert generator._get_performance_tier(0.50) == "Poor"
        assert generator._get_performance_tier(0.30) == "Very Poor"

    def test_calculate_category_breakdown(self, sample_benchmark_result):
        """Test category performance breakdown calculation."""
        generator = ReportGenerator(sample_benchmark_result)

        breakdown = generator._calculate_category_breakdown()

        assert "basic" in breakdown
        assert "tools" in breakdown

        basic_breakdown = breakdown["basic"]
        assert basic_breakdown["best_model"] == "model2"
        assert basic_breakdown["best_score"] == 0.95  # (1.0 + 0.9) / 2
        assert "model_scores" in basic_breakdown
        assert basic_breakdown["model_scores"]["model1"] == 0.6  # (0.9 + 0.3) / 2
        assert basic_breakdown["model_scores"]["model2"] == 0.95  # (1.0 + 0.9) / 2

        tools_breakdown = breakdown["tools"]
        assert tools_breakdown["best_model"] == "model2"
        assert tools_breakdown["best_score"] == 0.9  # (1.0 + 0.8) / 2

    def test_tier_css_classes(self, sample_benchmark_result):
        """Test CSS class generation for performance tiers."""
        generator = ReportGenerator(sample_benchmark_result)

        assert generator._get_tier_class(0.95) == "tier-excellent"
        assert generator._get_tier_class(0.80) == "tier-good"
        assert generator._get_tier_class(0.65) == "tier-fair"
        assert generator._get_tier_class(0.50) == "tier-poor"
        assert generator._get_tier_class(0.30) == "tier-very-poor"

        assert generator._get_progress_class(0.95) == "progress-excellent"
        assert generator._get_progress_class(0.80) == "progress-good"
        assert generator._get_progress_class(0.65) == "progress-fair"
        assert generator._get_progress_class(0.50) == "progress-poor"
        assert generator._get_progress_class(0.30) == "progress-very-poor"

    def test_get_all_test_ids(self, sample_benchmark_result):
        """Test getting all unique test IDs."""
        generator = ReportGenerator(sample_benchmark_result)

        test_ids = generator._get_all_test_ids()

        expected_ids = {"basic_001", "basic_002", "tools_001", "tools_002"}
        assert set(test_ids) == expected_ids

    def test_empty_results_handling(self):
        """Test handling of empty benchmark results."""
        config = BenchmarkConfig()
        result = BenchmarkResult(
            config=config,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=0.0,
            model_performances={},
            test_results={},
            summary={},
        )

        generator = ReportGenerator(result)

        # Should not crash with empty data
        breakdown = generator._calculate_category_breakdown()
        assert breakdown == {}

        test_ids = generator._get_all_test_ids()
        assert test_ids == []
