"""Tests for the benchmark core functionality."""

import pytest
import asyncio
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from tsugite.benchmark.core import (
    BenchmarkRunner,
    BenchmarkConfig,
    BenchmarkTest,
    BenchmarkResult,
)
from tsugite.benchmark.metrics import TestResult, ModelPerformance


@pytest.fixture
def temp_benchmark_dir(tmp_path):
    """Create a temporary benchmark directory structure."""
    benchmark_dir = tmp_path / "benchmarks"
    basic_dir = benchmark_dir / "basic"
    basic_dir.mkdir(parents=True)

    # Create a simple test agent
    test_agent = basic_dir / "test_simple.md"
    test_agent.write_text(
        """---
name: test_simple
test_id: test_001
model: "{{ model }}"
expected_output: "42"
expected_type: "number"
timeout: 10
weight: 1.0
---

# Task
Output: 42
"""
    )

    return benchmark_dir


@pytest.fixture
def benchmark_config():
    """Create a test benchmark configuration."""
    return BenchmarkConfig(
        models=["test-model:v1"],
        categories=["basic"],
        timeout=30,
        parallel=False,
        output_dir=Path("test_output"),
    )


@pytest.fixture
def mock_agent_run():
    """Mock the agent runner."""
    with patch("tsugite.benchmark.core.run_agent") as mock:
        mock.return_value = "42"
        yield mock


class TestBenchmarkConfig:
    """Test BenchmarkConfig functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BenchmarkConfig()

        assert config.models == []
        assert config.categories == ["basic", "tools", "scenarios", "performance"]
        assert config.timeout == 120
        assert config.parallel is True
        assert config.temperature == 0.1
        assert config.max_tokens == 2000

    def test_custom_config(self):
        """Test custom configuration."""
        config = BenchmarkConfig(
            models=["model1", "model2"],
            categories=["basic"],
            timeout=60,
            parallel=False,
        )

        assert config.models == ["model1", "model2"]
        assert config.categories == ["basic"]
        assert config.timeout == 60
        assert config.parallel is False


class TestBenchmarkTest:
    """Test BenchmarkTest functionality."""

    def test_benchmark_test_creation(self):
        """Test creating a benchmark test."""
        test = BenchmarkTest(
            name="test_simple",
            agent_path=Path("test.md"),
            test_id="test_001",
            category="basic",
            expected_output="42",
            expected_type="number",
        )

        assert test.name == "test_simple"
        assert test.test_id == "test_001"
        assert test.category == "basic"
        assert test.expected_output == "42"
        assert test.expected_type == "number"
        assert test.timeout == 60  # default
        assert test.weight == 1.0  # default


class TestBenchmarkRunner:
    """Test BenchmarkRunner functionality."""

    def test_runner_initialization(self, benchmark_config):
        """Test benchmark runner initialization."""
        runner = BenchmarkRunner(benchmark_config)

        assert runner.config == benchmark_config
        assert runner.benchmark_dir == Path("benchmarks")
        assert runner.test_cache == {}

    def test_discover_tests(self, benchmark_config, temp_benchmark_dir, monkeypatch):
        """Test test discovery functionality."""
        # Patch the benchmark directory
        monkeypatch.setattr("tsugite.benchmark.core.BenchmarkRunner.benchmark_dir", temp_benchmark_dir)

        runner = BenchmarkRunner(benchmark_config)
        tests = runner.discover_tests(["basic"])

        assert len(tests) == 1
        test = tests[0]
        assert test.name == "test_simple"
        assert test.test_id == "test_001"
        assert test.category == "basic"

    def test_parse_benchmark_test(self, benchmark_config, temp_benchmark_dir, monkeypatch):
        """Test parsing individual benchmark test."""
        monkeypatch.setattr("tsugite.benchmark.core.BenchmarkRunner.benchmark_dir", temp_benchmark_dir)

        runner = BenchmarkRunner(benchmark_config)
        agent_path = temp_benchmark_dir / "basic" / "test_simple.md"

        test = runner._parse_benchmark_test(agent_path, "basic")

        assert test.name == "test_simple"
        assert test.test_id == "test_001"
        assert test.expected_output == "42"
        assert test.expected_type == "number"
        assert test.timeout == 10
        assert test.weight == 1.0

    @pytest.mark.asyncio
    async def test_run_single_test(self, benchmark_config, temp_benchmark_dir, mock_agent_run, monkeypatch):
        """Test running a single test."""
        monkeypatch.setattr("tsugite.benchmark.core.BenchmarkRunner.benchmark_dir", temp_benchmark_dir)

        runner = BenchmarkRunner(benchmark_config)
        agent_path = temp_benchmark_dir / "basic" / "test_simple.md"
        test = runner._parse_benchmark_test(agent_path, "basic")

        result = await runner._run_single_test("test-model:v1", test)

        assert isinstance(result, TestResult)
        assert result.test_id == "test_001"
        assert result.model == "test-model:v1"
        assert result.output == "42"
        assert result.duration > 0

    @pytest.mark.asyncio
    async def test_evaluate_test_result(self, benchmark_config):
        """Test result evaluation."""
        runner = BenchmarkRunner(benchmark_config)

        test = BenchmarkTest(
            name="test_math",
            agent_path=Path("test.md"),
            test_id="test_001",
            category="basic",
            expected_output="42",
            expected_type="number",
        )

        # Test correct output
        evaluation = await runner._evaluate_test_result(test, "42", 1.0)
        assert evaluation["passed"] is True
        assert evaluation["score"] == 1.0

        # Test incorrect output
        evaluation = await runner._evaluate_test_result(test, "24", 1.0)
        assert evaluation["passed"] is False
        assert evaluation["score"] < 1.0

    @pytest.mark.asyncio
    async def test_run_benchmark_integration(self, benchmark_config, temp_benchmark_dir, mock_agent_run, monkeypatch):
        """Test full benchmark run integration."""
        monkeypatch.setattr("tsugite.benchmark.core.BenchmarkRunner.benchmark_dir", temp_benchmark_dir)

        runner = BenchmarkRunner(benchmark_config)

        result = await runner.run_benchmark(models=["test-model:v1"], categories=["basic"])

        assert isinstance(result, BenchmarkResult)
        assert result.config == benchmark_config
        assert len(result.model_performances) == 1
        assert "test-model:v1" in result.model_performances

        performance = result.model_performances["test-model:v1"]
        assert isinstance(performance, ModelPerformance)
        assert performance.total_tests == 1
        assert performance.model == "test-model:v1"

    def test_generate_summary(self, benchmark_config):
        """Test summary generation."""
        runner = BenchmarkRunner(benchmark_config)

        # Mock data
        model_performances = {
            "model1": ModelPerformance(
                model="model1",
                total_tests=2,
                passed_tests=2,
                accuracy=1.0,
                average_duration=1.0,
                total_duration=2.0,
                total_tokens=100,
                total_cost=0.01,
            ),
            "model2": ModelPerformance(
                model="model2",
                total_tests=2,
                passed_tests=1,
                accuracy=0.5,
                average_duration=2.0,
                total_duration=4.0,
                total_tokens=200,
                total_cost=0.02,
            ),
        }

        test_results = {}

        summary = runner._generate_summary(model_performances, test_results)

        assert summary["total_models"] == 2
        assert summary["best_model"] == "model1"
        assert summary["worst_model"] == "model2"
        assert summary["average_accuracy"] == 0.75  # (1.0 + 0.5) / 2

        rankings = summary["model_rankings"]
        assert len(rankings) == 2
        assert rankings[0]["model"] == "model1"  # Best first
        assert rankings[1]["model"] == "model2"


class TestBenchmarkResult:
    """Test BenchmarkResult functionality."""

    def test_benchmark_result_creation(self, benchmark_config):
        """Test creating benchmark result."""
        start_time = datetime.now()
        end_time = datetime.now()

        result = BenchmarkResult(
            config=benchmark_config,
            start_time=start_time,
            end_time=end_time,
            total_duration=10.0,
            model_performances={},
            test_results={},
            summary={},
        )

        assert result.config == benchmark_config
        assert result.start_time == start_time
        assert result.end_time == end_time
        assert result.total_duration == 10.0
        assert result.errors == []  # default


@pytest.mark.asyncio
async def test_benchmark_error_handling(benchmark_config, temp_benchmark_dir, monkeypatch):
    """Test error handling in benchmark execution."""
    monkeypatch.setattr("tsugite.benchmark.core.BenchmarkRunner.benchmark_dir", temp_benchmark_dir)

    # Mock run_agent to raise an exception
    with patch("tsugite.benchmark.core.run_agent") as mock_run:
        mock_run.side_effect = Exception("Test error")

        runner = BenchmarkRunner(benchmark_config)
        result = await runner.run_benchmark(models=["test-model:v1"], categories=["basic"])

        # Should handle error gracefully
        assert len(result.errors) > 0
        assert "test-model:v1" in result.model_performances

        # Test result should show failure
        model_results = result.test_results["test-model:v1"]
        test_result = list(model_results.values())[0]
        assert test_result.passed is False
        assert test_result.error is not None


def test_benchmark_with_no_tests(benchmark_config):
    """Test benchmark behavior with no tests found."""
    runner = BenchmarkRunner(benchmark_config)

    # Should raise error when no tests found
    with pytest.raises(ValueError, match="No tests found"):
        asyncio.run(runner.run_benchmark(models=["test-model:v1"], categories=["nonexistent"]))


def test_benchmark_with_no_models(benchmark_config):
    """Test benchmark behavior with no models specified."""
    runner = BenchmarkRunner(benchmark_config)

    # Should raise error when no models specified
    with pytest.raises(ValueError, match="No models specified"):
        asyncio.run(runner.run_benchmark(models=[], categories=["basic"]))
