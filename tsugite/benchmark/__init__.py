"""Tsugite benchmark framework for evaluating agent performance across different models."""

from .config import (
    COST_TIERS,
    EVALUATION_WEIGHTS,
    MODEL_COSTS,
    PERFORMANCE_TIERS,
    SIMILARITY_THRESHOLDS,
    TEST_CATEGORIES,
    get_cost_tier,
    get_performance_tier,
)
from .core import BenchmarkConfig, BenchmarkResult, BenchmarkRunner
from .discovery import BenchmarkTest, TestCase, TestDiscovery
from .evaluators import (
    CorrectnessEvaluator,
    CostEvaluator,
    LLMEvaluator,
    PerformanceEvaluator,
    QualityEvaluator,
)
from .execution import TestExecutor
from .metrics import (
    BenchmarkMetrics,
    BenchmarkTestResult,
    ModelPerformance,
)
from .reports import ReportGenerator

__all__ = [
    # Core
    "BenchmarkRunner",
    "BenchmarkResult",
    "BenchmarkConfig",
    # Discovery
    "TestDiscovery",
    "BenchmarkTest",
    "TestCase",
    # Execution
    "TestExecutor",
    # Evaluators
    "CorrectnessEvaluator",
    "PerformanceEvaluator",
    "QualityEvaluator",
    "CostEvaluator",
    "LLMEvaluator",
    # Metrics
    "BenchmarkMetrics",
    "BenchmarkTestResult",
    "ModelPerformance",
    # Reports
    "ReportGenerator",
    # Config
    "SIMILARITY_THRESHOLDS",
    "PERFORMANCE_TIERS",
    "EVALUATION_WEIGHTS",
    "MODEL_COSTS",
    "COST_TIERS",
    "TEST_CATEGORIES",
    "get_performance_tier",
    "get_cost_tier",
]
