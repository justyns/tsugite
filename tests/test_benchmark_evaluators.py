"""Tests for benchmark evaluators."""

import pytest

from tsugite.benchmark.evaluators import (
    CorrectnessEvaluator,
    CostEvaluator,
    PerformanceEvaluator,
    QualityEvaluator,
)


class TestCorrectnessEvaluator:
    """Test CorrectnessEvaluator functionality."""

    def setup_method(self):
        """Set up evaluator for each test."""
        self.evaluator = CorrectnessEvaluator()

    def test_string_evaluation_exact_match(self):
        """Test string evaluation with exact match."""
        result = self.evaluator.evaluate(output="Hello World", expected="Hello World", output_type="string")

        assert result["passed"] is True
        assert result["score"] == 1.0
        assert result["exact_match"] is True
        assert result["similarity"] == 1.0

    def test_string_evaluation_high_similarity(self):
        """Test string evaluation with high similarity."""
        result = self.evaluator.evaluate(output="Hello World!", expected="Hello World", output_type="string")

        assert result["passed"] is True  # Should pass with high similarity
        assert result["score"] > 0.9
        assert result["exact_match"] is False
        assert result["similarity"] > 0.9

    def test_string_evaluation_low_similarity(self):
        """Test string evaluation with low similarity."""
        result = self.evaluator.evaluate(output="Goodbye Moon", expected="Hello World", output_type="string")

        assert result["passed"] is False
        assert result["score"] < 0.5
        assert result["exact_match"] is False
        assert result["similarity"] < 0.5

    def test_number_evaluation_exact(self):
        """Test number evaluation with exact match."""
        result = self.evaluator.evaluate(output="42", expected="42", output_type="number")

        assert result["passed"] is True
        assert result["score"] == 1.0
        assert result["exact_match"] is True

    def test_number_evaluation_close(self):
        """Test number evaluation with close values."""
        result = self.evaluator.evaluate(output="42.01", expected="42.00", output_type="number")

        assert result["passed"] is True  # Should pass within tolerance
        assert result["score"] > 0.99
        assert result["exact_match"] is False

    def test_number_evaluation_different(self):
        """Test number evaluation with different values."""
        result = self.evaluator.evaluate(output="100", expected="42", output_type="number")

        assert result["passed"] is False
        assert result["score"] < 0.1
        assert result["exact_match"] is False

    def test_json_evaluation_exact(self):
        """Test JSON evaluation with exact match."""
        result = self.evaluator.evaluate(
            output='{"name": "test", "value": 42}',
            expected='{"name": "test", "value": 42}',
            output_type="json",
        )

        assert result["passed"] is True
        assert result["score"] == 1.0
        assert result["exact_match"] is True

    def test_json_evaluation_different_order(self):
        """Test JSON evaluation with different key order."""
        result = self.evaluator.evaluate(
            output='{"value": 42, "name": "test"}',
            expected='{"name": "test", "value": 42}',
            output_type="json",
        )

        assert result["passed"] is True
        assert result["score"] == 1.0
        assert result["exact_match"] is True

    def test_json_evaluation_invalid_json(self):
        """Test JSON evaluation with invalid JSON."""
        result = self.evaluator.evaluate(
            output='{"name": "test", "value":}',  # Invalid JSON
            expected='{"name": "test", "value": 42}',
            output_type="json",
        )

        assert result["passed"] is False
        assert result["score"] == 0.0
        assert "JSON decode error" in result.get("error", "")

    def test_code_evaluation(self):
        """Test code evaluation."""
        result = self.evaluator.evaluate(
            output="def hello():\n    return 'world'",
            expected="def hello():\n    return 'world'",
            output_type="code",
        )

        assert result["passed"] is True
        assert result["score"] == 1.0

    def test_code_evaluation_different_formatting(self):
        """Test code evaluation with different formatting."""
        result = self.evaluator.evaluate(
            output="def hello():\n    return 'world'",
            expected="def hello(): return 'world'",
            output_type="code",
        )

        # Should still pass with high similarity despite formatting differences
        assert result["score"] > 0.8


class TestPerformanceEvaluator:
    """Test PerformanceEvaluator functionality."""

    def setup_method(self):
        """Set up evaluator for each test."""
        self.evaluator = PerformanceEvaluator()

    def test_fast_execution(self):
        """Test evaluation of fast execution."""
        result = self.evaluator.evaluate(duration=1.0, timeout=10.0)

        assert result["timed_out"] is False
        assert result["speed_score"] > 0.9
        assert result["efficiency_tier"] == "Excellent"

    def test_slow_execution(self):
        """Test evaluation of slow execution."""
        result = self.evaluator.evaluate(duration=8.0, timeout=10.0)

        assert result["timed_out"] is False
        assert result["speed_score"] < 0.3
        assert result["efficiency_tier"] == "Poor"

    def test_timeout_execution(self):
        """Test evaluation of timed out execution."""
        result = self.evaluator.evaluate(duration=15.0, timeout=10.0)

        assert result["timed_out"] is True
        assert result["speed_score"] == 0.0
        assert result["efficiency_tier"] == "Timeout"

    def test_baseline_comparison(self):
        """Test performance comparison with baseline."""
        result = self.evaluator.evaluate(duration=2.0, timeout=10.0, baseline_duration=4.0)

        assert result["improvement_over_baseline"] == 0.5  # 50% improvement
        assert result["relative_speed"] == 2.0  # 2x faster


class TestQualityEvaluator:
    """Test QualityEvaluator functionality."""

    def setup_method(self):
        """Set up evaluator for each test."""
        self.evaluator = QualityEvaluator()

    @pytest.mark.asyncio
    async def test_keyword_evaluation(self):
        """Test keyword-based quality evaluation."""
        criteria = {
            "completeness": {
                "type": "keyword",
                "keywords": ["complete", "done"],
                "weight": 1.0,
            }
        }

        result = await self.evaluator.evaluate(output="The task is complete and done successfully.", criteria=criteria)

        assert result["score"] == 1.0
        assert result["overall_quality"] == "Excellent"
        assert result["criteria_scores"]["completeness"] == 1.0

    @pytest.mark.asyncio
    async def test_length_evaluation(self):
        """Test length-based quality evaluation."""
        criteria = {
            "appropriate_length": {
                "type": "length",
                "min_length": 10,
                "max_length": 100,
                "weight": 1.0,
            }
        }

        # Good length
        result = await self.evaluator.evaluate(output="This is a good length response for testing.", criteria=criteria)

        assert result["score"] == 1.0

        # Too short
        result = await self.evaluator.evaluate(output="Short", criteria=criteria)

        assert result["score"] == 0.0

    @pytest.mark.asyncio
    async def test_format_evaluation_json(self):
        """Test format evaluation for JSON."""
        criteria = {"json_format": {"type": "format", "format": "json", "weight": 1.0}}

        # Valid JSON
        result = await self.evaluator.evaluate(output='{"key": "value"}', criteria=criteria)

        assert result["score"] == 1.0

        # Invalid JSON
        result = await self.evaluator.evaluate(output="not json", criteria=criteria)

        assert result["score"] == 0.0

    @pytest.mark.asyncio
    async def test_format_evaluation_code(self):
        """Test format evaluation for code."""
        criteria = {"code_format": {"type": "format", "format": "code", "weight": 1.0}}

        # Valid code
        result = await self.evaluator.evaluate(output="def hello(): return 'world'", criteria=criteria)

        assert result["score"] == 1.0

        # Plain text
        result = await self.evaluator.evaluate(output="just some text", criteria=criteria)

        assert result["score"] == 0.5

    @pytest.mark.asyncio
    async def test_sentiment_evaluation(self):
        """Test sentiment-based evaluation."""
        criteria = {
            "positive_tone": {
                "type": "sentiment",
                "sentiment": "positive",
                "weight": 1.0,
            }
        }

        # Positive text
        result = await self.evaluator.evaluate(output="This is a great and wonderful solution!", criteria=criteria)

        assert result["score"] == 1.0

        # Negative text
        result = await self.evaluator.evaluate(output="This is terrible and awful.", criteria=criteria)

        assert result["score"] == 0.5

    @pytest.mark.asyncio
    async def test_multiple_criteria(self):
        """Test evaluation with multiple criteria."""
        criteria = {
            "keywords": {"type": "keyword", "keywords": ["complete"], "weight": 0.5},
            "length": {
                "type": "length",
                "min_length": 10,
                "max_length": 100,
                "weight": 0.5,
            },
        }

        result = await self.evaluator.evaluate(output="The task is complete and finished.", criteria=criteria)

        # Should score well on both criteria
        assert result["score"] == 1.0
        assert len(result["criteria_scores"]) == 2


class TestCostEvaluator:
    """Test CostEvaluator functionality."""

    def setup_method(self):
        """Set up evaluator for each test."""
        self.evaluator = CostEvaluator()

    def test_cost_evaluation_gpt4(self):
        """Test cost evaluation for GPT-4."""
        token_usage = {"input": 100, "output": 50, "total": 150}

        result = self.evaluator.evaluate(token_usage=token_usage, model="openai:gpt-4", duration=2.0)

        assert result["estimated_cost"] > 0
        assert result["cost_per_token"] > 0
        assert result["tokens_per_second"] == 75.0  # 150 tokens / 2 seconds
        assert result["cost_efficiency_tier"] in ["Very Low", "Low", "Medium"]

    def test_cost_evaluation_ollama(self):
        """Test cost evaluation for Ollama (free)."""
        token_usage = {"input": 100, "output": 50, "total": 150}

        result = self.evaluator.evaluate(token_usage=token_usage, model="ollama:llama2", duration=2.0)

        assert result["estimated_cost"] == 0.0
        assert result["cost_per_token"] == 0.0
        assert result["cost_efficiency_tier"] == "Free"

    def test_cost_evaluation_unknown_model(self):
        """Test cost evaluation for unknown model."""
        token_usage = {"input": 100, "output": 50, "total": 150}

        result = self.evaluator.evaluate(token_usage=token_usage, model="unknown:model", duration=2.0)

        # Should use default estimate
        assert result["estimated_cost"] > 0
        assert result["cost_per_token"] == 0.00001  # Default estimate

    def test_cost_tiers(self):
        """Test cost tier classification."""
        from tsugite.benchmark.config import get_cost_tier

        assert get_cost_tier(0.0) == "Free"
        assert get_cost_tier(0.0001) == "Very Low"
        assert get_cost_tier(0.005) == "Low"
        assert get_cost_tier(0.05) == "Medium"
        assert get_cost_tier(0.5) == "High"
        assert get_cost_tier(5.0) == "Very High"
