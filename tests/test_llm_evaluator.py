"""Tests for LLM evaluation functionality."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from tsugite.benchmark.evaluators import LLMEvaluator


class TestLLMEvaluator:
    """Test the LLM evaluator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = LLMEvaluator(evaluator_model="openai:gpt-4o-mini")

    @pytest.mark.asyncio
    async def test_llm_evaluator_initialization(self):
        """Test LLM evaluator initialization."""
        assert self.evaluator.evaluator_model == "openai:gpt-4o-mini"

        # Test custom model
        custom_evaluator = LLMEvaluator(evaluator_model="claude-3-sonnet")
        assert custom_evaluator.evaluator_model == "claude-3-sonnet"

    @pytest.mark.asyncio
    @patch("tsugite.agent_runner.run_agent")
    async def test_llm_evaluation_success(self, mock_run_agent, mock_llm_evaluation_response):
        """Test successful LLM evaluation."""
        # Mock the agent runner to return JSON evaluation
        mock_run_agent.return_value = mock_llm_evaluation_response

        result = await self.evaluator.evaluate(
            output="This is a well-written response to the task.",
            task_description="Write a brief explanation",
            evaluation_criteria="accuracy, clarity, completeness",
        )

        assert result["llm_score"] == 0.85  # Normalized from 8.5/10
        assert "good understanding" in result["llm_feedback"].lower()
        assert result["evaluator_model"] == "openai:gpt-4o-mini"
        assert "accuracy" in result["criteria_breakdown"]
        assert result["criteria_breakdown"]["accuracy"] == 0.9  # Normalized

        # Verify agent runner was called correctly
        mock_run_agent.assert_called_once()
        call_args = mock_run_agent.call_args[1]
        assert "Evaluation Criteria" in call_args["prompt"]

    @pytest.mark.asyncio
    @patch("tsugite.agent_runner.run_agent")
    async def test_llm_evaluation_with_rubric(self, mock_run_agent):
        """Test LLM evaluation with detailed rubric."""
        mock_run_agent.return_value = """
        {
            "score": 7,
            "feedback": "Meets most requirements",
            "reasoning": "Good structure but could be more detailed",
            "criteria_breakdown": {
                "technical_accuracy": 8,
                "code_quality": 6
            },
            "assessment": "Satisfactory"
        }
        """

        rubric = {
            "technical_accuracy": "Code should be syntactically correct and follow best practices",
            "code_quality": "Code should be clean, readable, and well-commented",
        }

        result = await self.evaluator.evaluate(
            output="def hello(): print('hello')",
            task_description="Write a hello world function",
            evaluation_criteria="technical accuracy, code quality",
            expected_format="Python code",
            rubric=rubric,
        )

        assert result["llm_score"] == 0.7
        assert result["criteria_breakdown"]["technical_accuracy"] == 0.8
        assert result["criteria_breakdown"]["code_quality"] == 0.6

        # Check that rubric was included in prompt
        call_args = mock_run_agent.call_args[1]
        assert "Detailed Rubric" in call_args["prompt"]
        assert "technical_accuracy" in call_args["prompt"]

    @pytest.mark.asyncio
    @patch("tsugite.agent_runner.run_agent")
    async def test_llm_evaluation_json_in_code_block(self, mock_run_agent):
        """Test parsing JSON wrapped in code blocks."""
        mock_run_agent.return_value = """
Here's my evaluation:

```json
{
    "score": 6,
    "feedback": "Decent attempt but needs improvement",
    "reasoning": "Missing some key requirements",
    "criteria_breakdown": {
        "completeness": 5,
        "accuracy": 7
    },
    "assessment": "Needs work"
}
```

Additional comments here...
        """

        result = await self.evaluator.evaluate(
            output="Incomplete response", task_description="Complete task", evaluation_criteria="completeness, accuracy"
        )

        assert result["llm_score"] == 0.6
        assert result["criteria_breakdown"]["completeness"] == 0.5
        assert result["criteria_breakdown"]["accuracy"] == 0.7

    @pytest.mark.asyncio
    @patch("tsugite.agent_runner.run_agent")
    async def test_llm_evaluation_fallback_parsing(self, mock_run_agent):
        """Test fallback parsing when JSON extraction fails."""
        mock_run_agent.return_value = """
        I would rate this response a 7 out of 10. The answer shows understanding
        but lacks detail. Overall, it's a satisfactory response.
        """

        result = await self.evaluator.evaluate(
            output="Brief answer",
            task_description="Provide detailed explanation",
            evaluation_criteria="detail, accuracy",
        )

        assert result["llm_score"] == 0.7  # Should extract "7" and normalize
        assert "fallback parsing" in result["llm_reasoning"].lower()
        assert "satisfactory" in result["llm_feedback"]

    @pytest.mark.asyncio
    @patch("tsugite.agent_runner.run_agent")
    async def test_llm_evaluation_error_handling(self, mock_run_agent):
        """Test error handling in LLM evaluation."""
        mock_run_agent.side_effect = Exception("Model unavailable")

        result = await self.evaluator.evaluate(
            output="Test output", task_description="Test task", evaluation_criteria="quality"
        )

        assert result["llm_score"] == 0.0
        assert "evaluation failed" in result["llm_feedback"].lower()
        assert "Model unavailable" in result["error"]
        assert result["overall_assessment"] == "Error"

    def test_create_evaluation_prompt(self):
        """Test evaluation prompt creation."""
        prompt = self.evaluator._create_evaluation_prompt(
            output="Sample output",
            task_description="Sample task",
            evaluation_criteria="accuracy, clarity",
            expected_format="markdown",
            rubric={"accuracy": "Must be factually correct"},
        )

        assert "Sample output" in prompt
        assert "Sample task" in prompt
        assert "accuracy, clarity" in prompt
        assert "Expected Format" in prompt
        assert "markdown" in prompt
        assert "Detailed Rubric" in prompt
        assert "Must be factually correct" in prompt
        assert "JSON object" in prompt

    def test_create_evaluator_agent(self):
        """Test evaluator agent creation."""
        agent_content = self.evaluator._create_evaluator_agent("test-model")

        assert "test-model" in agent_content
        assert "LLM Evaluator Agent" in agent_content
        assert "Accuracy" in agent_content
        assert "user_prompt" in agent_content

    def test_parse_evaluation_result_valid_json(self):
        """Test parsing valid JSON evaluation result."""
        json_result = """
        {
            "score": 8,
            "feedback": "Good work",
            "reasoning": "Well structured",
            "criteria_breakdown": {"quality": 8},
            "assessment": "Good"
        }
        """

        result = self.evaluator._parse_evaluation_result(json_result)

        assert result["score"] == 0.8  # Normalized
        assert result["feedback"] == "Good work"
        assert result["criteria_breakdown"]["quality"] == 0.8

    def test_parse_evaluation_result_score_normalization(self):
        """Test score normalization in different ranges."""
        # Test 0-10 range normalization
        result_10_scale = self.evaluator._parse_evaluation_result('{"score": 7.5, "feedback": "test"}')
        assert result_10_scale["score"] == 0.75

        # Test 0-1 range (no normalization needed)
        result_1_scale = self.evaluator._parse_evaluation_result('{"score": 0.85, "feedback": "test"}')
        assert result_1_scale["score"] == 0.85

        # Test out of bounds (should be clamped)
        result_high = self.evaluator._parse_evaluation_result('{"score": 15, "feedback": "test"}')
        assert result_high["score"] == 1.0

    def test_fallback_parse(self):
        """Test fallback parsing for non-JSON responses."""
        # Test score extraction
        result = self.evaluator._fallback_parse("I would score this 8.5 out of 10")
        assert result["score"] == 0.85

        # Test with percentage
        result_pct = self.evaluator._fallback_parse("Rating: 75%")
        assert result_pct["score"] == 0.75

        # Test with no score found
        result_no_score = self.evaluator._fallback_parse("This is good work")
        assert result_no_score["score"] == 0.5  # Default middle score

    @pytest.mark.asyncio
    async def test_integration_with_benchmark_config(self):
        """Test integration with benchmark configuration."""
        from tsugite.benchmark.core import BenchmarkConfig

        config = BenchmarkConfig(llm_evaluator_model="claude-3-haiku")

        # The evaluator should be initialized with the config model
        # This would be tested in the benchmark runner integration
        assert config.llm_evaluator_model == "claude-3-haiku"


class TestLLMEvaluatorPrompts:
    """Test LLM evaluator prompt generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = LLMEvaluator()

    def test_basic_prompt_generation(self):
        """Test basic evaluation prompt generation."""
        prompt = self.evaluator._create_evaluation_prompt(
            output="Hello world", task_description="Write a greeting", evaluation_criteria="friendliness, clarity"
        )

        assert "Hello world" in prompt
        assert "Write a greeting" in prompt
        assert "friendliness, clarity" in prompt
        assert "JSON object" in prompt
        assert "0-10" in prompt

    def test_prompt_with_all_options(self):
        """Test prompt generation with all optional parameters."""
        rubric = {"friendliness": "Should be warm and welcoming", "clarity": "Should be easy to understand"}

        prompt = self.evaluator._create_evaluation_prompt(
            output="Hello there, how are you?",
            task_description="Write a friendly greeting",
            evaluation_criteria="friendliness, clarity, length",
            expected_format="conversational text",
            rubric=rubric,
        )

        assert "Hello there, how are you?" in prompt
        assert "Write a friendly greeting" in prompt
        assert "friendliness, clarity, length" in prompt
        assert "Expected Format" in prompt
        assert "conversational text" in prompt
        assert "Detailed Rubric" in prompt
        assert "warm and welcoming" in prompt
        assert "easy to understand" in prompt

    def test_prompt_structure(self):
        """Test that prompt has proper structure."""
        prompt = self.evaluator._create_evaluation_prompt(
            output="Test", task_description="Test task", evaluation_criteria="quality"
        )

        # Check for required sections
        assert "## Task Description" in prompt
        assert "## Evaluation Criteria" in prompt
        assert "## Agent's Output" in prompt
        assert "## Instructions" in prompt
        assert "## Required Response Format" in prompt


class TestLLMEvaluatorEdgeCases:
    """Test edge cases for LLM evaluator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = LLMEvaluator()

    @pytest.mark.asyncio
    @patch("tsugite.agent_runner.run_agent")
    async def test_empty_output_evaluation(self, mock_run_agent):
        """Test evaluation of empty output."""
        mock_run_agent.return_value = '{"score": 0, "feedback": "No output provided", "reasoning": "Empty response", "criteria_breakdown": {}, "assessment": "Failed"}'

        result = await self.evaluator.evaluate(
            output="", task_description="Provide an explanation", evaluation_criteria="completeness"
        )

        assert result["llm_score"] == 0.0
        assert "no output" in result["llm_feedback"].lower()

    @pytest.mark.asyncio
    @patch("tsugite.agent_runner.run_agent")
    async def test_very_long_output_evaluation(self, mock_run_agent):
        """Test evaluation of very long output."""
        long_output = "This is a very long response. " * 1000  # ~30k characters

        mock_run_agent.return_value = '{"score": 5, "feedback": "Too verbose", "reasoning": "Excessive length", "criteria_breakdown": {"conciseness": 2}, "assessment": "Needs editing"}'

        result = await self.evaluator.evaluate(
            output=long_output, task_description="Provide a brief summary", evaluation_criteria="conciseness, clarity"
        )

        assert result["llm_score"] == 0.5
        assert result["criteria_breakdown"]["conciseness"] == 0.2

    def test_malformed_json_parsing(self):
        """Test parsing of malformed JSON responses."""
        malformed_responses = [
            '{"score": 8, "feedback": "Good" incomplete',
            '{"score": 8.5 "feedback": "Missing comma"}',
            'score: 7.5, feedback: "Not JSON format"',
            '{"score": "eight", "feedback": "Non-numeric score"}',
        ]

        for response in malformed_responses:
            result = self.evaluator._parse_evaluation_result(response)
            # Should fall back to fallback parsing
            assert "score" in result
            assert "feedback" in result
            assert isinstance(result["score"], float)
            assert 0.0 <= result["score"] <= 1.0
