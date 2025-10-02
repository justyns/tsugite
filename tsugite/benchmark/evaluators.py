"""Evaluators for different aspects of benchmark test results."""

import difflib
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseEvaluator(ABC):
    """Base class for all evaluators."""

    @abstractmethod
    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """Evaluate the given inputs and return a score and metrics."""
        pass


class CorrectnessEvaluator(BaseEvaluator):
    """Evaluates correctness of outputs against expected results."""

    def evaluate(self, output: str, expected: str, output_type: str = "string") -> Dict[str, Any]:
        """Evaluate correctness based on output type."""
        result = {
            "passed": False,
            "score": 0.0,
            "similarity": 0.0,
            "exact_match": False,
            "error": None,
        }

        try:
            if output_type == "string":
                result.update(self._evaluate_string(output, expected))
            elif output_type == "json":
                result.update(self._evaluate_json(output, expected))
            elif output_type == "code":
                result.update(self._evaluate_code(output, expected))
            elif output_type == "number":
                result.update(self._evaluate_number(output, expected))
            else:
                result.update(self._evaluate_string(output, expected))

        except Exception as e:
            result["error"] = str(e)

        return result

    def _evaluate_string(self, output: str, expected: str) -> Dict[str, Any]:
        """Evaluate string output."""
        output_clean = output.strip()
        expected_clean = expected.strip()

        exact_match = output_clean == expected_clean
        similarity = difflib.SequenceMatcher(None, output_clean.lower(), expected_clean.lower()).ratio()

        # Consider it passed if exact match or very high similarity
        passed = exact_match or similarity >= 0.9

        return {
            "passed": passed,
            "score": 1.0 if exact_match else similarity,
            "similarity": similarity,
            "exact_match": exact_match,
        }

    def _evaluate_json(self, output: str, expected: str) -> Dict[str, Any]:
        """Evaluate JSON output."""
        try:
            output_json = json.loads(output.strip())
            expected_json = json.loads(expected.strip())

            exact_match = output_json == expected_json
            score = 1.0 if exact_match else self._json_similarity(output_json, expected_json)

            return {
                "passed": exact_match or score >= 0.8,
                "score": score,
                "similarity": score,
                "exact_match": exact_match,
            }

        except json.JSONDecodeError as e:
            return {
                "passed": False,
                "score": 0.0,
                "similarity": 0.0,
                "exact_match": False,
                "error": f"JSON decode error: {e}",
            }

    def _evaluate_code(self, output: str, expected: str) -> Dict[str, Any]:
        """Evaluate code output (simplified)."""
        # Normalize whitespace and remove comments
        output_normalized = self._normalize_code(output)
        expected_normalized = self._normalize_code(expected)

        exact_match = output_normalized == expected_normalized
        similarity = difflib.SequenceMatcher(None, output_normalized, expected_normalized).ratio()

        return {
            "passed": exact_match or similarity >= 0.85,
            "score": 1.0 if exact_match else similarity,
            "similarity": similarity,
            "exact_match": exact_match,
        }

    def _evaluate_number(self, output: str, expected: str) -> Dict[str, Any]:
        """Evaluate numeric output."""
        try:
            # Extract numbers from strings
            output_nums = re.findall(r"-?\d+\.?\d*", output.strip())
            expected_nums = re.findall(r"-?\d+\.?\d*", expected.strip())

            if not output_nums or not expected_nums:
                return {
                    "passed": False,
                    "score": 0.0,
                    "similarity": 0.0,
                    "exact_match": False,
                    "error": "No numbers found",
                }

            output_val = float(output_nums[0])
            expected_val = float(expected_nums[0])

            exact_match = abs(output_val - expected_val) < 1e-10
            relative_error = abs(output_val - expected_val) / max(abs(expected_val), 1e-10)
            score = max(0.0, 1.0 - relative_error)

            return {
                "passed": exact_match or relative_error < 0.01,
                "score": score,
                "similarity": score,
                "exact_match": exact_match,
                "relative_error": relative_error,
            }

        except (ValueError, IndexError) as e:
            return {
                "passed": False,
                "score": 0.0,
                "similarity": 0.0,
                "exact_match": False,
                "error": f"Number evaluation error: {e}",
            }

    def _json_similarity(self, obj1: Any, obj2: Any) -> float:
        """Calculate similarity between JSON objects."""
        if type(obj1) is not type(obj2):
            return 0.0

        if isinstance(obj1, dict):
            if not obj1 and not obj2:
                return 1.0

            all_keys = set(obj1.keys()) | set(obj2.keys())
            if not all_keys:
                return 1.0

            key_scores = []
            for key in all_keys:
                if key in obj1 and key in obj2:
                    key_scores.append(self._json_similarity(obj1[key], obj2[key]))
                else:
                    key_scores.append(0.0)

            return sum(key_scores) / len(key_scores)

        elif isinstance(obj1, list):
            if len(obj1) != len(obj2):
                return 0.5 if obj1 == obj2 else 0.0

            if not obj1:
                return 1.0

            scores = [self._json_similarity(a, b) for a, b in zip(obj1, obj2)]
            return sum(scores) / len(scores)

        else:
            return 1.0 if obj1 == obj2 else 0.0

    def _normalize_code(self, code: str) -> str:
        """Normalize code for comparison."""
        # Remove comments and normalize whitespace
        lines = []
        for line in code.split("\n"):
            # Remove comments (simplified)
            line = re.sub(r"#.*$", "", line)
            line = re.sub(r"//.*$", "", line)
            line = line.strip()
            if line:
                lines.append(line)

        return "\n".join(lines)


class PerformanceEvaluator(BaseEvaluator):
    """Evaluates performance metrics like speed and efficiency."""

    def evaluate(self, duration: float, timeout: float, baseline_duration: Optional[float] = None) -> Dict[str, Any]:
        """Evaluate performance based on duration."""
        result = {
            "duration": duration,
            "timeout": timeout,
            "timed_out": duration >= timeout,
            "speed_score": 0.0,
            "efficiency_tier": "unknown",
        }

        # Calculate speed score (inverse of duration, normalized)
        if timeout > 0:
            # Score based on how much of the timeout was used
            time_ratio = duration / timeout
            speed_score = max(0.0, 1.0 - min(time_ratio, 1.0))

            # Reward significantly faster executions with a small bonus
            if time_ratio <= 0.1:
                speed_score = min(1.0, speed_score + 0.05)
            elif time_ratio <= 0.3:
                speed_score = min(1.0, speed_score + 0.02)
        else:
            speed_score = 1.0 if duration < 1.0 else max(0.0, 1.0 / duration)

        result["speed_score"] = speed_score

        # Efficiency tiers
        if duration <= timeout * 0.1:
            tier = "Excellent"
        elif duration <= timeout * 0.3:
            tier = "Good"
        elif duration <= timeout * 0.6:
            tier = "Fair"
        elif duration <= timeout:
            tier = "Poor"
        else:
            tier = "Timeout"

        result["efficiency_tier"] = tier

        # Compare to baseline if provided
        if baseline_duration:
            improvement = max(0.0, (baseline_duration - duration) / baseline_duration)
            result["improvement_over_baseline"] = improvement
            result["relative_speed"] = baseline_duration / duration if duration > 0 else float("inf")

        return result


class QualityEvaluator(BaseEvaluator):
    """Evaluates quality of outputs using criteria-based assessment."""

    async def evaluate(self, output: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate output quality based on criteria."""
        result = {
            "score": 0.0,
            "criteria_scores": {},
            "overall_quality": "unknown",
            "feedback": [],
        }

        total_score = 0.0
        total_weight = 0.0

        # Evaluate each criterion
        for criterion, config in criteria.items():
            weight = config.get("weight", 1.0)
            criterion_score = await self._evaluate_criterion(output, criterion, config)

            result["criteria_scores"][criterion] = criterion_score
            total_score += criterion_score * weight
            total_weight += weight

        # Calculate overall score
        if total_weight > 0:
            result["score"] = total_score / total_weight
        else:
            result["score"] = 0.0

        # Determine quality tier
        score = result["score"]
        if score >= 0.9:
            quality = "Excellent"
        elif score >= 0.75:
            quality = "Good"
        elif score >= 0.6:
            quality = "Fair"
        elif score >= 0.4:
            quality = "Poor"
        else:
            quality = "Very Poor"

        result["overall_quality"] = quality

        return result

    async def _evaluate_criterion(self, output: str, criterion: str, config: Dict[str, Any]) -> float:
        """Evaluate a specific quality criterion."""
        criterion_type = config.get("type", "keyword")

        if criterion_type == "keyword":
            return self._evaluate_keyword_presence(output, config)
        elif criterion_type == "length":
            return self._evaluate_length(output, config)
        elif criterion_type == "format":
            return self._evaluate_format(output, config)
        elif criterion_type == "sentiment":
            return self._evaluate_sentiment(output, config)
        else:
            # Default to simple keyword check
            return self._evaluate_keyword_presence(output, config)

    def _evaluate_keyword_presence(self, output: str, config: Dict[str, Any]) -> float:
        """Evaluate based on keyword presence."""
        required_keywords = config.get("keywords", [])
        if not required_keywords:
            return 1.0

        output_lower = output.lower()
        found_keywords = sum(1 for keyword in required_keywords if keyword.lower() in output_lower)

        return found_keywords / len(required_keywords)

    def _evaluate_length(self, output: str, config: Dict[str, Any]) -> float:
        """Evaluate based on output length."""
        min_length = config.get("min_length", 0)
        max_length = config.get("max_length", float("inf"))
        optimal_length = config.get("optimal_length")

        length = len(output.strip())

        if length < min_length:
            return 0.0
        elif length > max_length:
            return max(0.0, 1.0 - (length - max_length) / max_length)
        elif optimal_length:
            # Score based on distance from optimal
            distance = abs(length - optimal_length)
            return max(0.0, 1.0 - distance / optimal_length)
        else:
            return 1.0

    def _evaluate_format(self, output: str, config: Dict[str, Any]) -> float:
        """Evaluate based on format requirements."""
        format_type = config.get("format", "text")

        if format_type == "json":
            try:
                json.loads(output.strip())
                return 1.0
            except json.JSONDecodeError:
                return 0.0

        elif format_type == "code":
            # Simple check for code-like structure
            has_keywords = any(
                keyword in output for keyword in ["def ", "class ", "import ", "function", "var ", "let "]
            )
            has_structure = any(char in output for char in ["{", "}", "(", ")", "[", "]"])
            return 1.0 if has_keywords or has_structure else 0.5

        elif format_type == "markdown":
            # Check for markdown elements
            has_headers = bool(re.search(r"^#{1,6}\s", output, re.MULTILINE))
            has_formatting = bool(re.search(r"\*\*.*\*\*|\*.*\*|`.*`", output))
            return 1.0 if has_headers or has_formatting else 0.5

        else:
            return 1.0  # Default to passing for unknown formats

    def _evaluate_sentiment(self, output: str, config: Dict[str, Any]) -> float:
        """Evaluate sentiment (simplified implementation)."""
        expected_sentiment = config.get("sentiment", "neutral")

        # Simple keyword-based sentiment analysis
        positive_words = [
            "good",
            "great",
            "excellent",
            "wonderful",
            "amazing",
            "positive",
            "success",
        ]
        negative_words = [
            "bad",
            "terrible",
            "awful",
            "horrible",
            "negative",
            "failure",
            "error",
        ]

        output_lower = output.lower()
        positive_count = sum(1 for word in positive_words if word in output_lower)
        negative_count = sum(1 for word in negative_words if word in output_lower)

        if expected_sentiment == "positive":
            return 1.0 if positive_count > negative_count else 0.5
        elif expected_sentiment == "negative":
            return 1.0 if negative_count > positive_count else 0.5
        else:  # neutral
            return 1.0 if abs(positive_count - negative_count) <= 1 else 0.7


class CostEvaluator(BaseEvaluator):
    """Evaluates cost metrics and efficiency."""

    def evaluate(self, token_usage: Dict[str, int], model: str, duration: float) -> Dict[str, Any]:
        """Evaluate cost-related metrics."""
        # Simplified cost calculation (would need real pricing data)
        cost_per_token = self._get_cost_per_token(model)

        input_tokens = token_usage.get("input", 0)
        output_tokens = token_usage.get("output", 0)
        total_tokens = token_usage.get("total", input_tokens + output_tokens)

        estimated_cost = total_tokens * cost_per_token

        return {
            "estimated_cost": estimated_cost,
            "cost_per_token": cost_per_token,
            "tokens_per_second": total_tokens / duration if duration > 0 else 0,
            "cost_efficiency_tier": self._get_cost_tier(estimated_cost),
        }

    def _get_cost_per_token(self, model: str) -> float:
        """Get estimated cost per token for different models."""
        # Simplified pricing (in reality, would fetch from pricing API)
        cost_map = {
            "gpt-4": 0.00003,
            "gpt-3.5-turbo": 0.000002,
            "claude-3": 0.000015,
            "ollama": 0.0,  # Local models
        }

        for model_prefix, cost in cost_map.items():
            if model_prefix in model.lower():
                return cost

        return 0.00001  # Default estimate

    def _get_cost_tier(self, cost: float) -> str:
        """Classify cost into tiers."""
        if cost == 0:
            return "Free"
        elif cost < 0.001:
            return "Very Low"
        elif cost < 0.01:
            return "Low"
        elif cost < 0.1:
            return "Medium"
        elif cost < 1.0:
            return "High"
        else:
            return "Very High"


class LLMEvaluator(BaseEvaluator):
    """Evaluates outputs using another LLM as a judge."""

    def __init__(self, evaluator_model: str = "openai:gpt-4o-mini"):
        """Initialize the LLM evaluator.

        Args:
            evaluator_model: Model to use for evaluation (format: provider:model_name)
        """
        self.evaluator_model = evaluator_model

    async def evaluate(
        self,
        output: str,
        task_description: str,
        evaluation_criteria: str,
        expected_format: str = None,
        rubric: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Evaluate output using an LLM judge.

        Args:
            output: The agent's output to evaluate
            task_description: Description of what the agent was asked to do
            evaluation_criteria: Criteria for evaluation (e.g., "accuracy, clarity, completeness")
            expected_format: Expected format of the output (optional)
            rubric: Detailed scoring rubric (optional)

        Returns:
            Dictionary with evaluation results
        """
        try:
            import tempfile
            from pathlib import Path

            # Create evaluation prompt
            evaluation_prompt = self._create_evaluation_prompt(
                output, task_description, evaluation_criteria, expected_format, rubric
            )

            # Create a temporary evaluator agent
            evaluator_agent_content = self._create_evaluator_agent(self.evaluator_model)

            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
                f.write(evaluator_agent_content)
                temp_agent_path = f.name

            try:
                # Import here to avoid circular imports
                from ..agent_runner import run_agent

                # Run the evaluator agent
                evaluation_result = run_agent(
                    agent_path=Path(temp_agent_path),
                    prompt=evaluation_prompt,
                    context={},
                    model_override=None,
                    debug=False,
                )

                # Parse the evaluation result
                parsed_result = self._parse_evaluation_result(evaluation_result)

                return {
                    "llm_score": parsed_result.get("score", 0.0),
                    "llm_feedback": parsed_result.get("feedback", ""),
                    "llm_reasoning": parsed_result.get("reasoning", ""),
                    "criteria_breakdown": parsed_result.get("criteria_breakdown", {}),
                    "overall_assessment": parsed_result.get("assessment", ""),
                    "evaluator_model": self.evaluator_model,
                    "raw_evaluation": evaluation_result,
                }

            finally:
                # Clean up temporary file
                import os

                try:
                    os.unlink(temp_agent_path)
                except Exception:
                    pass

        except Exception as e:
            return {
                "llm_score": 0.0,
                "llm_feedback": f"Evaluation failed: {str(e)}",
                "llm_reasoning": "",
                "criteria_breakdown": {},
                "overall_assessment": "Error",
                "evaluator_model": self.evaluator_model,
                "error": str(e),
            }

    def _create_evaluation_prompt(
        self,
        output: str,
        task_description: str,
        evaluation_criteria: str,
        expected_format: str = None,
        rubric: Dict[str, Any] = None,
    ) -> str:
        """Create the evaluation prompt for the LLM judge."""

        prompt = f"""You are an expert evaluator tasked with assessing an AI agent's performance.

## Task Description
The agent was asked to: {task_description}

## Evaluation Criteria
Evaluate the output based on: {evaluation_criteria}

## Agent's Output
{output}

## Instructions
1. Carefully analyze the agent's output against the task requirements
2. Rate each criterion on a scale of 0-10 (0 = completely fails, 10 = exceeds expectations)
3. Provide constructive feedback explaining your scoring
4. Give an overall assessment and final score

"""

        if expected_format:
            prompt += f"\n## Expected Format\nThe output should follow this format: {expected_format}\n"

        if rubric:
            prompt += "\n## Detailed Rubric\n"
            for criterion, details in rubric.items():
                prompt += f"**{criterion}**: {details}\n"

        prompt += """
## Required Response Format
Please respond with a JSON object containing:
{
  "score": <overall_score_0_to_10>,
  "feedback": "<detailed_feedback>",
  "reasoning": "<explanation_of_scoring>",
  "criteria_breakdown": {
    "<criterion1>": <score_0_to_10>,
    "<criterion2>": <score_0_to_10>
  },
  "assessment": "<overall_quality_assessment>"
}

Provide thorough, constructive feedback that would help improve the agent's performance."""

        return prompt

    def _create_evaluator_agent(self, model: str) -> str:
        """Create a temporary evaluator agent."""
        return f"""---
name: llm_evaluator
model: {model}
max_steps: 3
tools: []
---

# LLM Evaluator Agent

You are an expert AI evaluator with deep knowledge of AI systems, natural language processing, and task completion assessment.

Your role is to provide fair, objective, and constructive evaluation of AI agent outputs.

## Evaluation Principles
- **Accuracy**: Does the output correctly address the task?
- **Completeness**: Are all requirements fulfilled?
- **Clarity**: Is the output clear and well-structured?
- **Relevance**: Does the output stay on topic and address the request?
- **Quality**: Is the output of high quality with attention to detail?

## Task
{{{{ user_prompt }}}}

## Instructions
Analyze the provided output carefully and return a properly formatted JSON response with scores and detailed feedback.
"""

    def _parse_evaluation_result(self, evaluation_result: str) -> Dict[str, Any]:
        """Parse the LLM evaluation result."""
        try:
            # Try to extract JSON from the result
            import re

            # Look for JSON block in the response
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", evaluation_result, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Look for JSON object directly with proper handling of nested braces
                json_match = re.search(r"\{(?:[^{}]|{[^}]*})*\}", evaluation_result, re.DOTALL)
                if json_match and '"score"' in json_match.group(0):
                    json_str = json_match.group(0)
                else:
                    # Try to find any JSON-like structure
                    json_str = evaluation_result.strip()

            # Parse the JSON
            result = json.loads(json_str)

            # Normalize score to 0-1 range if it's 0-10
            score = result.get("score", 0)

            # Convert to numeric if it's a string
            if isinstance(score, str):
                try:
                    score = float(score)
                except ValueError:
                    score = 0.5  # Default to middle score for non-numeric

            if isinstance(score, (int, float)) and score > 1:
                score = score / 10.0

            result["score"] = max(0.0, min(1.0, float(score)))

            # Normalize criteria breakdown scores
            if "criteria_breakdown" in result:
                normalized_breakdown = {}
                for criterion, score in result["criteria_breakdown"].items():
                    # Convert to numeric if it's a string
                    if isinstance(score, str):
                        try:
                            score = float(score)
                        except ValueError:
                            score = 0.5  # Default to middle score for non-numeric

                    if isinstance(score, (int, float)) and score > 1:
                        score = score / 10.0
                    normalized_breakdown[criterion] = max(0.0, min(1.0, float(score)))
                result["criteria_breakdown"] = normalized_breakdown

            return result

        except (json.JSONDecodeError, AttributeError):
            # Fallback parsing if JSON parsing fails
            return self._fallback_parse(evaluation_result)

    def _fallback_parse(self, evaluation_result: str) -> Dict[str, Any]:
        """Fallback parsing when JSON extraction fails."""
        import re

        # Try to extract score using multiple patterns (check in order of specificity)
        score = 0.5  # Default middle score

        # Pattern 1: Percentage (75%, 85%) - check first since it's most specific
        if re.search(r"(\d+(?:\.\d+)?)\s*%", evaluation_result):
            pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", evaluation_result)
            score = float(pct_match.group(1)) / 100.0

        # Pattern 2: "score/rate X out of Y"
        elif re.search(r"(?:score|rating|rate).*?(\d+(?:\.\d+)?)\s*out\s*of\s*(\d+)", evaluation_result, re.IGNORECASE):
            score_match = re.search(
                r"(?:score|rating|rate).*?(\d+(?:\.\d+)?)\s*out\s*of\s*(\d+)", evaluation_result, re.IGNORECASE
            )
            extracted_score = float(score_match.group(1))
            max_score = float(score_match.group(2))
            score = extracted_score / max_score

        # Pattern 3: "score/rate X" without "out of"
        elif re.search(r"(?:score|rating|rate).*?(\d+(?:\.\d+)?)", evaluation_result, re.IGNORECASE):
            score_match = re.search(r"(?:score|rating|rate).*?(\d+(?:\.\d+)?)", evaluation_result, re.IGNORECASE)
            extracted_score = float(score_match.group(1))
            if extracted_score > 1:
                score = extracted_score / 10.0
            else:
                score = extracted_score

        # Pattern 4: Just a number followed by descriptive text
        elif re.search(r"\b(\d+(?:\.\d+)?)\b", evaluation_result):
            num_match = re.search(r"\b(\d+(?:\.\d+)?)\b", evaluation_result)
            extracted_score = float(num_match.group(1))
            if extracted_score > 1:
                score = extracted_score / 10.0
            else:
                score = extracted_score

        return {
            "score": max(0.0, min(1.0, score)),
            "feedback": evaluation_result[:500] + "..." if len(evaluation_result) > 500 else evaluation_result,
            "reasoning": "Fallback parsing - JSON extraction failed",
            "criteria_breakdown": {},
            "assessment": "Evaluation completed with fallback parsing",
        }
