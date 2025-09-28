"""Evaluators for different aspects of benchmark test results."""

import re
import json
import ast
import difflib
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod


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
        if type(obj1) != type(obj2):
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
