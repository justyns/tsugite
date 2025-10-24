"""Benchmark configuration and shared constants."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SimilarityThresholds:
    """Thresholds for similarity-based evaluation."""

    string_high_similarity: float = 0.9  # High similarity for string matching
    code_similarity: float = 0.85  # Code similarity threshold
    json_similarity: float = 0.8  # JSON structure similarity
    behavior_pass_threshold: float = 0.6  # Minimum score for behavior tests
    custom_criteria_threshold: float = 0.7  # Minimum for custom criteria
    llm_evaluation_threshold: float = 0.7  # Minimum LLM judge score


@dataclass
class PerformanceTiers:
    """Performance tier thresholds."""

    excellent: float = 0.9
    good: float = 0.75
    fair: float = 0.6
    poor: float = 0.4
    # Below poor is "Very Poor"


@dataclass
class EvaluationWeights:
    """Weights for blending different evaluation scores."""

    llm_weight: float = 0.4  # Weight for LLM evaluation in blended score
    base_weight: float = 0.6  # Weight for base evaluation
    planning_weight: float = 0.3  # Weight for planning evaluation
    planning_minimum: float = 0.4  # Minimum planning score required


@dataclass
class BehaviorScores:
    """Score values for behavior-based evaluation."""

    tool_usage: float = 0.3
    file_created: float = 0.3


@dataclass
class ModelCosts:
    """Token costs for different model providers (USD per token)."""

    # OpenAI
    gpt_4: float = 0.00003
    gpt_4_turbo: float = 0.00001
    gpt_3_5_turbo: float = 0.000002

    # Anthropic
    claude_3: float = 0.000015
    claude_3_5: float = 0.000015

    # Google
    gemini_pro: float = 0.0000005

    # Local models
    ollama: float = 0.0

    # Default fallback
    default: float = 0.00001

    def get_cost_for_model(self, model: str) -> float:
        """Get cost per token for a model string."""
        model_lower = model.lower()

        # OpenAI models
        if "gpt-4-turbo" in model_lower or "gpt-4o" in model_lower:
            return self.gpt_4_turbo
        elif "gpt-4" in model_lower:
            return self.gpt_4
        elif "gpt-3.5" in model_lower or "gpt-35" in model_lower:
            return self.gpt_3_5_turbo

        # Anthropic models
        elif "claude-3-5" in model_lower or "claude-3.5" in model_lower:
            return self.claude_3_5
        elif "claude-3" in model_lower:
            return self.claude_3

        # Google models
        elif "gemini" in model_lower:
            return self.gemini_pro

        # Local models
        elif "ollama" in model_lower or "local" in model_lower:
            return self.ollama

        return self.default


@dataclass
class CostTiers:
    """Thresholds for cost tier classification."""

    very_low: float = 0.001
    low: float = 0.01
    medium: float = 0.1
    high: float = 1.0
    # Above high is "Very High", free is 0.0


@dataclass
class TestCategories:
    """Known test categories and their prefixes."""

    categories: Dict[str, str] = field(
        default_factory=lambda: {
            "basic": "basic_",
            "tools": "tools_",
            "scenarios": "scenarios_",
            "performance": "performance_",
            "complex": "complex_",
        }
    )

    def get_category(self, test_id: str) -> str:
        """Extract category from test ID."""
        for category, prefix in self.categories.items():
            if test_id.startswith(prefix):
                return category
        return "unknown"


# Global configuration instances
SIMILARITY_THRESHOLDS = SimilarityThresholds()
PERFORMANCE_TIERS = PerformanceTiers()
EVALUATION_WEIGHTS = EvaluationWeights()
BEHAVIOR_SCORES = BehaviorScores()
MODEL_COSTS = ModelCosts()
COST_TIERS = CostTiers()
TEST_CATEGORIES = TestCategories()


def get_performance_tier(accuracy: float) -> str:
    """Get performance tier label based on accuracy.

    Args:
        accuracy: Accuracy score from 0.0 to 1.0

    Returns:
        Performance tier label
    """
    if accuracy >= PERFORMANCE_TIERS.excellent:
        return "Excellent"
    elif accuracy >= PERFORMANCE_TIERS.good:
        return "Good"
    elif accuracy >= PERFORMANCE_TIERS.fair:
        return "Fair"
    elif accuracy >= PERFORMANCE_TIERS.poor:
        return "Poor"
    else:
        return "Very Poor"


def get_cost_tier(cost: float) -> str:
    """Get cost tier label based on cost in USD.

    Args:
        cost: Cost in USD

    Returns:
        Cost tier label
    """
    if cost == 0:
        return "Free"
    elif cost < COST_TIERS.very_low:
        return "Very Low"
    elif cost < COST_TIERS.low:
        return "Low"
    elif cost < COST_TIERS.medium:
        return "Medium"
    elif cost < COST_TIERS.high:
        return "High"
    else:
        return "Very High"
