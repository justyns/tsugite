"""Model registry with context limits and pricing for known models."""

from __future__ import annotations

from .base import ModelInfo, Usage

# fmt: off
_REGISTRY: dict[str, ModelInfo] = {
    # OpenAI
    "openai/gpt-4o":              ModelInfo(max_input_tokens=128_000, max_output_tokens=16_384, input_cost_per_million=2.50, output_cost_per_million=10.00, supports_vision=True),
    "openai/gpt-4o-mini":         ModelInfo(max_input_tokens=128_000, max_output_tokens=16_384, input_cost_per_million=0.15, output_cost_per_million=0.60, supports_vision=True),
    "openai/gpt-4o-2024-11-20":   ModelInfo(max_input_tokens=128_000, max_output_tokens=16_384, input_cost_per_million=2.50, output_cost_per_million=10.00, supports_vision=True),
    "openai/gpt-4-turbo":         ModelInfo(max_input_tokens=128_000, max_output_tokens=4_096, input_cost_per_million=10.00, output_cost_per_million=30.00, supports_vision=True),
    "openai/gpt-4":               ModelInfo(max_input_tokens=8_192, max_output_tokens=8_192, input_cost_per_million=30.00, output_cost_per_million=60.00),
    "openai/gpt-3.5-turbo":       ModelInfo(max_input_tokens=16_385, max_output_tokens=4_096, input_cost_per_million=0.50, output_cost_per_million=1.50),
    "openai/o1":                   ModelInfo(max_input_tokens=200_000, max_output_tokens=100_000, input_cost_per_million=15.00, output_cost_per_million=60.00, supports_vision=True, supports_reasoning=True),
    "openai/o1-mini":              ModelInfo(max_input_tokens=128_000, max_output_tokens=65_536, input_cost_per_million=3.00, output_cost_per_million=12.00, supports_reasoning=True),
    "openai/o1-preview":           ModelInfo(max_input_tokens=128_000, max_output_tokens=32_768, input_cost_per_million=15.00, output_cost_per_million=60.00, supports_reasoning=True),
    "openai/o3":                   ModelInfo(max_input_tokens=200_000, max_output_tokens=100_000, input_cost_per_million=10.00, output_cost_per_million=40.00, supports_vision=True, supports_reasoning=True),
    "openai/o3-mini":              ModelInfo(max_input_tokens=200_000, max_output_tokens=100_000, input_cost_per_million=1.10, output_cost_per_million=4.40, supports_reasoning=True),
    "openai/o4-mini":              ModelInfo(max_input_tokens=200_000, max_output_tokens=100_000, input_cost_per_million=1.10, output_cost_per_million=4.40, supports_vision=True, supports_reasoning=True),

    # Anthropic
    "anthropic/claude-opus-4-6":            ModelInfo(max_input_tokens=200_000, max_output_tokens=32_000, input_cost_per_million=15.00, output_cost_per_million=75.00, supports_vision=True),
    "anthropic/claude-sonnet-4-6":          ModelInfo(max_input_tokens=200_000, max_output_tokens=16_000, input_cost_per_million=3.00, output_cost_per_million=15.00, supports_vision=True),
    "anthropic/claude-3-5-sonnet-20241022": ModelInfo(max_input_tokens=200_000, max_output_tokens=8_192, input_cost_per_million=3.00, output_cost_per_million=15.00, supports_vision=True),
    "anthropic/claude-3-5-haiku-20241022":  ModelInfo(max_input_tokens=200_000, max_output_tokens=8_192, input_cost_per_million=0.80, output_cost_per_million=4.00, supports_vision=True),
    "anthropic/claude-3-haiku-20240307":    ModelInfo(max_input_tokens=200_000, max_output_tokens=4_096, input_cost_per_million=0.25, output_cost_per_million=1.25, supports_vision=True),
    "anthropic/claude-haiku-4-5-20251001":  ModelInfo(max_input_tokens=200_000, max_output_tokens=8_192, input_cost_per_million=0.80, output_cost_per_million=4.00, supports_vision=True),

    # Claude Code (subprocess)
    "claude_code/claude-opus-4-6":           ModelInfo(max_input_tokens=1_000_000, supports_vision=True),
    "claude_code/claude-sonnet-4-6":         ModelInfo(max_input_tokens=1_000_000, supports_vision=True),
    "claude_code/claude-haiku-4-5-20251001": ModelInfo(max_input_tokens=200_000, supports_vision=True),
}
# fmt: on


def get_model_info(provider: str, model: str) -> ModelInfo | None:
    """Look up model info by provider and model name.

    Tries exact match first, then longest-prefix match for versioned model names.
    """
    key = f"{provider}/{model}"
    if key in _REGISTRY:
        return _REGISTRY[key]

    # Longest-prefix match: lookup key starts with a registry key
    # e.g., "openai/gpt-4o-2024-11-20" matches "openai/gpt-4o" (not "openai/gpt-4")
    best_match = None
    best_len = 0
    for reg_key, info in _REGISTRY.items():
        if key.startswith(reg_key) and len(reg_key) > best_len:
            best_match = info
            best_len = len(reg_key)
    return best_match


def register_model(provider: str, model: str, info: ModelInfo) -> None:
    """Register or update model info. Called by providers during discovery."""
    _REGISTRY[f"{provider}/{model}"] = info


def calculate_cost(provider: str, model: str, usage: Usage) -> float | None:
    """Calculate cost from usage stats. Returns None for unknown models."""
    info = get_model_info(provider, model)
    if not info or info.input_cost_per_million is None or info.output_cost_per_million is None:
        return None

    input_cost = (usage.prompt_tokens * info.input_cost_per_million) / 1_000_000
    output_cost = (usage.completion_tokens * info.output_cost_per_million) / 1_000_000
    return input_cost + output_cost
