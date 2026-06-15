"""Dynamic model registry"""

from __future__ import annotations

from .base import ModelInfo, Usage

_REGISTRY: dict[str, ModelInfo] = {}


def get_model_info(provider: str, model: str) -> ModelInfo | None:
    """Look up model info by provider and model name.

    Tries exact match first, then longest-prefix match for versioned model names.
    """
    key = f"{provider}/{model}"
    if key in _REGISTRY:
        return _REGISTRY[key]

    best_match = None
    best_len = 0
    for reg_key, info in _REGISTRY.items():
        if not key.startswith(reg_key) or len(reg_key) <= best_len:
            continue
        # Only treat the prefix as the same model when the remainder is a date/version
        # continuation (e.g. "-20250805"), not a distinct variant ("-mini", "-turbo")
        # — otherwise an unlisted variant silently inherits a sibling's pricing.
        suffix = key[len(reg_key) :]
        if suffix and not (suffix.startswith("-") and len(suffix) > 1 and suffix[1].isdigit()):
            continue
        best_match = info
        best_len = len(reg_key)
    return best_match


def register_model(provider: str, model: str, info: ModelInfo) -> None:
    """Register or update model info. Called by providers during init or discovery."""
    _REGISTRY[f"{provider}/{model}"] = info


def register_models(models: dict[str, ModelInfo]) -> None:
    """Bulk register models. Keys are 'provider/model' strings."""
    _REGISTRY.update(models)


def list_models() -> dict[str, ModelInfo]:
    """Return a snapshot of all registered models, keyed by 'provider/model'."""
    return dict(_REGISTRY)


def calculate_cost(provider: str, model: str, usage: Usage) -> float | None:
    """Calculate cost from usage stats. Returns None for unknown models."""
    info = get_model_info(provider, model)
    if not info or info.input_cost_per_million is None or info.output_cost_per_million is None:
        return None

    input_cost = (usage.prompt_tokens * info.input_cost_per_million) / 1_000_000
    output_cost = (usage.completion_tokens * info.output_cost_per_million) / 1_000_000
    return input_cost + output_cost
