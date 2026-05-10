"""Model catalog and alias resolution for the ACP provider."""

from __future__ import annotations

from tsugite.providers.base import ModelInfo
from tsugite.providers.model_registry import register_models

_ACP_EFFORT_LEVELS = ["low", "medium", "high", "xhigh", "max"]


def _model(max_input_tokens: int) -> ModelInfo:
    return ModelInfo(
        max_input_tokens=max_input_tokens,
        supports_vision=True,
        supported_effort_levels=_ACP_EFFORT_LEVELS,
    )


_ACP_MODELS: dict[str, ModelInfo] = {
    "acp/claude-opus-4-7": _model(1_000_000),
    "acp/claude-opus-4-6": _model(1_000_000),
    "acp/claude-sonnet-4-6": _model(1_000_000),
    "acp/claude-haiku-4-5-20251001": _model(200_000),
}

_ALIASES: dict[str, str] = {
    "opus": "claude-opus-4-7",
    "opus-4-7": "claude-opus-4-7",
    "opus-4-6": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}


def resolve_model_alias(model: str) -> str:
    """Map a short alias (`opus`, `sonnet`) to a full model id; pass full ids through."""
    if not model:
        raise ValueError("model must be a non-empty string")
    return _ALIASES.get(model, model)


def register_acp_models() -> None:
    """Register the ACP catalog into tsugite's shared model registry."""
    register_models(_ACP_MODELS)
