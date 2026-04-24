"""Model adapters for Tsugite agents."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tsugite.providers.base import Provider

logger = logging.getLogger(__name__)


class UnsupportedEffortError(ValueError):
    """Raised when a reasoning_effort value is not supported for the resolved model."""

    def __init__(self, model: str, value: str, supported: list[str]):
        self.model = model
        self.value = value
        self.supported = supported
        super().__init__(
            f"reasoning_effort={value!r} is not supported for {model}. "
            f"Valid values: {', '.join(supported)}"
        )


_CLAUDE_CODE_MODEL_MAP = {
    "opus": "claude-opus-4-7",
    "opus-4-7": "claude-opus-4-7",
    "opus-4-6": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}


def resolve_effective_model(
    model_override: Optional[str] = None,
    agent_model: Optional[str] = None,
) -> Optional[str]:
    """Resolve the effective model from override -> agent config -> global default.

    Returns None if no model is configured anywhere.
    """
    model = model_override or agent_model
    if not model:
        from tsugite.config import load_config

        config = load_config()
        model = config.default_model
    return model


def resolve_model_alias(model_string: str) -> str:
    """Resolve a model alias to its full model string.

    Examples:
        >>> resolve_model_alias("cheap")  # if alias exists
        "openai:gpt-4o-mini"
        >>> resolve_model_alias("ollama:qwen2.5-coder:7b")
        "ollama:qwen2.5-coder:7b"
    """
    from tsugite.config import get_model_alias

    if ":" in model_string:
        return model_string

    alias_value = get_model_alias(model_string)
    if alias_value:
        return alias_value

    return model_string


def parse_model_string(model_string: str) -> tuple[str, str, Optional[str]]:
    """Parse Tsugite model string format.

    Args:
        model_string: Format like "ollama:qwen2.5-coder:14b" or "openai:gpt-4"

    Returns:
        Tuple of (provider, model_name, variant)
    """
    parts = model_string.split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid model string format: {model_string}")

    provider = parts[0]
    model_name = parts[1]
    variant = parts[2] if len(parts) > 2 else None

    return provider, model_name, variant


def get_model_id(model_string: str) -> str:
    """Get the model ID to pass to the provider's acompletion().

    For most providers this is the model_name (+ variant for ollama).
    For claude_code, maps short names to full model IDs.
    """
    resolved = resolve_model_alias(model_string)
    provider, model_name, variant = parse_model_string(resolved)

    if provider == "claude_code":
        return _CLAUDE_CODE_MODEL_MAP.get(model_name, model_name)

    if provider == "ollama" and variant:
        return f"{model_name}:{variant}"

    return model_name


def get_provider_and_model(model_string: str) -> tuple[str, Provider, str]:
    """Parse a model string and return (provider_name, provider_instance, model_id).

    This is the main entry point for resolving a tsugite model string
    into something usable for LLM calls.
    """
    from tsugite.providers import get_provider

    resolved = resolve_model_alias(model_string)
    provider_name, model_name, variant = parse_model_string(resolved)

    provider = get_provider(provider_name)
    model_id = get_model_id(resolved)

    return provider_name, provider, model_id


def is_reasoning_model_without_stop_support(model_string: str) -> bool:
    """Check if model is a reasoning model that doesn't support stop sequences."""
    from tsugite.providers.model_registry import get_model_info

    try:
        resolved = resolve_model_alias(model_string)
        provider, model_name, variant = parse_model_string(resolved)
    except ValueError:
        return False

    model_id = get_model_id(resolved)
    info = get_model_info(provider, model_id)
    if info and info.supports_reasoning:
        return True

    # Fallback regex for models not in the registry (e.g., new OpenAI reasoning models)
    if provider == "openai":
        return bool(re.match(r"^(o1|o3|o4)(-mini|-preview)?(-\d{4}-\d{2}-\d{2})?$", model_name))
    return False


def filter_reasoning_model_params(model_name: str, params: dict) -> dict:
    """Filter out unsupported parameters for reasoning models.

    `reasoning_effort` is handled upstream via `ModelInfo.supported_effort_levels`
    and runner-level validation, so it is not filtered here.
    """
    del model_name
    unsupported_params = ["stop", "temperature", "top_p", "presence_penalty", "frequency_penalty"]
    for param in unsupported_params:
        params.pop(param, None)
    return params


def resolve_reasoning_effort(model_string: str, value: Optional[str]) -> Optional[str]:
    """Validate a reasoning_effort value against the resolved model's capability list.

    Returns the value verbatim if the model declares it in ``supported_effort_levels``.
    Returns ``None`` and logs a warning if the model does not support any effort levels.
    Raises ``UnsupportedEffortError`` if the value is not in the model's list.
    """
    if value is None:
        return None

    from tsugite.providers import get_provider

    try:
        resolved = resolve_model_alias(model_string)
        provider_name, _model_name, _variant = parse_model_string(resolved)
    except ValueError:
        logger.warning("Cannot parse model %r; dropping reasoning_effort=%r", model_string, value)
        return None

    model_id = get_model_id(resolved)
    try:
        info = get_provider(provider_name).get_model_info(model_id)
    except Exception:  # noqa: BLE001
        info = None
    if info is None or not info.supported_effort_levels:
        logger.warning(
            "Model %s does not support reasoning_effort; dropping value %r",
            resolved,
            value,
        )
        return None

    if value not in info.supported_effort_levels:
        raise UnsupportedEffortError(resolved, value, info.supported_effort_levels)

    return value


def get_model_kwargs(model_string: str, **kwargs) -> dict:
    """Prepare kwargs for provider.acompletion(), filtering reasoning model params.

    Returns cleaned kwargs dict ready to pass as **kwargs to provider.acompletion().
    """
    resolved = resolve_model_alias(model_string)

    params = dict(kwargs)
    if is_reasoning_model_without_stop_support(resolved):
        _, model_name, _ = parse_model_string(resolved)
        params = filter_reasoning_model_params(model_name, params)

    return params
