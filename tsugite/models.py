"""Model adapters for Tsugite agents."""

import re
from typing import Optional


def resolve_model_alias(model_string: str) -> str:
    """Resolve a model alias to its full model string.

    Args:
        model_string: Either an alias name or full model string

    Returns:
        Full model string (resolved alias or original string)

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


def is_reasoning_model_without_stop_support(model_string: str) -> bool:
    """Check if model is a reasoning model that doesn't support stop sequences.

    OpenAI's o1/o3 reasoning models don't support the stop parameter.

    Args:
        model_string: Full model string like "openai:o1-mini"

    Returns:
        True if this is a reasoning model that doesn't support stop sequences
    """
    try:
        provider, model_name, variant = parse_model_string(model_string)
    except ValueError:
        return False

    # Only OpenAI reasoning models have this limitation
    if provider != "openai":
        return False

    # Check if it's an o1 or o3 model (with optional version suffix)
    # Matches: o1, o1-mini, o1-preview, o1-2024-12-17, o3, o3-mini, o3-2025-01-31, etc.
    pattern = r"^(o1|o3)(-mini|-preview)?(-\d{4}-\d{2}-\d{2})?$"
    return bool(re.match(pattern, model_name))


def get_model_params(model_string: str, **kwargs) -> dict:
    """Get parameters for direct litellm.acompletion() calls.

    Returns a dict with model ID and parameters for direct LiteLLM usage.

    Args:
        model_string: Model specification like "openai:gpt-4o-mini" or an alias
        **kwargs: Additional model parameters (reasoning_effort, response_format, temperature, etc.)

    Returns:
        Dict with "model" key and all parameters ready for litellm.acompletion()

    Examples:
        >>> params = get_model_params("openai:gpt-4o-mini", temperature=0.7)
        >>> params["model"]
        'openai/gpt-4o-mini'
        >>> params["temperature"]
        0.7

        >>> # Reasoning models - unsupported params filtered out
        >>> params = get_model_params("openai:o1", temperature=0.7)
        >>> "temperature" not in params  # Filtered for o1
        True
    """
    import os

    # Resolve aliases
    resolved_model = resolve_model_alias(model_string)
    provider, model_name, variant = parse_model_string(resolved_model)

    # Build parameters dict
    params = dict(kwargs)

    # Handle reasoning models - filter out unsupported parameters
    if is_reasoning_model_without_stop_support(resolved_model):
        unsupported_params = ["stop", "temperature", "top_p", "presence_penalty", "frequency_penalty"]

        # o1-mini specifically doesn't support reasoning_effort
        if "o1-mini" in model_name:
            unsupported_params.append("reasoning_effort")

        # Remove unsupported parameters
        for param in unsupported_params:
            params.pop(param, None)

    # Build model ID for LiteLLM
    if provider == "ollama":
        # Ollama: use full model name with variant
        full_model_name = f"{model_name}:{variant}" if variant else model_name
        params["model"] = full_model_name
        params.setdefault("api_base", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
        params.setdefault("api_key", "ollama")  # Ollama doesn't need real API key

    elif provider == "openai":
        params["model"] = f"openai/{model_name}"
        if "api_key" not in params:
            params["api_key"] = os.getenv("OPENAI_API_KEY")

    elif provider == "anthropic":
        params["model"] = f"anthropic/{model_name}"
        if "api_key" not in params:
            params["api_key"] = os.getenv("ANTHROPIC_API_KEY")

    elif provider == "google":
        params["model"] = f"gemini/{model_name}"
        if "api_key" not in params:
            params["api_key"] = os.getenv("GOOGLE_API_KEY")

    elif provider == "github_copilot":
        params["model"] = f"github_copilot/{model_name}"

        # GitHub Copilot requires specific headers
        extra_headers = params.get("extra_headers", {})
        if "editor-version" not in extra_headers:
            extra_headers["editor-version"] = "vscode/1.95.0"
        if "Copilot-Integration-Id" not in extra_headers:
            extra_headers["Copilot-Integration-Id"] = "vscode-chat"
        params["extra_headers"] = extra_headers

    else:
        # Fallback: try LiteLLM with the provider prefix
        params["model"] = f"{provider}/{model_name}"

    return params
