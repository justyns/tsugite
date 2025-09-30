"""Model adapters for Tsugite agents using smolagents."""

from typing import Optional
from smolagents import LiteLLMModel, OpenAIServerModel


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


def get_model(model_string: str, **kwargs):
    """Create a smolagents model from Tsugite model string.

    Args:
        model_string: Model specification like "ollama:qwen2.5-coder:14b" or an alias
        **kwargs: Additional model configuration options

    Returns:
        smolagents Model instance

    Examples:
        >>> model = get_model("ollama:qwen2.5-coder:14b")
        >>> model = get_model("openai:gpt-4", api_key="sk-...")
        >>> model = get_model("cheap")  # if alias exists
    """
    resolved_model = resolve_model_alias(model_string)
    provider, model_name, variant = parse_model_string(resolved_model)

    if provider == "ollama":
        # Ollama typically runs on localhost:11434 with OpenAI-compatible API
        full_model_name = f"{model_name}:{variant}" if variant else model_name
        api_base = kwargs.get("api_base", "http://localhost:11434/v1")

        return OpenAIServerModel(
            model_id=full_model_name,
            api_base=api_base,
            api_key=kwargs.get("api_key", "ollama"),  # Ollama doesn't need real API key
            **{k: v for k, v in kwargs.items() if k not in ["api_base", "api_key"]},
        )

    elif provider == "openai":
        # Use LiteLLM for OpenAI (supports many providers)
        litellm_model = f"openai/{model_name}"
        return LiteLLMModel(
            model_id=litellm_model,
            api_key=kwargs.get("api_key"),
            **{k: v for k, v in kwargs.items() if k != "api_key"},
        )

    elif provider == "anthropic":
        # Use LiteLLM for Anthropic
        litellm_model = f"anthropic/{model_name}"
        return LiteLLMModel(
            model_id=litellm_model,
            api_key=kwargs.get("api_key"),
            **{k: v for k, v in kwargs.items() if k != "api_key"},
        )

    elif provider == "google":
        # Use LiteLLM for Google (Gemini)
        litellm_model = f"gemini/{model_name}"
        return LiteLLMModel(
            model_id=litellm_model,
            api_key=kwargs.get("api_key"),
            **{k: v for k, v in kwargs.items() if k != "api_key"},
        )

    else:
        # Fallback: try LiteLLM with the provider prefix
        # LiteLLM supports 100+ providers
        litellm_model = f"{provider}/{model_name}"
        return LiteLLMModel(model_id=litellm_model, **kwargs)


def create_ollama_model(model_name: str, api_base: str = "http://localhost:11434/v1", **kwargs):
    """Convenience function to create Ollama model.

    Args:
        model_name: Name of the Ollama model (e.g., "qwen2.5-coder:14b")
        api_base: Ollama API base URL
        **kwargs: Additional model configuration

    Returns:
        OpenAIServerModel configured for Ollama
    """
    return OpenAIServerModel(
        model_id=model_name,
        api_base=api_base,
        api_key="ollama",
        **kwargs,
    )


def create_openai_model(model_name: str = "gpt-4", api_key: str = None, **kwargs):
    """Convenience function to create OpenAI model.

    Args:
        model_name: OpenAI model name
        api_key: OpenAI API key
        **kwargs: Additional model configuration

    Returns:
        LiteLLMModel configured for OpenAI
    """
    return LiteLLMModel(model_id=f"openai/{model_name}", api_key=api_key, **kwargs)
