"""Memory management helpers for daemon."""

from typing import List, Optional

DEFAULT_COMPACT_MODEL = "openai:gpt-4o-mini"

PROVIDER_COMPACT_MODELS = {
    "openai": DEFAULT_COMPACT_MODEL,
    "anthropic": "anthropic:claude-3-haiku-20240307",
    "google": "google:gemini-2.0-flash-lite",
    "openrouter": "openrouter:openai/gpt-4o-mini",
    "ollama": None,  # use agent model as-is
}


def infer_compaction_model(agent_model: str) -> str:
    """Pick a cheap model from the same provider as the agent."""
    from tsugite.models import parse_model_string, resolve_model_alias

    resolved = resolve_model_alias(agent_model)
    try:
        provider, _, _ = parse_model_string(resolved)
    except ValueError:
        return DEFAULT_COMPACT_MODEL

    if provider not in PROVIDER_COMPACT_MODELS:
        return resolved

    compact = PROVIDER_COMPACT_MODELS[provider]
    return compact if compact is not None else resolved


async def summarize_session(conversation_history: List[dict], model: Optional[str] = None) -> str:
    """Summarize conversation history using LLM.

    Args:
        conversation_history: List of {role, content} dicts
        model: Model to use for summarization (Tsugite format: provider:model).
               Defaults to openai:gpt-4o-mini.

    Returns:
        Concise summary of conversation
    """
    from litellm import acompletion

    from tsugite.models import get_model_params

    if model is None:
        model = DEFAULT_COMPACT_MODEL

    messages = [
        {
            "role": "system",
            "content": "Summarize this conversation concisely. Focus on: decisions made, important facts learned, user preferences, action items. Keep it under 500 words.",
        }
    ]

    convo_text = "\n\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in conversation_history)
    messages.append({"role": "user", "content": convo_text})

    params = get_model_params(model, messages=messages)
    response = await acompletion(**params)
    return response.choices[0].message.content
