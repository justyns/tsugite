"""Memory management helpers for daemon."""

import asyncio
import logging

DEFAULT_COMPACT_MODEL = "openai:gpt-4o-mini"
DEFAULT_CONTEXT_LIMIT = 128_000
CONTEXT_RESERVE_RATIO = 0.25

PROVIDER_COMPACT_MODELS = {
    "openai": DEFAULT_COMPACT_MODEL,
    "anthropic": "anthropic:claude-3-haiku-20240307",
    "google": "google:gemini-2.0-flash-lite",
    "openrouter": "openrouter:openai/gpt-4o-mini",
    "ollama": None,  # use agent model as-is
}

logger = logging.getLogger(__name__)

SUMMARIZE_SYSTEM_PROMPT = (
    "Summarize this conversation concisely. Focus on: decisions made, "
    "important facts learned, user preferences, action items. Keep it under 500 words."
)

COMBINE_SYSTEM_PROMPT = (
    "You are given multiple summaries of consecutive conversation chunks. "
    "Combine them into a single coherent summary. Focus on: decisions made, "
    "important facts learned, user preferences, action items. Keep it under 500 words."
)


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


def _get_context_limit(model: str, fallback: int | None = None) -> int:
    """Get context limit for a model via litellm.

    Priority: litellm model info -> fallback param -> DEFAULT_CONTEXT_LIMIT.
    """
    from litellm import get_model_info

    from tsugite.models import get_model_params

    litellm_model = get_model_params(model)["model"]
    try:
        info = get_model_info(litellm_model)
        limit = info.get("max_input_tokens")
        if limit is not None:
            return limit
    except Exception:
        pass
    return fallback if fallback is not None else DEFAULT_CONTEXT_LIMIT


def _count_tokens(text: str, model: str) -> int:
    """Count tokens using litellm, falling back to len//4."""
    try:
        from litellm import token_counter

        from tsugite.models import get_model_params

        litellm_model = get_model_params(model)["model"]
        return token_counter(model=litellm_model, text=text)
    except Exception:
        return len(text) // 4


def _message_text(msg: dict) -> str:
    """Extract text content from a message dict."""
    content = msg.get("content")
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block if isinstance(block, str) else block["text"]
            for block in content
            if isinstance(block, str) or (isinstance(block, dict) and "text" in block)
        ]
        return "\n".join(parts)
    return str(content)


def _chunk_messages(messages: list[dict], max_chunk_tokens: int, model: str) -> list[list[dict]]:
    """Split messages into chunks that fit within token budget.

    Keeps user/assistant pairs together when possible.
    """
    if not messages:
        return []

    chunks: list[list[dict]] = []
    current_chunk: list[dict] = []
    current_tokens = 0

    for msg in messages:
        msg_text = _message_text(msg)
        text = f"{msg.get('role', 'user').upper()}: {msg_text}"
        msg_tokens = _count_tokens(text, model)

        if msg_tokens > max_chunk_tokens:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            chars_budget = max_chunk_tokens * 4
            chunks.append([{**msg, "content": msg_text[:chars_budget]}])
            continue

        # Would adding this message exceed the budget?
        if current_tokens + msg_tokens > max_chunk_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.append(msg)
        current_tokens += msg_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


async def _llm_complete(system_prompt: str, user_content: str, model: str) -> str:
    """Send a system+user message pair to the LLM and return the response text."""
    from litellm import acompletion

    from tsugite.models import get_model_params

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    params = get_model_params(model, messages=messages)
    response = await acompletion(**params)
    return response.choices[0].message.content


async def _summarize_chunk(messages: list[dict], model: str) -> str:
    """Summarize a single chunk of conversation messages."""
    convo_text = "\n\n".join(f"{msg['role'].upper()}: {_message_text(msg)}" for msg in messages)
    return await _llm_complete(SUMMARIZE_SYSTEM_PROMPT, convo_text, model)


async def _combine_summaries(summaries: list[str], model: str) -> str:
    """Combine multiple chunk summaries into one."""
    numbered = "\n\n".join(f"--- Chunk {i + 1} ---\n{s}" for i, s in enumerate(summaries))
    return await _llm_complete(COMBINE_SYSTEM_PROMPT, numbered, model)


async def summarize_session(
    conversation_history: list[dict],
    model: str | None = None,
    max_context_tokens: int | None = None,
) -> str:
    """Summarize conversation history using LLM with chunked map-reduce.

    Args:
        conversation_history: List of {role, content} dicts
        model: Model to use for summarization (Tsugite format: provider:model).
        max_context_tokens: Override context limit (for Ollama/custom models).
    """
    if not conversation_history:
        return "No conversation to summarize."

    if model is None:
        model = DEFAULT_COMPACT_MODEL

    context_limit = _get_context_limit(model, fallback=max_context_tokens)
    usable_tokens = int(context_limit * (1 - CONTEXT_RESERVE_RATIO))

    chunks = _chunk_messages(conversation_history, usable_tokens, model)
    if not chunks:
        return "No conversation to summarize."

    if len(chunks) == 1:
        return await _summarize_chunk(chunks[0], model)

    logger.info("Summarizing %d chunks (context limit: %d, usable: %d)", len(chunks), context_limit, usable_tokens)
    chunk_summaries = await asyncio.gather(*[_summarize_chunk(chunk, model) for chunk in chunks])
    return await _combine_summaries(chunk_summaries, model)
