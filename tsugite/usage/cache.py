"""Cache-usage breakdown, normalized across provider token-reporting conventions.

Anthropic, claude_code and acp report input_tokens as the uncached remainder,
with cache creation/read tokens additive. Every other provider uses the OpenAI-compatible
convention, where prompt_tokens already includes the cached_tokens subset. The
provider is parsed from the stored "provider:model" string.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_ADDITIVE_CACHE_PROVIDERS = frozenset({"anthropic", "claude_code", "acp"})


def derive_cache_breakdown(
    model: str | None,
    input_tokens: int,
    cache_creation_tokens: int,
    cache_read_tokens: int,
) -> dict[str, float | int | None]:
    """Normalize raw usage columns into cached-read, cache-write and uncached-input terms.

    Returns cached_read_tokens, cache_write_tokens, uncached_input_tokens and
    cache_hit_rate. Providers without a cache-creation count give a None
    cache_write_tokens, and no input gives a None cache_hit_rate.

    cache_hit_rate is read / (read + write + uncached) for every provider, i.e.
    cached reads over all input tokens. An inclusive provider reporting more
    cache reads than input tokens is clamped to 0 uncached and logged.
    """
    additive = bool(model and model.split(":", 1)[0] in _ADDITIVE_CACHE_PROVIDERS)

    cached_read_tokens = cache_read_tokens
    cache_write_tokens = cache_creation_tokens if additive else None
    if additive:
        uncached_input_tokens = input_tokens
    else:
        uncached_input_tokens = input_tokens - cache_read_tokens
        if uncached_input_tokens < 0:
            logger.warning(
                "cache_read_tokens (%d) exceed input_tokens (%d) for %s; provider may be additive",
                cache_read_tokens,
                input_tokens,
                model,
            )
            uncached_input_tokens = 0

    denom = cached_read_tokens + (cache_write_tokens or 0) + uncached_input_tokens
    cache_hit_rate = cached_read_tokens / denom if denom > 0 else None

    return {
        "cached_read_tokens": cached_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "cache_hit_rate": cache_hit_rate,
    }
