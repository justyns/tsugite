"""Provider protocol and shared types for LLM backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol, runtime_checkable

_O200K_PREFIXES = ("gpt-4o", "o1", "o3", "o4", "chatgpt-4o")
_tiktoken_cache: dict[str, Any] = {}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass
class CompletionResponse:
    content: str = ""
    reasoning_content: str | None = None
    usage: Usage | None = None
    cost: float | None = None
    raw: Any = None


@dataclass
class StreamChunk:
    content: str = ""
    done: bool = False
    usage: Usage | None = None
    cost: float | None = None


@dataclass
class ModelInfo:
    max_input_tokens: int = 128_000
    max_output_tokens: int | None = None
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    supports_vision: bool = False
    supports_audio: bool = False
    supports_reasoning: bool = False
    supports_streaming: bool = True


@runtime_checkable
class Provider(Protocol):
    name: str

    async def acompletion(
        self,
        messages: list[dict],
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse | AsyncIterator[StreamChunk]: ...

    def count_tokens(self, text: str, model: str) -> int: ...

    def get_model_info(self, model: str) -> ModelInfo | None: ...

    async def list_models(self) -> list[str]: ...

    def set_context(self, **kwargs: Any) -> None:
        """Pass session context to the provider. No-op for stateless providers."""

    def get_state(self) -> dict | None:
        """Return provider-specific session state, or None for stateless providers."""
        return None

    async def stop(self) -> None:
        """Clean up resources. No-op for stateless providers."""


def default_count_tokens(text: str, model: str) -> int:
    """Shared token counting using tiktoken with cached encodings."""
    try:
        import tiktoken

        encoding_name = "o200k_base" if any(model.startswith(p) for p in _O200K_PREFIXES) else "cl100k_base"
        if encoding_name not in _tiktoken_cache:
            _tiktoken_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
        return len(_tiktoken_cache[encoding_name].encode(text))
    except Exception:
        return len(text) // 4
