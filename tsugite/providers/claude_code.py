"""Claude Code provider — lists available models for discovery. Actual LLM calls go through the subprocess."""

from __future__ import annotations

from typing import Any, AsyncIterator

from .base import CompletionResponse, ModelInfo, StreamChunk, Usage
from .model_registry import get_model_info as _get_model_info, register_models

_CLAUDE_CODE_MODELS: dict[str, ModelInfo] = {
    "claude_code/claude-opus-4-6": ModelInfo(max_input_tokens=1_000_000, supports_vision=True),
    "claude_code/claude-sonnet-4-6": ModelInfo(max_input_tokens=1_000_000, supports_vision=True),
    "claude_code/claude-haiku-4-5-20251001": ModelInfo(max_input_tokens=200_000, supports_vision=True),
}

# Short aliases used in model strings (e.g., claude_code:sonnet)
_ALIASES = {
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}


class ClaudeCodeProvider:
    """Provider for Claude Code CLI. Only used for model discovery and info lookups."""

    def __init__(self, name: str = "claude_code"):
        self.name = name
        register_models(_CLAUDE_CODE_MODELS)

    async def acompletion(self, messages: list[dict], model: str, stream: bool = False, **kwargs: Any) -> CompletionResponse | AsyncIterator[StreamChunk]:
        raise NotImplementedError("Claude Code uses subprocess, not acompletion()")

    def count_tokens(self, text: str, model: str) -> int:
        from .base import default_count_tokens
        return default_count_tokens(text, model)

    def get_model_info(self, model: str) -> ModelInfo | None:
        resolved = _ALIASES.get(model, model)
        return _get_model_info(self.name, resolved)

    async def list_models(self) -> list[str]:
        return list(_ALIASES.keys())


def create_provider(name: str = "claude_code", **kwargs: Any) -> ClaudeCodeProvider:
    return ClaudeCodeProvider(name=name)
