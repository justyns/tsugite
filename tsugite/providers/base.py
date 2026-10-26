"""Provider protocol and shared types for LLM backends."""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol, runtime_checkable

_O200K_PREFIXES = ("gpt-4o", "o1", "o3", "o4", "chatgpt-4o")
_tiktoken_cache: dict[str, Any] = {}
API_KEY_CACHE_TTL_SECONDS = 60
_secret_api_keys: dict[str, tuple[float, str | None]] = {}
_backend_errors_logged: set[str] = set()

logger = logging.getLogger(__name__)


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
    reasoning_content: str = ""
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
    supported_effort_levels: list[str] | None = None
    # How the model exposes reasoning control: "budget_tokens" (extended
    # thinking via thinking budgets) or "adaptive" (native effort parameter;
    # budget_tokens and sampling params are rejected with 400). None when the
    # model has neither or the style is unknown.
    thinking_style: str | None = None


@runtime_checkable
class Provider(Protocol):
    name: str
    cacheable: bool
    # True when the provider exposes a finite, known set of model ids (CLI-backed
    # providers like claude_code/codex_cli). API providers accept arbitrary ids, so
    # they leave this False and callers must not treat an unlisted id as wrong.
    models_are_definitive: bool = False

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


def api_key_secret_name(env_var: str) -> str:
    """Secrets-backend name for the key read from ``env_var``."""
    return env_var.lower().replace("_", "-")


def resolve_api_key(env_var: str) -> str | None:
    """Key from the secrets backend, else ``env_var``. Backend lookups are cached for ``API_KEY_CACHE_TTL_SECONDS``."""
    now = time.monotonic()
    cached = _secret_api_keys.get(env_var)
    if cached is None or now - cached[0] >= API_KEY_CACHE_TTL_SECONDS:
        from tsugite.secrets import get_backend

        try:
            _secret_api_keys[env_var] = (now, get_backend().get(api_key_secret_name(env_var)))
        except Exception as e:
            if env_var not in _backend_errors_logged:
                _backend_errors_logged.add(env_var)
                logger.warning("Secrets backend failed reading the key for %s, using the env var: %s", env_var, e)
            return os.getenv(env_var)
    return _secret_api_keys[env_var][1] or os.getenv(env_var)


def invalidate_api_key_cache() -> None:
    _secret_api_keys.clear()
    _backend_errors_logged.clear()


def configured_binary(provider: str) -> str | None:
    from tsugite.config import load_config

    return load_config().plugins.get(provider, {}).get("binary") or None


def resolve_binary(provider: str, default: str) -> str | None:
    """Path of a CLI provider's executable from config ``plugins.<provider>.binary``, else ``default`` on PATH."""
    configured = configured_binary(provider)
    return shutil.which(os.path.expanduser(configured) if configured else default)
