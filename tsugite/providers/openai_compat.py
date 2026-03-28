"""OpenAI-compatible provider covering OpenAI, Ollama, OpenRouter, Together, Mistral, GitHub Copilot."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator

import httpx

from .base import CompletionResponse, ModelInfo, StreamChunk, Usage, default_count_tokens
from .model_registry import calculate_cost, get_model_info as _get_model_info

logger = logging.getLogger(__name__)

_PROVIDER_CONFIGS: dict[str, dict[str, Any]] = {
    "openai": {
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "ollama": {
        "api_base_env": "OLLAMA_BASE_URL",
        "api_base_default": "http://localhost:11434/v1",
        "api_key": "ollama",
    },
    "openrouter": {
        "api_base": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "together": {
        "api_base": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
    },
    "mistral": {
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
    },
    "github_copilot": {
        "api_base": "https://api.githubcopilot.com",
        "api_key_env": "GITHUB_COPILOT_TOKEN",
        "extra_headers": {
            "editor-version": "vscode/1.95.0",
            "Copilot-Integration-Id": "vscode-chat",
        },
    },
}


class OpenAICompatProvider:
    """Provider for any OpenAI-compatible chat completions API."""

    def __init__(
        self,
        name: str,
        api_base: str,
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ):
        self.name = name
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.extra_headers = extra_headers or {}
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=300)
        return self._client

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def acompletion(
        self,
        messages: list[dict],
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse | AsyncIterator[StreamChunk]:
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": stream, **kwargs}
        body = {k: v for k, v in body.items() if not k.startswith("_") and v is not None}
        url = f"{self.api_base}/chat/completions"
        headers = self._build_headers()

        if stream:
            body["stream_options"] = {"include_usage": True}
            return self._stream(url, headers, body)

        client = self._get_client()
        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        return self._parse_response(resp.json(), model)

    async def _stream(self, url: str, headers: dict, body: dict) -> AsyncIterator[StreamChunk]:
        client = self._get_client()
        stream_usage: Usage | None = None
        stream_cost: float | None = None
        model = body.get("model", "")

        async with client.stream("POST", url, json=body, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    yield StreamChunk(content="", done=True, usage=stream_usage, cost=stream_cost)
                    return
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                # Final chunk with usage data (empty choices, usage present)
                usage_data = data.get("usage")
                if usage_data and not data.get("choices"):
                    stream_usage = Usage(
                        prompt_tokens=usage_data.get("prompt_tokens", 0),
                        completion_tokens=usage_data.get("completion_tokens", 0),
                        total_tokens=usage_data.get("total_tokens", 0),
                    )
                    stream_cost = calculate_cost(self.name, model, stream_usage)
                    continue

                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield StreamChunk(content=content)

    def _parse_response(self, data: dict, model: str) -> CompletionResponse:
        choices = data.get("choices", [])
        content = ""
        reasoning_content = None

        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "") or ""
            reasoning_content = message.get("reasoning_content")

        usage_data = data.get("usage")
        usage = None
        if usage_data:
            reasoning_tokens = None
            details = usage_data.get("completion_tokens_details")
            if details:
                reasoning_tokens = details.get("reasoning_tokens")

            prompt_details = usage_data.get("prompt_tokens_details")
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                cached_tokens=prompt_details.get("cached_tokens") if prompt_details else None,
                reasoning_tokens=reasoning_tokens,
            )

        cost = calculate_cost(self.name, model, usage) if usage else None

        return CompletionResponse(
            content=content,
            reasoning_content=reasoning_content,
            usage=usage,
            cost=cost,
            raw=data,
        )

    def count_tokens(self, text: str, model: str) -> int:
        return default_count_tokens(text, model)

    def get_model_info(self, model: str) -> ModelInfo | None:
        return _get_model_info(self.name, model)

    async def list_models(self) -> list[str]:
        headers = self._build_headers()
        try:
            client = self._get_client()
            if self.name == "ollama":
                return await self._list_ollama_models(client, headers)

            resp = await client.get(f"{self.api_base}/models", headers=headers)
            resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])]
        except Exception as e:
            logger.debug("Failed to list models for %s: %s", self.name, e)
            return []

    async def _list_ollama_models(self, client: httpx.AsyncClient, headers: dict) -> list[str]:
        """List Ollama models and register context lengths from /api/show."""
        from .model_registry import register_model

        base = self.api_base.replace("/v1", "")
        resp = await client.get(f"{base}/api/tags", headers=headers)
        resp.raise_for_status()
        names = [m["name"] for m in resp.json().get("models", [])]

        for name in names:
            try:
                show_resp = await client.post(f"{base}/api/show", json={"name": name}, headers=headers)
                if show_resp.status_code == 200:
                    data = show_resp.json()
                    model_info = data.get("model_info", {})

                    # Context length is under {arch}.context_length (e.g., qwen2.context_length)
                    ctx_len = None
                    for key, val in model_info.items():
                        if key.endswith(".context_length") and isinstance(val, int):
                            ctx_len = val
                            break

                    # Fallback: check num_ctx in parameters string
                    if not ctx_len:
                        for line in data.get("parameters", "").splitlines():
                            if "num_ctx" in line:
                                try:
                                    ctx_len = int(line.split()[-1])
                                except (ValueError, IndexError):
                                    pass

                    if ctx_len:
                        register_model("ollama", name, ModelInfo(max_input_tokens=ctx_len))
            except Exception:
                pass

        return names


def create_provider(name: str = "openai", **kwargs: Any) -> OpenAICompatProvider:
    """Factory function — same interface as external plugin entry points."""
    config = _PROVIDER_CONFIGS.get(name, {})

    api_base = kwargs.get("api_base") or config.get("api_base")
    if not api_base:
        env_key = config.get("api_base_env")
        api_base = os.getenv(env_key) if env_key else None
    if not api_base:
        api_base = config.get("api_base_default", "https://api.openai.com/v1")

    api_key = kwargs.get("api_key") or config.get("api_key")
    if not api_key:
        env_key = config.get("api_key_env")
        api_key = os.getenv(env_key) if env_key else None

    extra_headers = {**config.get("extra_headers", {}), **kwargs.get("extra_headers", {})}

    return OpenAICompatProvider(
        name=name,
        api_base=api_base,
        api_key=api_key,
        extra_headers=extra_headers,
    )
