"""Codex CLI provider: ChatGPT subscription auth via OpenAI Responses API."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

import httpx

from tsugite.providers.base import CompletionResponse, ModelInfo, StreamChunk, Usage, default_count_tokens
from tsugite.providers.model_registry import get_model_info as _get_model_info
from tsugite.providers.model_registry import register_models
from tsugite.user_agent import set_user_agent_header
from tsugite_codex_cli.auth import CodexAuthError, CodexAuthStore

logger = logging.getLogger(__name__)

API_BASE = "https://chatgpt.com/backend-api/codex"
_REASONING_LEVELS = ["low", "medium", "high"]


def _codex_model_info(max_input_tokens: int) -> ModelInfo:
    # Subscription billing means per-token cost is not surfaced here.
    return ModelInfo(
        max_input_tokens=max_input_tokens,
        max_output_tokens=128_000,
        input_cost_per_million=0.0,
        output_cost_per_million=0.0,
        supports_vision=True,
        supports_reasoning=True,
        supported_effort_levels=_REASONING_LEVELS,
    )


_CODEX_CLI_MODELS: dict[str, ModelInfo] = {
    "codex_cli/gpt-5.4": _codex_model_info(1_050_000),
    "codex_cli/gpt-5.4-mini": _codex_model_info(272_000),
    "codex_cli/gpt-5.4-nano": _codex_model_info(272_000),
}

# Used when /models is unreachable; kept in sync with the registry above.
_FALLBACK_MODELS = [k.split("/", 1)[1] for k in _CODEX_CLI_MODELS]


class CodexResponsesError(Exception):
    """Raised when chatgpt.com/backend-api/codex/responses returns a 4xx/5xx error."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def _translate_content_block(block: Any) -> Any:
    """Chat-Completions-shape block → Responses input_* shape (request side only)."""
    if not isinstance(block, dict):
        return block
    btype = block.get("type")
    if btype == "text":
        return {"type": "input_text", "text": block.get("text", "")}
    if btype == "image_url":
        image_url = block.get("image_url")
        if isinstance(image_url, dict):
            image_url = image_url.get("url", "")
        return {"type": "input_image", "image_url": image_url}
    if btype == "file":
        file_field = block.get("file") or {}
        return {"type": "input_file", **file_field}
    return block


def _translate_message_content(content: Any) -> Any:
    if isinstance(content, list):
        return [_translate_content_block(b) for b in content]
    return content


def _normalise_kwargs(kwargs: dict) -> dict:
    """Map Chat-Completions-style kwargs onto Responses request fields."""
    body: dict[str, Any] = {}
    for key, value in kwargs.items():
        if value is None or key.startswith("_"):
            continue
        if key == "max_tokens":
            body["max_output_tokens"] = value
        elif key == "reasoning_effort":
            body["reasoning"] = {"effort": value}
        elif key == "response_format":
            body["text"] = {"format": value}
        elif key in {"temperature", "top_p", "stop", "presence_penalty", "frequency_penalty"}:
            body[key] = value
        else:
            logger.debug("codex_cli: dropping unknown kwarg %s for /responses", key)
    return body


def _build_usage(usage_in: dict | None) -> Usage:
    if not usage_in:
        return Usage()
    input_tokens = int(usage_in.get("input_tokens") or 0)
    output_tokens = int(usage_in.get("output_tokens") or 0)
    total = usage_in.get("total_tokens")
    total_tokens = int(total) if total is not None else input_tokens + output_tokens
    details = usage_in.get("output_tokens_details") or {}
    reasoning_tokens = details.get("reasoning_tokens")
    return Usage(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=int(reasoning_tokens) if reasoning_tokens is not None else None,
    )


class CodexCliProvider:
    """ChatGPT subscription provider routing through the Codex Responses API."""

    # Stateless across calls (auth refresh is internal), so the registry can cache one instance.
    cacheable = True

    def __init__(self, name: str = "codex_cli"):
        self.name = name
        self._auth = CodexAuthStore()
        self._client: httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        register_models(_CODEX_CLI_MODELS)

    def _get_client(self) -> httpx.AsyncClient:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if self._client is None or self._client.is_closed or self._client_loop is not current_loop:
            self._client = httpx.AsyncClient(timeout=300)
            self._client_loop = current_loop
        return self._client

    def _build_headers(self, access_token: str, account_id: str) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
            "ChatGPT-Account-ID": account_id,
        }
        set_user_agent_header(headers)
        return headers

    def _build_request_body(self, messages: list[dict], model: str, stream: bool, **kwargs: Any) -> dict:
        instructions_parts: list[str] = []
        input_messages: list[dict] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            instructions_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            instructions_parts.append(block)
                elif isinstance(content, str):
                    instructions_parts.append(content)
                continue
            translated = {"role": role, "content": _translate_message_content(content)}
            input_messages.append(translated)

        # store=false and instructions are both mandatory on the Codex backend
        # (omitting either 400s with `{"detail":"Store must be set to false"}`
        # or `{"detail":"Instructions are required"}`).
        body: dict[str, Any] = {
            "model": model,
            "input": input_messages,
            "store": False,
            "instructions": "\n".join(instructions_parts),
        }
        if stream:
            body["stream"] = True
        body.update(_normalise_kwargs(kwargs))
        return body

    @staticmethod
    def _error_from_response(resp: httpx.Response) -> CodexResponsesError:
        # Caller must `await resp.aread()` first when resp came from client.stream(),
        # otherwise resp.json()/resp.text raise httpx.ResponseNotRead.
        message: str | None = None
        try:
            payload = resp.json()
        except (ValueError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                message = err.get("message")
            elif isinstance(err, str):
                message = err
        if not message:
            try:
                text = (resp.text or "").strip()
            except httpx.ResponseNotRead:
                text = ""
            message = text or f"HTTP {resp.status_code}"
        return CodexResponsesError(message, status_code=resp.status_code)

    @staticmethod
    def _error_from_event(event: dict) -> CodexResponsesError:
        """Translate a `response.failed` / `error` SSE event into a typed exception."""
        message = None
        response_obj = event.get("response") or {}
        err = response_obj.get("error") or event.get("error") or {}
        if isinstance(err, dict):
            message = err.get("message")
        elif isinstance(err, str):
            message = err
        if not message:
            message = event.get("message") or "Codex stream reported an error"
        return CodexResponsesError(message, status_code=200)

    async def acompletion(
        self,
        messages: list[dict],
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse | AsyncIterator[StreamChunk]:
        access_token, account_id = await self._auth.get_access_token()
        # Codex backend mandates stream=true; non-stream callers get the chunks
        # collected into a CompletionResponse below.
        body = self._build_request_body(messages, model, stream=True, **kwargs)
        headers = self._build_headers(access_token, account_id)
        url = f"{API_BASE}/responses"

        if stream:
            return self._stream(url, headers, body)
        return await self._collect_stream(url, headers, body)

    async def _collect_stream(self, url: str, headers: dict, body: dict) -> CompletionResponse:
        parts: list[str] = []
        reasoning_parts: list[str] = []
        usage = Usage()
        async for chunk in self._stream(url, headers, body):
            if chunk.content:
                parts.append(chunk.content)
            if chunk.reasoning_content:
                reasoning_parts.append(chunk.reasoning_content)
            if chunk.done and chunk.usage is not None:
                usage = chunk.usage
        return CompletionResponse(
            content="".join(parts),
            reasoning_content="".join(reasoning_parts) or None,
            usage=usage,
            cost=0.0,
        )

    async def _stream(self, url: str, headers: dict, body: dict) -> AsyncIterator[StreamChunk]:
        client = self._get_client()
        usage: Usage = Usage()
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                raise self._error_from_response(resp)
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type")
                if etype == "response.output_text.delta":
                    delta = event.get("delta") or ""
                    if delta:
                        yield StreamChunk(content=delta)
                elif etype in ("response.reasoning_summary_text.delta", "response.reasoning.delta"):
                    delta = event.get("delta") or ""
                    if delta:
                        yield StreamChunk(reasoning_content=delta)
                elif etype == "response.completed":
                    response_obj = event.get("response") or {}
                    usage = _build_usage(response_obj.get("usage"))
                elif etype in ("response.failed", "response.incomplete", "error"):
                    raise self._error_from_event(event)
        yield StreamChunk(content="", done=True, usage=usage, cost=0.0)

    def count_tokens(self, text: str, model: str) -> int:
        # All current Codex models share the GPT-4o/o-series o200k_base BPE; base's
        # default_count_tokens only matches gpt-4o/o1/o3/o4 prefixes so it would
        # silently fall back to cl100k for gpt-5.x.
        try:
            import tiktoken

            return len(tiktoken.get_encoding("o200k_base").encode(text))
        except Exception:
            return default_count_tokens(text, model)

    def get_model_info(self, model: str) -> ModelInfo | None:
        return _get_model_info(self.name, model)

    def set_context(self, **kwargs: Any) -> None:
        pass

    def get_state(self) -> dict | None:
        return None

    async def stop(self) -> None:
        # No-op: the provider is cached and the httpx client is shared across
        # concurrent agent runs. Closing it here would break in-flight streams
        # on parallel sessions. The client is released when the process exits.
        pass

    async def list_models(self) -> list[str]:
        # Auth lookup is inside the try: model-picker UIs should always get a
        # usable list even before `codex login` has been run.
        try:
            access_token, account_id = await self._auth.get_access_token()
            client = self._get_client()
            resp = await client.get(
                f"{API_BASE}/models?client_version=1.0.0",
                headers=self._build_headers(access_token, account_id),
            )
            resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])]
        except (httpx.HTTPError, KeyError, ValueError, CodexAuthError):
            return list(_FALLBACK_MODELS)


def create_provider(name: str = "codex_cli", **kwargs: Any) -> CodexCliProvider:
    return CodexCliProvider(name=name)
