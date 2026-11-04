"""Cursor provider: Cursor models through the Cursor Python SDK (`cursor-sdk`), with Cursor's built-in tools disabled."""

from __future__ import annotations

import contextlib
import logging
import re
from pathlib import Path
from typing import Any, AsyncIterator

from tsugite.config import get_xdg_data_path
from tsugite.exceptions import AgentExecutionError
from tsugite.providers.base import (
    CompletionResponse,
    ModelInfo,
    StreamChunk,
    Usage,
    api_key_secret_name,
    default_count_tokens,
    resolve_api_key,
)

logger = logging.getLogger(__name__)

# The SDK's model catalog has no context window. A `context=` model parameter overrides this default.
_DEFAULT_CONTEXT_TOKENS = 200_000
_CONTEXT_PARAM = re.compile(r"^(\d+)([km])$", re.IGNORECASE)
_MODEL = re.compile(r"^(?P<id>[^\[\]]+?)(?:\[(?P<params>[^\]]*)\])?$")


def parse_model(model: str) -> dict:
    """`composer-2.5` or `claude-opus-4-8[context=1m,effort=high]` into an SDK ModelSelection."""
    match = _MODEL.match(model.strip())
    if not match:
        raise ValueError(f"Invalid Cursor model {model!r}")
    model_id = "default" if match["id"] == "auto" else match["id"]
    params = []
    for item in filter(None, (p.strip() for p in (match["params"] or "").split(","))):
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Invalid Cursor model parameter {item!r} in {model!r}, expected key=value")
        params.append({"id": key.strip(), "value": value.strip()})
    return {"id": model_id, **({"params": params} if params else {})}


def session_cwd() -> Path:
    """Empty directory every Cursor agent runs in. The SDK scopes resumable agents to their cwd."""
    path = get_xdg_data_path("cursor/cwd")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sdk():
    import cursor_sdk

    return cursor_sdk


def _api_key() -> str:
    if key := resolve_api_key(CursorProvider.api_key_env):
        return key
    raise AgentExecutionError(
        f"No Cursor API key found. Create one in the Cursor dashboard and set {CursorProvider.api_key_env} "
        f"or the {api_key_secret_name(CursorProvider.api_key_env)} secret."
    )


@contextlib.contextmanager
def _cursor_errors():
    sdk = _sdk()
    try:
        yield
    except sdk.AuthenticationError as e:
        raise AgentExecutionError(f"Cursor rejected the API key: {e}") from e
    except sdk.CursorSDKError as e:
        raise AgentExecutionError(f"Cursor error: {e}") from e


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(
            b if isinstance(b, str) else b.get("text", "")
            for b in content
            if isinstance(b, str) or b.get("type") == "text"
        )
    return content or ""


class CursorProvider:
    """One SDK bridge process, and so one Cursor agent, per instance."""

    api_key_env = "CURSOR_API_KEY"
    cacheable = False
    models_are_definitive = False  # the catalog depends on the account

    def __init__(self, name: str = "cursor"):
        self.name = name
        self._client = None
        self._agent = None
        self._context_window: int | None = None
        self._attachments: list = []
        self._skills: list = []
        self._previous_messages: list = []
        self._resume_session: str | None = None
        self._resume_after_compaction = False

    def set_context(self, **kwargs: Any) -> None:
        self._attachments = kwargs.get("attachments", [])
        self._skills = kwargs.get("skills", [])
        self._previous_messages = kwargs.get("previous_messages", [])
        self._resume_session = kwargs.get("resume_session")
        self._resume_after_compaction = kwargs.get("resume_after_compaction", False)

    def get_state(self) -> dict:
        return {"session_id": self._agent.agent_id if self._agent else None, "context_window": self._context_window}

    async def acompletion(
        self,
        messages: list[dict],
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse | AsyncIterator[StreamChunk]:
        if self._agent is None:
            system_prompt = ""
            if messages[0].get("role") == "system":
                system_prompt = _text(messages[0]["content"])
                messages = messages[1:]
            self._context_window = self.get_model_info(model).max_input_tokens
            await self._open(model)
            text = self._build_first_message(messages, system_prompt)
        else:
            text = _text(messages[-1]["content"])

        chunks = self._stream(text)
        return chunks if stream else await self._collect(chunks)

    async def _open(self, model: str) -> None:
        sdk = _sdk()
        key = _api_key()
        cwd = str(session_cwd())
        with _cursor_errors():
            client = await sdk.AsyncCursorClient.launch_bridge(workspace=cwd)
            try:
                options = {
                    "api_key": key,
                    "name": "tsugite",
                    "model": await self._with_default_params(client, key, parse_model(model)),
                    "tools": [],
                    "local": {"cwd": cwd, "setting_sources": []},
                }
                if self._resume_session:
                    try:
                        self._agent = await sdk.AsyncAgent.resume(self._resume_session, options, client=client)
                    except sdk.AgentNotFoundError as e:
                        logger.warning(
                            "Cursor agent %s cannot be resumed (%s), starting a new one with the conversation history",
                            self._resume_session,
                            e,
                        )
                        self._resume_session = None
                if self._agent is None:
                    self._agent = await sdk.AsyncAgent.create(options, client=client)
            except BaseException:
                await client.aclose()
                raise
        self._client = client

    @staticmethod
    async def _with_default_params(client, key: str, model: dict) -> dict:
        """Fill in the catalog's default variant params. Some plans refuse a bare id ("Model unavailable on Plus ... with fixed settings")."""
        if "params" in model or model["id"] == "default":
            return model
        catalog = await _sdk().AsyncCursor.models.list(client=client, api_key=key)
        entry = next((m for m in catalog if m.id == model["id"]), None)
        variant = next((v for v in entry.variants if v.is_default), None) if entry else None
        if variant is None:
            return model
        return {"id": entry.id, "params": [{"id": p.id, "value": p.value} for p in variant.params]}

    async def _collect(self, chunks: AsyncIterator[StreamChunk]) -> CompletionResponse:
        content, reasoning, usage = [], [], None
        async for chunk in chunks:
            content.append(chunk.content)
            reasoning.append(chunk.reasoning_content)
            if chunk.done:
                usage = chunk.usage
        return CompletionResponse(content="".join(content), reasoning_content="".join(reasoning) or None, usage=usage)

    async def _stream(self, text: str) -> AsyncIterator[StreamChunk]:
        sdk = _sdk()
        with _cursor_errors():
            # run.events() yields interaction updates only while an on_delta callback is registered.
            run = await self._agent.send(text, sdk.SendOptions(on_delta=lambda update: None))
            result = None
            try:
                async for event in run.events():
                    update = event.interaction_update
                    if isinstance(update, sdk.TextDeltaUpdate):
                        yield StreamChunk(content=update.text)
                    elif isinstance(update, sdk.ThinkingDeltaUpdate):
                        yield StreamChunk(reasoning_content=update.text)
                    elif isinstance(update, (sdk.ToolCallStartedUpdate, sdk.PartialToolCallUpdate)):
                        raise AgentExecutionError(
                            f"Cursor started a tool call ({update.tool_call.get('type', 'unknown')}) although tools are disabled."
                        )
                result = await run.wait()
            finally:
                if result is None and run.status not in ("finished", "error", "cancelled", "expired"):
                    try:
                        await run.cancel()
                    except sdk.CursorSDKError as e:
                        logger.warning("Cursor run cancel failed: %s", e)
        if result.status != "finished":
            raise AgentExecutionError(f"Cursor run {result.status}: {result.error_message or 'no details'}")
        usage = result.usage
        if usage is not None:
            # input_tokens includes cache reads.
            usage = Usage(
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
                cached_tokens=usage.cache_read_tokens,
                reasoning_tokens=usage.reasoning_tokens,
            )
        yield StreamChunk(done=True, usage=usage)

    def _build_first_message(self, messages: list[dict], system_prompt: str) -> str:
        """Latest user message, preceded by the system prompt, context, history and delivery cards the agent has not seen."""
        from tsugite.context_block import build_context_el
        from tsugite.prompt_xml import El, Raw

        parts = []
        if not self._resume_session or self._resume_after_compaction:
            if context := build_context_el(self._attachments, self._skills, skill_char_limit=4000):
                parts.append(context.render() + "\n")
        if self._previous_messages and not self._resume_session:
            kept = self._trim_history(self._previous_messages, self._context_window // 2)
            dropped = len(self._previous_messages) - len(kept)
            lines = [f"{m.get('role', 'unknown').capitalize()}: {m.get('content', '')}" for m in kept]
            note = f"{dropped} older messages omitted for context" if dropped else None
            history = El("conversation_history", [Raw("\n\n".join(lines))], attrs={"note": note})
            parts.append(history.render() + "\n")
        if self._resume_session:
            parts.extend(self._unseen_delivery_cards())
        for message in reversed(messages):
            if message.get("role") == "user":
                parts.append(_text(message["content"]))
                break
        text = "\n".join(parts)
        if system_prompt and not self._resume_session:
            text = El("tsugite_system_prompt", [Raw(system_prompt)]).render() + "\n\n" + text
        return text

    @staticmethod
    def _trim_history(history: list[dict], budget: int) -> list[dict]:
        kept, used = [], 0
        for message in reversed(history):
            content = message.get("content", "")
            estimate = len(content) // 4 if isinstance(content, str) else 100
            if used + estimate > budget and kept:
                break
            kept.append(message)
            used += estimate
        return kept[::-1]

    def _unseen_delivery_cards(self) -> list[str]:
        """Delivery cards other than needs_ack recorded after the last assistant reply. The resumed agent has not seen them."""
        last_reply = max((i for i, m in enumerate(self._previous_messages) if m.get("role") == "assistant"), default=-1)
        cards = []
        for message in self._previous_messages[last_reply + 1 :]:
            content = message.get("content")
            if isinstance(content, str) and content.startswith("<tsugite_delivery "):
                if 'kind="needs_ack"' not in content.split("\n", 1)[0]:
                    cards.append(content + "\n")
        return cards

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
            self._agent = None

    def count_tokens(self, text: str, model: str) -> int:
        return default_count_tokens(text, model)

    def get_model_info(self, model: str) -> ModelInfo:
        context = next((p["value"] for p in parse_model(model).get("params", []) if p["id"] == "context"), "")
        match = _CONTEXT_PARAM.match(context)
        if not match:
            return ModelInfo(max_input_tokens=_DEFAULT_CONTEXT_TOKENS)
        scale = 1_000_000 if match[2].lower() == "m" else 1_000
        return ModelInfo(max_input_tokens=int(match[1]) * scale)

    async def list_models(self) -> list[str]:
        sdk = _sdk()
        key = _api_key()
        with _cursor_errors():
            async with await sdk.AsyncCursorClient.launch_bridge(workspace=str(session_cwd())) as client:
                return [m.id for m in await sdk.AsyncCursor.models.list(client=client, api_key=key)]


def create_provider(name: str = "cursor", **kwargs: Any) -> CursorProvider:
    return CursorProvider(name=name)
