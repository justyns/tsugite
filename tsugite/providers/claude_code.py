"""Claude Code provider — routes LLM calls through `claude --print` subprocess."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from .base import CompletionResponse, ModelInfo, StreamChunk, Usage, default_count_tokens
from .model_registry import get_model_info as _get_model_info
from .model_registry import register_models

logger = logging.getLogger(__name__)

_CLAUDE_CODE_MODELS: dict[str, ModelInfo] = {
    "claude_code/claude-opus-4-7": ModelInfo(max_input_tokens=1_000_000, supports_vision=True),
    "claude_code/claude-opus-4-6": ModelInfo(max_input_tokens=1_000_000, supports_vision=True),
    "claude_code/claude-sonnet-4-6": ModelInfo(max_input_tokens=1_000_000, supports_vision=True),
    "claude_code/claude-haiku-4-5-20251001": ModelInfo(max_input_tokens=200_000, supports_vision=True),
}

_ALIASES = {
    "opus": "claude-opus-4-7",
    "opus-4-7": "claude-opus-4-7",
    "opus-4-6": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}


class ClaudeCodeProvider:
    """Provider that manages a persistent `claude` CLI subprocess.

    Stateful: the subprocess persists across acompletion() calls within a session.
    First call starts the process; subsequent calls send observations to it.
    Call stop() to clean up the subprocess when done.
    """

    cacheable = False

    def __init__(self, name: str = "claude_code"):
        self.name = name
        self._process = None
        self._turn_count = 0
        self._resolved_model: str | None = None

        # Context set via set_context()
        self._attachments = []
        self._skills = []
        self._resume_session = None
        self._resume_after_compaction = False
        self._previous_messages = []

        # Session state
        self._session_id: str | None = None
        self._compacted: bool = False
        self._context_window: int | None = None
        self._cache_creation_tokens: int = 0
        self._cache_read_tokens: int = 0
        self._cumulative_cost: float = 0.0

        register_models(_CLAUDE_CODE_MODELS)

    def set_context(self, **kwargs: Any) -> None:
        self._attachments = kwargs.get("attachments", [])
        self._skills = kwargs.get("skills", [])
        self._resume_session = kwargs.get("resume_session")
        self._resume_after_compaction = kwargs.get("resume_after_compaction", False)
        self._previous_messages = kwargs.get("previous_messages", [])

    def get_state(self) -> dict | None:
        return {
            "session_id": self._session_id,
            "compacted": self._compacted,
            "context_window": self._context_window,
            "cache_creation_tokens": self._cache_creation_tokens,
            "cache_read_tokens": self._cache_read_tokens,
        }

    async def acompletion(
        self,
        messages: list[dict],
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse | AsyncIterator[StreamChunk]:
        from tsugite.core.claude_code import ClaudeCodeProcess

        resolved_model = _ALIASES.get(model, model)
        if resolved_model != model and self._resolved_model != resolved_model:
            logger.info("claude_code model alias %r -> %s", model, resolved_model)
        self._resolved_model = resolved_model

        if self._process is None:
            self._process = ClaudeCodeProcess()
            system_prompt = ""
            if messages and messages[0].get("role") == "system":
                system_prompt = messages[0]["content"]
                messages = messages[1:]

            await self._process.start(
                model=resolved_model,
                system_prompt=system_prompt,
                resume_session=self._resume_session,
            )
            user_content = self._build_first_message(messages)
        else:
            # Subsequent turns: subprocess has context, send the last observation
            user_content = messages[-1]["content"] if messages else ""

        self._turn_count += 1

        if stream:
            return self._stream(user_content)

        return await self._collect(user_content)

    async def _collect(self, user_content: str) -> CompletionResponse:
        """Send message and collect full response."""
        accumulated = ""
        usage = Usage()
        cost = 0.0

        async for event in self._process.send_message(user_content):
            if event["type"] == "text_delta":
                accumulated += event["text"]
            elif event["type"] == "result":
                if not accumulated:
                    accumulated = event.get("text", "")
                cost = self._cost_delta(event.get("cost_usd") or 0.0)
                usage = self._extract_usage(event)

        return CompletionResponse(
            content=accumulated,
            usage=usage,
            cost=cost,
        )

    async def _stream(self, user_content: str) -> AsyncIterator[StreamChunk]:
        """Send message and yield streaming chunks."""
        usage = Usage()
        cost = 0.0

        async for event in self._process.send_message(user_content):
            if event["type"] == "text_delta":
                yield StreamChunk(content=event["text"])
            elif event["type"] == "result":
                cost = self._cost_delta(event.get("cost_usd") or 0.0)
                usage = self._extract_usage(event)

        yield StreamChunk(content="", done=True, usage=usage, cost=cost)

    def _cost_delta(self, cumulative_cost: float) -> float:
        """Convert Claude CLI's cumulative cost to a per-turn delta."""
        delta = cumulative_cost - self._cumulative_cost
        self._cumulative_cost = cumulative_cost
        return max(delta, 0.0)

    def _extract_usage(self, event: dict) -> Usage:
        """Extract usage from a subprocess result event and update session state."""
        input_tokens = event.get("input_tokens") or 0
        cache_creation = event.get("cache_creation_input_tokens") or 0
        cache_read = event.get("cache_read_input_tokens") or 0
        output_tokens = event.get("output_tokens") or 0

        self._cache_creation_tokens += cache_creation
        self._cache_read_tokens += cache_read
        self._session_id = event.get("session_id", self._session_id)
        if event.get("context_window"):
            self._context_window = event["context_window"]

        return Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + cache_creation + cache_read + output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
        )

    def _build_first_message(self, messages: list[dict]) -> str:
        """Build the first user message, inlining attachments, skills, and history."""
        parts = []

        include_context = not self._resume_session or self._resume_after_compaction
        if include_context and (self._attachments or self._skills):
            context_parts = []
            for att in self._attachments:
                from tsugite.attachments.base import AttachmentContentType

                if att.content_type == AttachmentContentType.TEXT:
                    context_parts.append(f'<attachment name="{att.name}">')
                    context_parts.append(att.content)
                    context_parts.append("</attachment>")
            for skill in self._skills:
                content = skill.content
                if len(content) > 4000:
                    content = content[:4000] + "\n... (truncated)"
                context_parts.append(f'<skill name="{skill.name}">')
                context_parts.append(content)
                context_parts.append("</skill>")
            if context_parts:
                parts.append("<context>\n" + "\n".join(context_parts) + "\n</context>\n")

        if self._previous_messages and not self._resume_session:
            budget = self._get_history_budget()
            trimmed = self._trim_to_budget(self._previous_messages, budget)
            dropped = len(self._previous_messages) - len(trimmed)
            history_lines = [f"{msg.get('role', 'unknown').capitalize()}: {msg.get('content', '')}" for msg in trimmed]
            header = "<conversation_history"
            if dropped > 0:
                header += f' note="{dropped} older messages omitted for context"'
            header += ">"
            parts.append(header + "\n" + "\n\n".join(history_lines) + "\n</conversation_history>\n")

        # Add the task (last user message — earlier user messages are context/history)
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg["content"]
                if isinstance(content, list):
                    content = "\n".join(
                        b if isinstance(b, str) else b.get("text", "")
                        for b in content
                        if isinstance(b, str) or b.get("type") == "text"
                    )
                parts.append(content)
                break

        return "\n".join(parts)

    def _get_history_budget(self) -> int:
        info = self.get_model_info(self._resolved_model) if self._resolved_model else None
        context_limit = info.max_input_tokens if info else 200_000
        return context_limit // 2

    @staticmethod
    def _trim_to_budget(messages: list[dict], budget_tokens: int) -> list[dict]:
        """Keep the most recent messages that fit within a token budget."""
        kept = []
        used = 0
        for msg in reversed(messages):
            content = msg.get("content", "")
            est = len(content) // 4 if isinstance(content, str) else 100
            if used + est > budget_tokens and kept:
                break
            kept.append(msg)
            used += est
        kept.reverse()
        return kept

    async def stop(self) -> None:
        if self._process:
            self._session_id = self._process.session_id
            self._compacted = self._process.compacted
            await self._process.stop()
            self._process = None
            self._turn_count = 0

    def count_tokens(self, text: str, model: str) -> int:
        return default_count_tokens(text, model)

    def get_model_info(self, model: str) -> ModelInfo | None:
        resolved = _ALIASES.get(model, model)
        return _get_model_info(self.name, resolved)

    async def list_models(self) -> list[str]:
        return list(_ALIASES.keys())


def create_provider(name: str = "claude_code", **kwargs: Any) -> ClaudeCodeProvider:
    return ClaudeCodeProvider(name=name)
