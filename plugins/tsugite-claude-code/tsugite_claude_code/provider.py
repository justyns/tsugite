"""Claude Code provider — routes LLM calls through `claude --print` subprocess."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from tsugite.exceptions import AgentExecutionError
from tsugite.providers.base import CompletionResponse, ModelInfo, StreamChunk, Usage, default_count_tokens
from tsugite.providers.model_registry import get_model_info as _get_model_info
from tsugite.providers.model_registry import register_aliases, register_models

logger = logging.getLogger(__name__)

_CLAUDE_CODE_EFFORT_LEVELS = ["low", "medium", "high", "xhigh", "max"]

_CLAUDE_CODE_MODELS: dict[str, ModelInfo] = {
    "claude_code/claude-fable-5": ModelInfo(
        max_input_tokens=1_000_000, supports_vision=True, supported_effort_levels=_CLAUDE_CODE_EFFORT_LEVELS
    ),
    "claude_code/claude-opus-5": ModelInfo(
        max_input_tokens=1_000_000, supports_vision=True, supported_effort_levels=_CLAUDE_CODE_EFFORT_LEVELS
    ),
    "claude_code/claude-opus-4-8": ModelInfo(
        max_input_tokens=1_000_000, supports_vision=True, supported_effort_levels=_CLAUDE_CODE_EFFORT_LEVELS
    ),
    "claude_code/claude-opus-4-7": ModelInfo(
        max_input_tokens=1_000_000, supports_vision=True, supported_effort_levels=_CLAUDE_CODE_EFFORT_LEVELS
    ),
    "claude_code/claude-opus-4-6": ModelInfo(
        max_input_tokens=1_000_000, supports_vision=True, supported_effort_levels=_CLAUDE_CODE_EFFORT_LEVELS
    ),
    "claude_code/claude-sonnet-5": ModelInfo(
        max_input_tokens=1_000_000, supports_vision=True, supported_effort_levels=_CLAUDE_CODE_EFFORT_LEVELS
    ),
    "claude_code/claude-sonnet-4-6": ModelInfo(
        max_input_tokens=1_000_000, supports_vision=True, supported_effort_levels=_CLAUDE_CODE_EFFORT_LEVELS
    ),
    "claude_code/claude-haiku-4-5-20251001": ModelInfo(
        max_input_tokens=200_000, supports_vision=True, supported_effort_levels=_CLAUDE_CODE_EFFORT_LEVELS
    ),
}

# Bare tier aliases track the newest model in that tier; version-pinned aliases
# stay for reproducible configs. Exception: opus stays on 4-8. Opus 5's replies
# through Claude Code's resume path can leave an empty assistant block that wedges
# the sidecar transcript (400 "text content blocks must be non-empty"), and the
# fresh-session fallback only covers the first send, so opus-5 is opt-in via its
# pinned alias until a mid-conversation poison recovers cleanly.
_ALIASES = {
    "fable": "claude-fable-5",
    "fable-5": "claude-fable-5",
    "opus": "claude-opus-4-8",
    "opus-5": "claude-opus-5",
    "opus-4-8": "claude-opus-4-8",
    "opus-4-7": "claude-opus-4-7",
    "opus-4-6": "claude-opus-4-6",
    "sonnet": "claude-sonnet-5",
    "sonnet-5": "claude-sonnet-5",
    "sonnet-4-6": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}


def _is_unresumable_history_error(exc: BaseException) -> bool:
    """A 400 on a resume replay means the sidecar transcript itself is malformed
    (e.g. an empty text content block) — retrying the same resume can never
    succeed. Anything else (prompt too long, 429/529, execution errors) must
    keep surfacing so the daemon's existing retry paths handle it."""
    return "API Error: 400" in str(exc)


def _raise_if_error(result_event: dict) -> None:
    """Translate a Claude CLI error result into AgentExecutionError.

    The CLI reports failures (context overflow, max-turns, etc.) as a result
    event with is_error=true and a non-success subtype, NOT as a non-zero exit
    or stderr. Without this conversion the failure text reaches the user as
    the assistant's reply and bypasses the daemon's prompt-too-long retry path.
    """
    if not result_event.get("is_error"):
        return
    text = result_event.get("text") or "Claude Code returned an error result"
    subtype = result_event.get("subtype") or "error"
    raise AgentExecutionError(f"{text} (subtype={subtype})")


class ClaudeCodeProvider:
    """Provider that manages a persistent `claude` CLI subprocess.

    Stateful: the subprocess persists across acompletion() calls within a session.
    First call starts the process; subsequent calls send observations to it.
    Call stop() to clean up the subprocess when done.
    """

    cacheable = False
    models_are_definitive = True  # CLI exposes a fixed model set; unlisted ids are typos

    def __init__(self, name: str = "claude_code"):
        self.name = name
        self._process = None
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
        register_aliases(self.name, _ALIASES)

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
        from tsugite_claude_code.process import ClaudeCodeProcess

        resolved_model = _ALIASES.get(model, model)
        if resolved_model != model and self._resolved_model != resolved_model:
            logger.info("claude_code model alias %r -> %s", model, resolved_model)
        self._resolved_model = resolved_model

        first_turn = self._process is None
        if first_turn:
            self._process = ClaudeCodeProcess()
            system_prompt = ""
            if messages and messages[0].get("role") == "system":
                system_prompt = messages[0]["content"]
                messages = messages[1:]

            # Kept so the poisoned-resume fallback can restart fresh and
            # rebuild the first message with serialized history included.
            self._start_system_prompt = system_prompt
            self._start_effort = kwargs.get("reasoning_effort")
            self._first_messages = messages

            await self._process.start(
                model=resolved_model,
                system_prompt=system_prompt,
                resume_session=self._resume_session,
                effort=kwargs.get("reasoning_effort"),
            )
            user_content = self._build_first_message(messages)
        else:
            # Subsequent turns: subprocess has context, send the last observation
            user_content = messages[-1]["content"] if messages else ""

        # Never send empty content: the CLI would persist an empty text block
        # into its sidecar transcript, and the next --resume of that transcript
        # is rejected wholesale (400 text content blocks must be non-empty). A
        # block list (image turn) is never empty and skips this guard.
        if isinstance(user_content, str) and not user_content.strip():
            user_content = "(empty message)"

        if stream:
            return self._stream(user_content, resume_fallback=first_turn)

        try:
            return await self._collect(user_content)
        except AgentExecutionError as e:
            if not self._should_fallback_to_fresh_session(first_turn, e):
                raise
            retry_content = await self._fallback_to_fresh_session(e)
            return await self._collect(retry_content)

    def _should_fallback_to_fresh_session(self, first_turn: bool, err: BaseException) -> bool:
        """Only a first-send 400 on a live resume is a poisoned transcript we can recover from."""
        return bool(first_turn and self._resume_session and _is_unresumable_history_error(err))

    async def _fallback_to_fresh_session(self, err: BaseException) -> str:
        """Sever a poisoned resume: replace the subprocess with a fresh session
        seeded from tsugite's serialized history, and return the rebuilt first
        message. The serializer renders "Role: content" lines, so a turn whose
        raw content was empty survives as a non-empty block."""
        from tsugite_claude_code.process import ClaudeCodeProcess

        logger.warning(
            "Resume of provider session %s was rejected (%s); starting a fresh session seeded from serialized history",
            self._resume_session,
            err,
        )
        await self._process.stop()
        self._resume_session = None
        self._resume_after_compaction = False
        self._process = ClaudeCodeProcess()
        await self._process.start(
            model=self._resolved_model,
            system_prompt=self._start_system_prompt,
            resume_session=None,
            effort=self._start_effort,
        )
        return self._build_first_message(self._first_messages)

    async def _collect(self, user_content: str) -> CompletionResponse:
        """Send message and collect full response."""
        accumulated = ""
        usage = Usage()
        cost = 0.0

        async for event in self._process.send_message(user_content):
            if event["type"] == "text_delta":
                accumulated += event["text"]
            elif event["type"] == "result":
                _raise_if_error(event)
                if not accumulated:
                    accumulated = event.get("text", "")
                cost = self._cost_delta(event.get("cost_usd") or 0.0)
                usage = self._extract_usage(event)

        return CompletionResponse(
            content=accumulated,
            usage=usage,
            cost=cost,
        )

    async def _stream(self, user_content: str, resume_fallback: bool = False) -> AsyncIterator[StreamChunk]:
        """Send message and yield streaming chunks."""
        usage = Usage()
        cost = 0.0
        yielded = False

        try:
            async for event in self._process.send_message(user_content):
                if event["type"] == "text_delta":
                    yielded = True
                    yield StreamChunk(content=event["text"])
                elif event["type"] == "result":
                    _raise_if_error(event)
                    cost = self._cost_delta(event.get("cost_usd") or 0.0)
                    usage = self._extract_usage(event)
        except AgentExecutionError as e:
            # A poisoned resume fails wholesale before generating anything, so
            # falling back is only safe when no content has been streamed yet.
            if yielded or not self._should_fallback_to_fresh_session(resume_fallback, e):
                raise
            retry_content = await self._fallback_to_fresh_session(e)
            async for chunk in self._stream(retry_content):
                yield chunk
            return

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

    def _build_first_message(self, messages: list[dict]) -> str | list[dict]:
        """Build the first user message, inlining attachments, skills, and history.

        Returns a bare string for text-only turns (transcripts unchanged). When
        image attachments are present, returns an Anthropic content-block list
        (one text block carrying the inlined context/history/task, plus one image
        block per attachment); the CLI forwards those blocks to the API.
        """
        parts = []

        include_context = not self._resume_session or self._resume_after_compaction
        if include_context:
            from tsugite.context_block import build_context_el

            context = build_context_el(self._attachments, self._skills, skill_char_limit=4000)
            if context is not None:
                parts.append(context.render() + "\n")

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

        text = "\n".join(parts)

        # Image blocks are NOT gated on include_context: a resumed session (every
        # ongoing daemon chat turn) has include_context false, but this turn's
        # uploaded image belongs to this turn and isn't in the CLI transcript yet.
        # The gate exists only to avoid re-inlining standing TEXT context.
        image_blocks = self._image_blocks()
        if not image_blocks:
            return text
        blocks: list[dict] = [{"type": "image", "source": src} for src in image_blocks]
        # A leading, non-empty text block keeps the task/context ahead of the
        # images; an empty text block would be rejected (400 must be non-empty).
        if text.strip():
            blocks.insert(0, {"type": "text", "text": text})
        return blocks

    def _image_blocks(self) -> list[dict]:
        """base64 image `source` payloads for the API-supported image attachments.

        Unsupported image types (svg/bmp/tiff) are skipped as defense-in-depth --
        the daemon already routes them to the workspace-only path, so in practice
        only standing agent-file images (essentially nonexistent) hit this filter.
        """
        from tsugite.attachments.base import SUPPORTED_INLINE_IMAGE_MEDIA_TYPES, AttachmentContentType

        sources = []
        for att in self._attachments:
            if att.content_type != AttachmentContentType.IMAGE or not att.content:
                continue
            if att.mime_type not in SUPPORTED_INLINE_IMAGE_MEDIA_TYPES:
                continue
            sources.append({"type": "base64", "media_type": att.mime_type, "data": att.content})
        return sources

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

    def count_tokens(self, text: str, model: str) -> int:
        return default_count_tokens(text, model)

    def get_model_info(self, model: str) -> ModelInfo | None:
        resolved = _ALIASES.get(model, model)
        return _get_model_info(self.name, resolved)

    async def list_models(self) -> list[str]:
        return list(_ALIASES.keys())


def create_provider(name: str = "claude_code", **kwargs: Any) -> ClaudeCodeProvider:
    return ClaudeCodeProvider(name=name)
