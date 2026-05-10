"""ACPProvider: routes tsugite completions through an ACP-compatible agent subprocess."""

from __future__ import annotations

from typing import Any, AsyncIterator, Callable

from acp.schema import TextContentBlock

from tsugite.providers.base import CompletionResponse, ModelInfo, StreamChunk, Usage, default_count_tokens
from tsugite.providers.model_registry import get_model_info as _registry_get
from tsugite_acp.client import ACPClientSession
from tsugite_acp.config import workspace_cwd
from tsugite_acp.models import _ALIASES, register_acp_models, resolve_model_alias

register_acp_models()


class ACPProvider:
    """Provider backed by an ACP agent process (default: claude-agent-acp).

    Stateful: the agent subprocess persists across acompletion() calls within a
    session. First acompletion spawns and handshakes; subsequent calls reuse the
    session. Call stop() to release the subprocess.
    """

    cacheable = False

    def __init__(self, name: str = "acp", session_factory: Callable[[], ACPClientSession] | None = None):
        self.name = name
        self._session: ACPClientSession | None = None
        self._session_factory = session_factory

        self._attachments: list = []
        self._skills: list = []
        self._previous_messages: list[dict] = []
        self._resume_session: str | None = None
        self._resume_after_compaction: bool = False

        self._session_id: str | None = None
        self._cache_creation_tokens: int = 0
        self._cache_read_tokens: int = 0
        self._context_window: int | None = None

    def set_context(self, **kwargs: Any) -> None:
        self._attachments = kwargs.get("attachments", [])
        self._skills = kwargs.get("skills", [])
        self._previous_messages = kwargs.get("previous_messages", [])
        self._resume_session = kwargs.get("resume_session")
        self._resume_after_compaction = kwargs.get("resume_after_compaction", False)

    def get_state(self) -> dict | None:
        return {
            "session_id": self._session_id,
            "cache_creation_tokens": self._cache_creation_tokens,
            "cache_read_tokens": self._cache_read_tokens,
            "context_window": self._context_window,
        }

    async def stop(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def acompletion(
        self,
        messages: list[dict],
        model: str,
        stream: bool = False,
        **_kwargs: Any,
    ) -> CompletionResponse | AsyncIterator[StreamChunk]:
        resolve_model_alias(model)  # validate non-empty

        if self._session is None:
            await self._spawn_session()
            blocks = self._build_first_prompt(messages)
        else:
            blocks = self._latest_user_blocks(messages)

        if stream:
            return self._stream_turn(blocks)
        return await self._collect_turn(blocks)

    async def _spawn_session(self) -> None:
        if self._session_factory is not None:
            self._session = self._session_factory()
        else:
            from tsugite_acp.client import spawn_acp_session
            from tsugite_acp.config import resolve_command

            cmd = resolve_command()
            self._session = await spawn_acp_session(
                command=cmd.command,
                args=cmd.args,
                env=cmd.env,
                cwd=cmd.cwd or workspace_cwd(),
            )

        await self._session.start(cwd=workspace_cwd(), resume_session_id=self._resume_session)

    def _build_first_prompt(self, messages: list[dict]) -> list[TextContentBlock]:
        from tsugite.attachments.base import AttachmentContentType, format_attachment_open_tag

        parts: list[str] = []

        include_context = not self._resume_session or self._resume_after_compaction
        if include_context and (self._attachments or self._skills):
            ctx: list[str] = []
            for att in self._attachments:
                if getattr(att, "content_type", None) != AttachmentContentType.TEXT:
                    continue
                ctx.append(format_attachment_open_tag(att))
                ctx.append(att.content)
                ctx.append("</attachment>")
            for skill in self._skills:
                content = getattr(skill, "content", "")
                if len(content) > 4000:
                    content = content[:4000] + "\n... (truncated)"
                ctx.append(f'<skill_content name="{skill.name}">\n{content}\n</skill_content>')
            if ctx:
                parts.append("<context>\n" + "\n".join(ctx) + "\n</context>")

        if self._previous_messages and not self._resume_session:
            history = "\n\n".join(
                f"{m.get('role', 'unknown').capitalize()}: {m.get('content', '')}" for m in self._previous_messages
            )
            parts.append(f"<conversation_history>\n{history}\n</conversation_history>")

        parts.append(_extract_latest_user_text(messages))
        text = "\n".join(p for p in parts if p)
        return [TextContentBlock(type="text", text=text)]

    @staticmethod
    def _latest_user_blocks(messages: list[dict]) -> list[TextContentBlock]:
        return [TextContentBlock(type="text", text=_extract_latest_user_text(messages))]

    async def _collect_turn(self, blocks: list[TextContentBlock]) -> CompletionResponse:
        accumulated = ""
        usage = Usage()
        async for ev in self._session.prompt(blocks):
            if ev.kind == "text":
                accumulated += ev.text
            elif ev.kind == "done":
                usage = self._extract_usage(ev.usage)
                self._session_id = self._session.session_id
        return CompletionResponse(content=accumulated, usage=usage, cost=0.0)

    async def _stream_turn(self, blocks: list[TextContentBlock]) -> AsyncIterator[StreamChunk]:
        usage = Usage()
        async for ev in self._session.prompt(blocks):
            if ev.kind == "text":
                yield StreamChunk(content=ev.text)
            elif ev.kind == "thought":
                yield StreamChunk(reasoning_content=ev.text)
            elif ev.kind == "done":
                usage = self._extract_usage(ev.usage)
                self._session_id = self._session.session_id
        yield StreamChunk(content="", done=True, usage=usage, cost=0.0)

    def _extract_usage(self, raw_usage: dict | None) -> Usage:
        if not raw_usage:
            return Usage()
        prompt_tokens = int(raw_usage.get("input_tokens") or 0)
        completion_tokens = int(raw_usage.get("output_tokens") or 0)
        cache_creation = int(raw_usage.get("cache_creation_input_tokens") or 0)
        cache_read = int(raw_usage.get("cache_read_input_tokens") or 0)
        self._cache_creation_tokens += cache_creation
        self._cache_read_tokens += cache_read
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens + cache_creation + cache_read,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
        )

    def count_tokens(self, text: str, model: str) -> int:
        return default_count_tokens(text, model)

    def get_model_info(self, model: str) -> ModelInfo | None:
        return _registry_get(self.name, resolve_model_alias(model))

    async def list_models(self) -> list[str]:
        return list(_ALIASES.keys())


def _extract_latest_user_text(messages: list[dict]) -> str:
    """Return the text of the most recent user message, flattening list-of-blocks content."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg["content"]
        if isinstance(content, list):
            return "\n".join(
                b if isinstance(b, str) else b.get("text", "") for b in content if isinstance(b, (str, dict))
            )
        return content
    return ""


def create_provider(name: str = "acp", **_kwargs: Any) -> ACPProvider:
    return ACPProvider(name=name)
