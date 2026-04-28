"""Memory management helpers for daemon."""

import asyncio
import logging
import re

DEFAULT_COMPACT_MODEL = "openai:gpt-4o-mini"
DEFAULT_CONTEXT_LIMIT = 128_000
CONTEXT_RESERVE_RATIO = 0.25
RETENTION_BUDGET_RATIO = 0.15
MIN_RETAINED_TURNS = 2

PROVIDER_COMPACT_MODELS = {
    "openai": DEFAULT_COMPACT_MODEL,
    "anthropic": "anthropic:claude-3-haiku-20240307",
    "google": "google:gemini-2.0-flash-lite",
    "openrouter": "openrouter:openai/gpt-4o-mini",
    "claude_code": "claude_code:haiku",
    "ollama": None,  # use agent model as-is
}

logger = logging.getLogger(__name__)

_FILE_PATH_PATTERN = re.compile(r'(?:file_path|path|filename)["\s:=]+["\'`]?(/[^\s"\'`\],}]+)', re.IGNORECASE)

_SUMMARY_FORMAT = """\
## Current Task
What is actively being worked on or discussed right now.

## Key Decisions
Important decisions made during the conversation.

## Facts & Preferences
Facts learned about the user, project, or environment. User preferences and conventions.

## Files Accessed
Files that were read, written, or modified during the conversation. Include full paths.

## Work Progress
What was completed, what was attempted and failed, what is partially done.

## Open Questions
Unresolved questions or ambiguities that still need answers.

## Action Items
Concrete next steps or pending tasks."""

_ATTACHMENT_DIRECTIVE = (
    "Do not include or repeat content from auto-attached workspace files "
    "(e.g., AGENTS.md, MEMORY.md, IDENTITY.md, USER.md, CLAUDE.md). "
    "The agent re-loads those automatically; summarizing them wastes tokens "
    "and produces a stale snapshot if those files change."
)

SUMMARIZE_SYSTEM_PROMPT = (
    "Summarize this conversation using the structured format below.\n"
    f"{_ATTACHMENT_DIRECTIVE}\n"
    "Keep the total summary under 800 words. Omit any section that has no content.\n\n"
    f"{_SUMMARY_FORMAT}"
)

COMBINE_SYSTEM_PROMPT = (
    "You are given multiple summaries of consecutive conversation chunks.\n"
    "Combine them into a single coherent summary using the structured format below.\n"
    f"{_ATTACHMENT_DIRECTIVE}\n"
    "Keep the total summary under 800 words. Omit any section that has no content.\n\n"
    f"{_SUMMARY_FORMAT}"
)


def infer_compaction_model(agent_model: str) -> str:
    """Pick a cheap model from the same provider as the agent."""
    from tsugite.models import parse_model_string, resolve_model_alias

    resolved = resolve_model_alias(agent_model)
    try:
        provider, _, _ = parse_model_string(resolved)
    except ValueError:
        return DEFAULT_COMPACT_MODEL

    if provider not in PROVIDER_COMPACT_MODELS:
        return resolved

    compact = PROVIDER_COMPACT_MODELS[provider]
    return compact if compact is not None else resolved


def get_context_limit(model: str, fallback: int | None = None) -> int:
    """Get context limit for a model via the provider system.

    Priority: provider model_info -> fallback -> DEFAULT_CONTEXT_LIMIT.
    """
    from tsugite.models import get_provider_and_model

    try:
        provider_name, provider, model_id = get_provider_and_model(model)
        info = provider.get_model_info(model_id)
        if info and info.max_input_tokens:
            return info.max_input_tokens
    except Exception:
        pass

    return fallback if fallback is not None else DEFAULT_CONTEXT_LIMIT


def _count_tokens(text: str, model: str) -> int:
    """Count tokens using the provider, falling back to len//4."""
    from tsugite.models import get_provider_and_model

    try:
        _, provider, model_id = get_provider_and_model(model)
        return provider.count_tokens(text, model_id)
    except Exception:
        return len(text) // 4


def _message_text(msg: dict) -> str:
    """Extract text content from a message dict."""
    content = msg.get("content")
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block if isinstance(block, str) else block["text"]
            for block in content
            if isinstance(block, str) or (isinstance(block, dict) and "text" in block)
        ]
        return "\n".join(parts)
    return str(content)


def _chunk_messages(messages: list[dict], max_chunk_tokens: int, model: str) -> list[list[dict]]:
    """Split messages into chunks that fit within token budget.

    Keeps user/assistant pairs together when possible.
    """
    if not messages:
        return []

    chunks: list[list[dict]] = []
    current_chunk: list[dict] = []
    current_tokens = 0

    for msg in messages:
        msg_text = _message_text(msg)
        text = f"{msg.get('role', 'user').upper()}: {msg_text}"
        msg_tokens = _count_tokens(text, model)

        if msg_tokens > max_chunk_tokens:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            chars_budget = max_chunk_tokens * 4
            chunks.append([{**msg, "content": msg_text[:chars_budget]}])
            continue

        # Would adding this message exceed the budget?
        if current_tokens + msg_tokens > max_chunk_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.append(msg)
        current_tokens += msg_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


_MISSING = object()


async def _llm_complete(system_prompt: str, user_content: str, model: str) -> str:
    """Send a system+user message pair to the LLM and return the response text."""
    from tsugite.models import get_provider_and_model

    try:
        _, provider, model_id = get_provider_and_model(model)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        # Shield the provider's per-instance _context_window from this internal
        # call. Summarization (and title generation) typically use a smaller
        # model than the agent; without this guard, the smaller model's
        # context_window leaks into shared provider state and corrupts the next
        # agent turn's reported context limit.
        saved_context_window = getattr(provider, "_context_window", _MISSING)
        try:
            response = await provider.acompletion(messages=messages, model=model_id)
        finally:
            if saved_context_window is not _MISSING:
                provider._context_window = saved_context_window
        content = response.content
    except Exception as e:
        raise RuntimeError(f"LLM call failed ({model}): {e}") from e

    if not content:
        raise RuntimeError(f"LLM returned empty response ({model})")
    return content


async def _summarize_chunk(messages: list[dict], model: str) -> str:
    """Summarize a single chunk of conversation messages."""
    convo_text = "\n\n".join(f"{msg['role'].upper()}: {_message_text(msg)}" for msg in messages)
    return await _llm_complete(SUMMARIZE_SYSTEM_PROMPT, convo_text, model)


async def _combine_summaries(summaries: list[str], model: str) -> str:
    """Combine multiple chunk summaries into one."""
    numbered = "\n\n".join(f"--- Chunk {i + 1} ---\n{s}" for i, s in enumerate(summaries))
    return await _llm_complete(COMBINE_SYSTEM_PROMPT, numbered, model)


async def summarize_session(
    conversation_history: list[dict],
    model: str | None = None,
    max_context_tokens: int | None = None,
    progress_callback=None,
) -> str:
    """Summarize conversation history using LLM with chunked map-reduce.

    Args:
        conversation_history: List of {role, content} dicts
        model: Model to use for summarization (Tsugite format: provider:model).
        max_context_tokens: Override context limit (for Ollama/custom models).
        progress_callback: Optional fire-and-forget sync fn called with phase
            payloads ({"phase": "chunking"}, {"phase": "summarizing",
            "chunk_index": i, "chunk_total": n}, {"phase": "combining"}).
    """

    def _emit(payload: dict) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(payload)
        except Exception:
            logger.debug("compaction progress_callback raised", exc_info=True)

    if not conversation_history:
        return "No conversation to summarize."

    if model is None:
        model = DEFAULT_COMPACT_MODEL

    context_limit = get_context_limit(model, fallback=max_context_tokens)
    usable_tokens = int(context_limit * (1 - CONTEXT_RESERVE_RATIO))

    _emit({"phase": "chunking"})
    chunks = _chunk_messages(conversation_history, usable_tokens, model)
    if not chunks:
        return "No conversation to summarize."

    if len(chunks) == 1:
        _emit({"phase": "summarizing", "chunk_index": 1, "chunk_total": 1})
        return await _summarize_chunk(chunks[0], model)

    logger.info("Summarizing %d chunks (context limit: %d, usable: %d)", len(chunks), context_limit, usable_tokens)
    completed = 0

    async def _summarize_with_progress(chunk: list[dict]) -> str:
        nonlocal completed
        result = await _summarize_chunk(chunk, model)
        completed += 1
        _emit({"phase": "summarizing", "chunk_index": completed, "chunk_total": len(chunks)})
        return result

    chunk_summaries = await asyncio.gather(*[_summarize_with_progress(chunk) for chunk in chunks])
    _emit({"phase": "combining"})
    return await _combine_summaries(chunk_summaries, model)


TITLE_SYSTEM_PROMPT = (
    "Generate a short title (3-6 words) for this conversation. "
    "Return only the title, nothing else. No quotes, no punctuation at the end."
)

TITLE_TIMEOUT = 30
SHORT_TITLE_THRESHOLD = 60


async def generate_session_title(messages: list[dict], model: str) -> str:
    """Generate a short title from conversation messages using a cheap model."""
    text_parts = []
    for msg in messages:
        content = _message_text(msg)
        if content:
            text_parts.append(f"{msg.get('role', 'user').upper()}: {content[:500]}")
    if not text_parts:
        return ""
    convo_text = "\n\n".join(text_parts)
    title = await asyncio.wait_for(
        _llm_complete(TITLE_SYSTEM_PROMPT, convo_text, model),
        timeout=TITLE_TIMEOUT,
    )
    return title.strip().strip("\"'")[:80]


async def compute_session_title(
    user_content: str,
    assistant_content: str,
    agent_model: str,
) -> str:
    """Compute a title for a session. Returns empty string if no title could be generated."""
    if len(user_content) <= SHORT_TITLE_THRESHOLD:
        return user_content
    model = infer_compaction_model(agent_model)
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    return await generate_session_title(messages, model)


def extract_file_paths_from_events(events: list) -> list[str]:
    """Extract file paths from any text payloads in the event stream."""
    paths: set[str] = set()
    for event in events:
        for value in event.data.values():
            if isinstance(value, str):
                paths.update(_FILE_PATH_PATTERN.findall(value))
    return sorted(paths)


def split_events_for_compaction(
    events: list,
    model: str,
    retention_budget_tokens: int,
    min_retained_turns: int = MIN_RETAINED_TURNS,
) -> tuple[list, list]:
    """Split events into (old, recent) groups along user_input boundaries.

    Walks backward by `user_input` boundary, keeping recent turns whose total
    estimated tokens fit within the budget. Always keeps at least
    `min_retained_turns` worth of events. If all turns fit, returns ([], events).
    """
    if not events:
        return [], []

    boundaries = [i for i, e in enumerate(events) if e.type == "user_input"]
    if len(boundaries) <= min_retained_turns:
        return [], list(events)

    cutoff = None
    used = 0
    kept_turns = 0
    for i in range(len(boundaries) - 1, -1, -1):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(events)
        cost = sum(_count_tokens(_event_text(e), model) for e in events[start:end])
        if kept_turns >= min_retained_turns and used + cost > retention_budget_tokens:
            cutoff = boundaries[i + 1] if i + 1 < len(boundaries) else None
            break
        used += cost
        kept_turns += 1
        cutoff = start

    if cutoff is None or cutoff == 0:
        return [], list(events)
    return events[:cutoff], events[cutoff:]


def _event_text(event) -> str:
    """Cheap text approximation of an event for token counting."""
    if event.type == "user_input":
        return event.data.get("text", "")
    if event.type == "model_response":
        return event.data.get("raw_content", "")
    if event.type == "code_execution":
        return f"{event.data.get('code', '')}\n{event.data.get('output', '') or ''}"
    return ""


_ATTACHMENT_BLOCK = re.compile(
    r"<attachment\b([^>]*)>.*?</attachment>",
    re.DOTALL,
)
_NAMED_ATTACHMENT_BLOCK = re.compile(
    r"<Attachment:\s*([^>]+?)>(.*?)</Attachment:\s*\1\s*>",
    re.DOTALL,
)
_CONTEXT_BLOCK = re.compile(r"<context>.*?</context>", re.DOTALL)
_SKILL_BLOCK = re.compile(
    r"<skill_content\b([^>]*)>.*?</skill_content>",
    re.DOTALL,
)
_EXECUTION_RESULT = re.compile(
    r"(<tsugite_execution_result\b[^>]*>)(.*?)(</tsugite_execution_result>)",
    re.DOTALL,
)
_OUTPUT_BODY = re.compile(r"(<output>)(.*?)(</output>)", re.DOTALL)
_TRUNCATION_MARKER = "...[truncated]..."


def _elide_block(match: "re.Match[str]", tag: str) -> str:
    attrs = match.group(1).strip()
    if attrs:
        return f'<{tag} {attrs} elided="true"/>'
    return f'<{tag} elided="true"/>'


def _truncate_output_body(body: str, head_chars: int, tail_chars: int) -> str:
    if _TRUNCATION_MARKER in body:
        return body
    if len(body) <= head_chars + tail_chars + len(_TRUNCATION_MARKER):
        return body
    return f"{body[:head_chars]}\n{_TRUNCATION_MARKER}\n{body[-tail_chars:]}"


def _elide_attachment_aware_outputs(content: str, basenames: set[str]) -> str:
    """Elide `<output>...</output>` bodies inside `<tsugite_execution_result>`
    blocks whose code or output references a known attachment file basename.

    These blocks are typically `read_note('MEMORY.md')` style tool calls that
    inline auto-attached file contents into the event stream, where the
    size-based truncation may not catch them (small files, big budget). The
    agent re-loads attachments every turn anyway.
    """
    valid = [n for n in basenames if n]
    if not valid:
        return content

    name_pattern = re.compile("|".join(re.escape(n) for n in valid))

    def _maybe_elide(match: "re.Match[str]") -> str:
        opening, inner, closing = match.group(1), match.group(2), match.group(3)
        hit = name_pattern.search(match.group(0))
        if not hit:
            return match.group(0)
        shrunk_inner = _OUTPUT_BODY.sub(
            f'<output elided="attachment_file: {hit.group(0)}"/>',
            inner,
        )
        return f"{opening}{shrunk_inner}{closing}"

    return _EXECUTION_RESULT.sub(_maybe_elide, content)


def sanitize_for_summary(
    messages: list[dict],
    per_message_token_budget: int = 1500,
    model: str = DEFAULT_COMPACT_MODEL,
    attachment_basenames: set[str] = frozenset(),
) -> list[dict]:
    """Strip workspace-scaffolding blocks and truncate oversized tool outputs.

    Applied just before `summarize_session` so the summary describes the
    conversation, not file dumps or context-turn scaffolding that may have
    been replayed via `code_execution` outputs (e.g. read_note('AGENTS.md')).

    - Inline `<attachment>...</attachment>`, `<Attachment: name>...</Attachment: name>`,
      `<context>...</context>`, and `<skill_content ...>...</skill_content>`
      blocks are replaced with one-line elision tags that preserve their
      identifying attributes.
    - When `attachment_basenames` is provided, `<tsugite_execution_result>`
      blocks whose code or output references a known basename have their
      `<output>` body elided regardless of size (small reads of MEMORY.md
      otherwise slip past the size budget).
    - For oversized `<tsugite_execution_result>` blocks (token count exceeding
      `per_message_token_budget`), the `<output>...</output>` body is truncated
      to head + marker + tail, preserving the wrapping tags so structure is
      kept intact.
    - String content only; non-string content (e.g. multimodal blocks) passes
      through unchanged.
    """
    head_chars = max(per_message_token_budget * 4 // 4, 400)
    tail_chars = max(per_message_token_budget * 4 // 8, 200)

    out: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, str):
            out.append(msg)
            continue

        new_content = _ATTACHMENT_BLOCK.sub(lambda m: _elide_block(m, "attachment"), content)
        new_content = _NAMED_ATTACHMENT_BLOCK.sub(
            lambda m: f'<attachment name="{m.group(1).strip()}" elided="true"/>', new_content
        )
        new_content = _CONTEXT_BLOCK.sub('<context elided="true"/>', new_content)
        new_content = _SKILL_BLOCK.sub(lambda m: _elide_block(m, "skill_content"), new_content)
        new_content = _elide_attachment_aware_outputs(new_content, attachment_basenames)

        if _count_tokens(new_content, model) > per_message_token_budget:

            def _shrink(match: "re.Match[str]") -> str:
                opening, inner, closing = match.group(1), match.group(2), match.group(3)
                shrunk_inner = _OUTPUT_BODY.sub(
                    lambda om: (
                        f"{om.group(1)}{_truncate_output_body(om.group(2), head_chars, tail_chars)}{om.group(3)}"
                    ),
                    inner,
                )
                return f"{opening}{shrunk_inner}{closing}"

            new_content = _EXECUTION_RESULT.sub(_shrink, new_content)

        if new_content == content:
            out.append(msg)
        else:
            out.append({**msg, "content": new_content})
    return out
