"""Tests for daemon memory helpers."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tsugite.daemon.memory import (
    DEFAULT_CONTEXT_LIMIT,
    MIN_RETAINED_TURNS,
    SUMMARIZE_SYSTEM_PROMPT,
    _chunk_messages,
    _combine_summaries,
    _count_tokens,
    _estimate_turn_tokens,
    _llm_complete,
    _message_text,
    _summarize_chunk,
    extract_file_paths_from_turns,
    get_context_limit,
    infer_compaction_model,
    split_turns_for_compaction,
    summarize_session,
)
from tsugite.history.models import Turn
from tsugite.providers.base import CompletionResponse, Usage


@pytest.fixture
def predictable_tokens():
    """Mock _count_tokens to return len//4 for predictable tests."""
    with patch("tsugite.daemon.memory._count_tokens", side_effect=lambda text, model: len(text) // 4):
        yield


def _mock_provider_response(content: str = "Summary") -> CompletionResponse:
    """Create a mock provider CompletionResponse."""
    return CompletionResponse(content=content, usage=Usage(total_tokens=50), cost=0.001)


def _mock_provider(return_value=None, side_effect=None):
    """Create a mock provider with acompletion set up."""
    provider = MagicMock()
    provider.acompletion = AsyncMock(return_value=return_value, side_effect=side_effect)
    provider.count_tokens = MagicMock(return_value=10)
    provider.get_model_info = MagicMock(return_value=None)
    return provider


class TestInferCompactionModel:
    def test_openai_agent(self):
        assert infer_compaction_model("openai:gpt-4o") == "openai:gpt-4o-mini"

    def test_anthropic_agent(self):
        assert infer_compaction_model("anthropic:claude-3-5-sonnet-20241022") == "anthropic:claude-3-haiku-20240307"

    def test_google_agent(self):
        assert infer_compaction_model("google:gemini-2.0-pro") == "google:gemini-2.0-flash-lite"

    def test_ollama_uses_agent_model(self):
        assert infer_compaction_model("ollama:llama3:8b") == "ollama:llama3:8b"

    def test_openrouter_agent(self):
        assert infer_compaction_model("openrouter:openai/gpt-5.2") == "openrouter:openai/gpt-4o-mini"

    def test_unknown_provider_uses_agent_model(self):
        assert infer_compaction_model("bedrock:some-model") == "bedrock:some-model"

    def test_unparseable_falls_back_to_openai(self):
        assert infer_compaction_model("no-colon-model") == "openai:gpt-4o-mini"

    def test_alias_resolved(self):
        """If the model is an alias, it should be resolved first."""
        with patch("tsugite.models.resolve_model_alias", return_value="anthropic:claude-3-5-sonnet-20241022"):
            assert infer_compaction_model("smart") == "anthropic:claude-3-haiku-20240307"


class TestMessageText:
    def test_string_content(self):
        assert _message_text({"role": "user", "content": "hello"}) == "hello"

    def test_list_content_with_text_blocks(self):
        msg = {"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]}
        assert _message_text(msg) == "hello\nworld"

    def test_list_content_with_string_blocks(self):
        msg = {"role": "user", "content": ["hello", "world"]}
        assert _message_text(msg) == "hello\nworld"

    def test_missing_content(self):
        assert _message_text({"role": "user"}) == ""

    def test_none_content(self):
        assert _message_text({"role": "user", "content": None}) == ""

    def test_list_with_non_text_blocks(self):
        msg = {"role": "user", "content": [{"type": "image_url", "url": "http://example.com"}]}
        assert _message_text(msg) == ""


class TestCountTokens:
    def test_uses_provider(self):
        mock_prov = _mock_provider()
        mock_prov.count_tokens.return_value = 42
        with patch("tsugite.models.get_provider_and_model", return_value=("openai", mock_prov, "gpt-4o-mini")):
            result = _count_tokens("hello world", "openai:gpt-4o-mini")
        assert result == 42

    def test_falls_back_on_error(self):
        with patch("tsugite.models.get_provider_and_model", side_effect=Exception("no provider")):
            result = _count_tokens("hello world test", "openai:gpt-4o-mini")
        assert result == len("hello world test") // 4


class TestGetContextLimit:
    def test_returns_provider_value(self):
        from tsugite.providers.base import ModelInfo

        mock_prov = _mock_provider()
        mock_prov.get_model_info.return_value = ModelInfo(max_input_tokens=200_000)
        with patch("tsugite.models.get_provider_and_model", return_value=("openai", mock_prov, "gpt-4o-mini")):
            assert get_context_limit("openai:gpt-4o-mini") == 200_000

    def test_fallback_on_error(self):
        with patch("tsugite.models.get_provider_and_model", side_effect=Exception("unknown")):
            assert get_context_limit("ollama:llama3:8b", fallback=32_000) == 32_000

    def test_fallback_to_default(self):
        with patch("tsugite.models.get_provider_and_model", side_effect=Exception("unknown")):
            assert get_context_limit("ollama:llama3:8b") == DEFAULT_CONTEXT_LIMIT

    def test_fallback_when_no_model_info(self):
        mock_prov = _mock_provider()
        mock_prov.get_model_info.return_value = None
        with patch("tsugite.models.get_provider_and_model", return_value=("openai", mock_prov, "gpt-4o-mini")):
            assert get_context_limit("openai:gpt-4o-mini", fallback=64_000) == 64_000


class TestChunkMessages:
    """Test message chunking with mocked token counting."""

    @pytest.fixture(autouse=True)
    def _mock_tokens(self, predictable_tokens):
        pass

    def test_empty_input(self):
        assert _chunk_messages([], 1000, "openai:gpt-4o-mini") == []

    def test_single_chunk_fits(self):
        messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        chunks = _chunk_messages(messages, 1000, "openai:gpt-4o-mini")
        assert len(chunks) == 1
        assert chunks[0] == messages

    def test_splits_when_exceeds_budget(self):
        messages = [{"role": "user", "content": "x" * 100} for _ in range(10)]
        chunks = _chunk_messages(messages, 60, "openai:gpt-4o-mini")
        assert len(chunks) > 1
        total = sum(len(c) for c in chunks)
        assert total == 10

    def test_truncates_oversized_message(self):
        messages = [{"role": "user", "content": "x" * 10000}]
        chunks = _chunk_messages(messages, 10, "openai:gpt-4o-mini")
        assert len(chunks) == 1
        assert len(chunks[0][0]["content"]) == 40  # 10 * 4

    def test_no_empty_chunks(self):
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
        chunks = _chunk_messages(messages, 100, "openai:gpt-4o-mini")
        for chunk in chunks:
            assert len(chunk) > 0


@pytest.mark.asyncio
async def test_summarize_chunk():
    messages = [{"role": "user", "content": "hello"}]
    mock_prov = _mock_provider(return_value=_mock_provider_response("Chunk summary"))
    with patch("tsugite.models.get_provider_and_model", return_value=("openai", mock_prov, "gpt-4o-mini")):
        result = await _summarize_chunk(messages, "openai:gpt-4o-mini")
    assert result == "Chunk summary"


@pytest.mark.asyncio
async def test_combine_summaries():
    summaries = ["Summary A", "Summary B"]
    mock_prov = _mock_provider(return_value=_mock_provider_response("Combined"))
    with patch("tsugite.models.get_provider_and_model", return_value=("openai", mock_prov, "gpt-4o-mini")):
        result = await _combine_summaries(summaries, "openai:gpt-4o-mini")
    assert result == "Combined"


@pytest.mark.asyncio
async def test_summarize_session_passes_model():
    messages = [{"role": "user", "content": "hello"}]

    mock_prov = _mock_provider(return_value=_mock_provider_response("Summary of conversation"))
    with (
        patch("tsugite.daemon.memory.get_context_limit", return_value=128_000),
        patch("tsugite.models.get_provider_and_model", return_value=("anthropic", mock_prov, "claude-3-haiku-20240307")),
    ):
        result = await summarize_session(messages, model="anthropic:claude-3-haiku-20240307")

    assert result == "Summary of conversation"
    mock_prov.acompletion.assert_called_once()


@pytest.mark.asyncio
async def test_summarize_session_default_model():
    messages = [{"role": "user", "content": "hello"}]

    mock_prov = _mock_provider(return_value=_mock_provider_response())
    with (
        patch("tsugite.daemon.memory.get_context_limit", return_value=128_000),
        patch("tsugite.models.get_provider_and_model", return_value=("openai", mock_prov, "gpt-4o-mini")),
    ):
        await summarize_session(messages)

    mock_prov.acompletion.assert_called_once()


@pytest.mark.asyncio
async def test_summarize_session_empty_history():
    result = await summarize_session([])
    assert result == "No conversation to summarize."


@pytest.mark.asyncio
async def test_summarize_session_single_chunk():
    """Single chunk should make exactly 1 LLM call."""
    messages = [{"role": "user", "content": "hi"}]

    mock_prov = _mock_provider(return_value=_mock_provider_response("Single"))
    with (
        patch("tsugite.daemon.memory.get_context_limit", return_value=128_000),
        patch("tsugite.daemon.memory._count_tokens", return_value=10),
        patch("tsugite.models.get_provider_and_model", return_value=("openai", mock_prov, "gpt-4o-mini")),
    ):
        result = await summarize_session(messages)

    assert result == "Single"
    assert mock_prov.acompletion.call_count == 1


@pytest.mark.asyncio
async def test_summarize_session_multiple_chunks():
    """Multiple chunks should make N+1 LLM calls (N chunks + 1 combine)."""
    messages = [{"role": "user", "content": "x" * 400} for _ in range(3)]

    mock_prov = _mock_provider(return_value=_mock_provider_response())
    with (
        patch("tsugite.daemon.memory.get_context_limit", return_value=160),
        patch("tsugite.daemon.memory._count_tokens", return_value=100),
        patch("tsugite.models.get_provider_and_model", return_value=("openai", mock_prov, "gpt-4o-mini")),
    ):
        result = await summarize_session(messages)

    assert mock_prov.acompletion.call_count == 4  # 3 chunks + 1 combine
    assert result == "Summary"


@pytest.mark.asyncio
async def test_summarize_session_backward_compatible():
    """Old callers without max_context_tokens should still work."""
    messages = [{"role": "user", "content": "hello"}]

    mock_prov = _mock_provider(return_value=_mock_provider_response("OK"))
    with (
        patch("tsugite.daemon.memory.get_context_limit", return_value=128_000),
        patch("tsugite.models.get_provider_and_model", return_value=("openai", mock_prov, "gpt-4o-mini")),
    ):
        result = await summarize_session(messages, model="openai:gpt-4o-mini")

    assert result == "OK"


@pytest.mark.asyncio
async def test_summarize_session_passes_max_context_tokens():
    """max_context_tokens should be forwarded as fallback."""
    messages = [{"role": "user", "content": "hello"}]

    mock_prov = _mock_provider(return_value=_mock_provider_response("OK"))
    with (
        patch("tsugite.daemon.memory.get_context_limit", return_value=32_000) as mock_limit,
        patch("tsugite.daemon.memory._count_tokens", return_value=5),
        patch("tsugite.models.get_provider_and_model", return_value=("ollama", mock_prov, "llama3:8b")),
    ):
        await summarize_session(messages, model="ollama:llama3:8b", max_context_tokens=32_000)

    mock_limit.assert_called_once_with("ollama:llama3:8b", fallback=32_000)


def _make_turn(content: str) -> Turn:
    """Helper to create a Turn with a simple user+assistant exchange."""
    return Turn(
        timestamp=datetime.now(timezone.utc),
        messages=[
            {"role": "user", "content": content},
            {"role": "assistant", "content": f"reply to {content}"},
        ],
    )


class TestEstimateTurnTokens:
    @pytest.fixture(autouse=True)
    def _mock_tokens(self, predictable_tokens):
        pass

    def test_basic(self):
        turn = _make_turn("hello world")
        tokens = _estimate_turn_tokens(turn, "openai:gpt-4o-mini")
        assert tokens > 0

    def test_empty_messages(self):
        turn = Turn(timestamp=datetime.now(timezone.utc), messages=[])
        assert _estimate_turn_tokens(turn, "openai:gpt-4o-mini") == 0


class TestSplitTurnsForCompaction:
    @pytest.fixture(autouse=True)
    def _mock_tokens(self, predictable_tokens):
        pass

    def test_empty_turns(self):
        old, recent = split_turns_for_compaction([], "openai:gpt-4o-mini", 1000)
        assert old == []
        assert recent == []

    def test_single_turn_kept(self):
        turns = [_make_turn("hello")]
        old, recent = split_turns_for_compaction(turns, "openai:gpt-4o-mini", 1000)
        assert old == []
        assert recent == turns

    def test_all_fit_in_budget(self):
        turns = [_make_turn(f"msg {i}") for i in range(3)]
        old, recent = split_turns_for_compaction(turns, "openai:gpt-4o-mini", 100_000)
        assert old == []
        assert recent == turns

    def test_partial_split(self):
        turns = [_make_turn("x" * 200) for _ in range(10)]
        old, recent = split_turns_for_compaction(turns, "openai:gpt-4o-mini", 50)
        assert len(old) > 0
        assert len(recent) >= MIN_RETAINED_TURNS
        assert old + recent == turns

    def test_min_retained_respected(self):
        turns = [_make_turn("x" * 200) for _ in range(5)]
        old, recent = split_turns_for_compaction(turns, "openai:gpt-4o-mini", 1)
        assert len(recent) >= 2
        assert old + recent == turns

    def test_custom_min_retained(self):
        turns = [_make_turn("x" * 200) for _ in range(5)]
        old, recent = split_turns_for_compaction(turns, "openai:gpt-4o-mini", 1, min_retained=3)
        assert len(recent) >= 3
        assert old + recent == turns

    def test_two_turns_always_kept(self):
        turns = [_make_turn("a"), _make_turn("b")]
        old, recent = split_turns_for_compaction(turns, "openai:gpt-4o-mini", 1)
        assert old == []
        assert recent == turns


class TestExtractFilePathsFromTurns:
    def test_extracts_file_paths(self):
        turn = Turn(
            timestamp=datetime.now(timezone.utc),
            messages=[
                {"role": "user", "content": 'read_file(file_path="/home/user/project/main.py")'},
                {"role": "assistant", "content": 'write_file(path="/tmp/output.txt", content="hello")'},
            ],
        )
        paths = extract_file_paths_from_turns([turn])
        assert "/home/user/project/main.py" in paths
        assert "/tmp/output.txt" in paths

    def test_empty_turns(self):
        assert extract_file_paths_from_turns([]) == []

    def test_no_paths(self):
        turn = Turn(
            timestamp=datetime.now(timezone.utc),
            messages=[{"role": "user", "content": "just a regular message"}],
        )
        assert extract_file_paths_from_turns([turn]) == []

    def test_deduplicates(self):
        turn = Turn(
            timestamp=datetime.now(timezone.utc),
            messages=[
                {"role": "user", "content": 'file_path="/src/app.py"'},
                {"role": "assistant", "content": 'file_path="/src/app.py"'},
            ],
        )
        paths = extract_file_paths_from_turns([turn])
        assert paths.count("/src/app.py") == 1


class TestLlmCompleteErrorHandling:
    @pytest.mark.asyncio
    async def test_llm_call_failure_raises_runtime_error(self):
        mock_prov = _mock_provider(side_effect=Exception("connection refused"))
        with patch("tsugite.models.get_provider_and_model", return_value=("openai", mock_prov, "gpt-4o-mini")):
            with pytest.raises(RuntimeError, match=r"LLM call failed.*connection refused"):
                await _llm_complete("system", "user", "openai:gpt-4o-mini")

    @pytest.mark.asyncio
    async def test_empty_response_raises_runtime_error(self):
        mock_prov = _mock_provider(return_value=_mock_provider_response(""))
        with patch("tsugite.models.get_provider_and_model", return_value=("openai", mock_prov, "gpt-4o-mini")):
            with pytest.raises(RuntimeError, match=r"LLM returned empty response"):
                await _llm_complete("system", "user", "openai:gpt-4o-mini")

    @pytest.mark.asyncio
    async def test_none_response_raises_runtime_error(self):
        mock_prov = _mock_provider(return_value=CompletionResponse(content=None))
        with patch("tsugite.models.get_provider_and_model", return_value=("openai", mock_prov, "gpt-4o-mini")):
            with pytest.raises(RuntimeError, match=r"LLM returned empty response"):
                await _llm_complete("system", "user", "openai:gpt-4o-mini")

    @pytest.mark.asyncio
    async def test_successful_call_returns_content(self):
        mock_prov = _mock_provider(return_value=_mock_provider_response("hello"))
        with patch("tsugite.models.get_provider_and_model", return_value=("openai", mock_prov, "gpt-4o-mini")):
            result = await _llm_complete("system", "user", "openai:gpt-4o-mini")
        assert result == "hello"


class TestInferCompactionModelClaudeCode:
    def test_claude_code_sonnet_returns_haiku(self):
        assert infer_compaction_model("claude_code:sonnet") == "claude_code:haiku"

    def test_claude_code_opus_returns_haiku(self):
        assert infer_compaction_model("claude_code:opus") == "claude_code:haiku"

    def test_claude_code_haiku_returns_haiku(self):
        assert infer_compaction_model("claude_code:haiku") == "claude_code:haiku"


class TestLlmCompleteClaudeCodeRouting:
    @pytest.mark.asyncio
    async def test_routes_to_claude_code_complete(self):
        with patch(
            "tsugite.core.claude_code.claude_code_complete",
            new_callable=AsyncMock,
            return_value="claude code result",
        ) as mock_cc:
            result = await _llm_complete("system", "user", "claude_code:sonnet")
        assert result == "claude code result"
        mock_cc.assert_called_once_with("system", "user", "claude-sonnet-4-6")

    @pytest.mark.asyncio
    async def test_claude_code_empty_response_raises(self):
        with patch("tsugite.core.claude_code.claude_code_complete", new_callable=AsyncMock, return_value=""):
            with pytest.raises(RuntimeError, match=r"LLM returned empty response"):
                await _llm_complete("system", "user", "claude_code:haiku")

    @pytest.mark.asyncio
    async def test_claude_code_error_raises_runtime_error(self):
        with patch(
            "tsugite.core.claude_code.claude_code_complete",
            new_callable=AsyncMock,
            side_effect=Exception("cli not found"),
        ):
            with pytest.raises(RuntimeError, match=r"LLM call failed.*cli not found"):
                await _llm_complete("system", "user", "claude_code:sonnet")


class TestSummaryFormatIncludesNewSections:
    def test_files_accessed_section(self):
        assert "## Files Accessed" in SUMMARIZE_SYSTEM_PROMPT

    def test_work_progress_section(self):
        assert "## Work Progress" in SUMMARIZE_SYSTEM_PROMPT
