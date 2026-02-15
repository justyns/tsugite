"""Tests for daemon memory helpers."""

from unittest.mock import AsyncMock, patch

import pytest

from tsugite.daemon.memory import infer_compaction_model, summarize_session


class TestInferCompactionModel:
    def test_openai_agent(self):
        assert infer_compaction_model("openai:gpt-4o") == "openai:gpt-4o-mini"

    def test_anthropic_agent(self):
        assert infer_compaction_model("anthropic:claude-3-5-sonnet-20241022") == "anthropic:claude-3-haiku-20240307"

    def test_google_agent(self):
        assert infer_compaction_model("google:gemini-2.0-pro") == "google:gemini-2.0-flash-lite"

    def test_ollama_uses_agent_model(self):
        assert infer_compaction_model("ollama:llama3:8b") == "ollama:llama3:8b"

    def test_unknown_provider_uses_agent_model(self):
        assert infer_compaction_model("bedrock:some-model") == "bedrock:some-model"

    def test_unparseable_falls_back_to_openai(self):
        assert infer_compaction_model("no-colon-model") == "openai:gpt-4o-mini"

    def test_alias_resolved(self):
        """If the model is an alias, it should be resolved first."""
        with patch("tsugite.models.resolve_model_alias", return_value="anthropic:claude-3-5-sonnet-20241022"):
            assert infer_compaction_model("smart") == "anthropic:claude-3-haiku-20240307"


def _mock_llm_response(content: str = "Summary") -> AsyncMock:
    """Create a mock LLM response with the given content."""
    response = AsyncMock()
    response.choices = [AsyncMock()]
    response.choices[0].message.content = content
    return response


@pytest.mark.asyncio
async def test_summarize_session_passes_model():
    messages = [{"role": "user", "content": "hello"}]

    with patch(
        "litellm.acompletion", new_callable=AsyncMock, return_value=_mock_llm_response("Summary of conversation")
    ) as mock_llm:
        result = await summarize_session(messages, model="anthropic:claude-3-haiku-20240307")

    assert result == "Summary of conversation"
    call_kwargs = mock_llm.call_args[1]
    assert "anthropic/" in call_kwargs["model"]


@pytest.mark.asyncio
async def test_summarize_session_default_model():
    messages = [{"role": "user", "content": "hello"}]

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=_mock_llm_response()) as mock_llm:
        await summarize_session(messages)

    call_kwargs = mock_llm.call_args[1]
    assert "openai/" in call_kwargs["model"]
