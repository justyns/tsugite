"""Tests for reasoning model support and parameter filtering."""

from tsugite.models import get_model_id, get_model_kwargs, is_reasoning_model_without_stop_support


class TestReasoningModelDetection:
    """Test reasoning model detection."""

    def test_o1_models_detected(self):
        assert is_reasoning_model_without_stop_support("openai:o1")
        assert is_reasoning_model_without_stop_support("openai:o1-mini")
        assert is_reasoning_model_without_stop_support("openai:o1-preview")
        assert is_reasoning_model_without_stop_support("openai:o1-2024-12-17")

    def test_o3_models_detected(self):
        assert is_reasoning_model_without_stop_support("openai:o3")
        assert is_reasoning_model_without_stop_support("openai:o3-mini")
        assert is_reasoning_model_without_stop_support("openai:o3-2025-01-31")

    def test_non_reasoning_models_not_detected(self):
        assert not is_reasoning_model_without_stop_support("openai:gpt-4")
        assert not is_reasoning_model_without_stop_support("openai:gpt-4o")
        assert not is_reasoning_model_without_stop_support("openai:gpt-4-turbo")
        assert not is_reasoning_model_without_stop_support("anthropic:claude-3-7-sonnet")
        assert not is_reasoning_model_without_stop_support("ollama:qwen2.5-coder:7b")

    def test_invalid_model_strings(self):
        assert not is_reasoning_model_without_stop_support("invalid")
        assert not is_reasoning_model_without_stop_support("")


class TestReasoningModelParameterFiltering:
    """Test parameter filtering for reasoning models."""

    def test_o1_mini_removes_all_unsupported_params(self):
        kwargs = get_model_kwargs("openai:o1-mini", reasoning_effort="high", temperature=0.7, top_p=0.9)

        assert "reasoning_effort" not in kwargs
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs

    def test_o1_keeps_reasoning_effort(self):
        kwargs = get_model_kwargs("openai:o1", reasoning_effort="high", temperature=0.7)

        assert kwargs.get("reasoning_effort") == "high"
        assert "temperature" not in kwargs

    def test_o3_keeps_reasoning_effort(self):
        kwargs = get_model_kwargs("openai:o3-mini", reasoning_effort="medium")

        assert kwargs.get("reasoning_effort") == "medium"
        assert "temperature" not in kwargs

    def test_gpt4_keeps_all_params(self):
        kwargs = get_model_kwargs("openai:gpt-4", temperature=0.7, top_p=0.9)

        assert kwargs.get("temperature") == 0.7
        assert kwargs.get("top_p") == 0.9

    def test_model_id_set_correctly(self):
        model_id = get_model_id("openai:o1")
        assert model_id == "o1"
