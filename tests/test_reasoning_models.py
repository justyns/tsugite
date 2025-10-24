"""Tests for reasoning model support and parameter filtering."""

from tsugite.models import get_model_params, is_reasoning_model_without_stop_support


class TestReasoningModelDetection:
    """Test reasoning model detection."""

    def test_o1_models_detected(self):
        """Test that o1 models are correctly detected."""
        assert is_reasoning_model_without_stop_support("openai:o1")
        assert is_reasoning_model_without_stop_support("openai:o1-mini")
        assert is_reasoning_model_without_stop_support("openai:o1-preview")
        assert is_reasoning_model_without_stop_support("openai:o1-2024-12-17")

    def test_o3_models_detected(self):
        """Test that o3 models are correctly detected."""
        assert is_reasoning_model_without_stop_support("openai:o3")
        assert is_reasoning_model_without_stop_support("openai:o3-mini")
        assert is_reasoning_model_without_stop_support("openai:o3-2025-01-31")

    def test_non_reasoning_models_not_detected(self):
        """Test that non-reasoning models are not detected."""
        assert not is_reasoning_model_without_stop_support("openai:gpt-4")
        assert not is_reasoning_model_without_stop_support("openai:gpt-4o")
        assert not is_reasoning_model_without_stop_support("openai:gpt-4-turbo")
        assert not is_reasoning_model_without_stop_support("anthropic:claude-3-7-sonnet")
        assert not is_reasoning_model_without_stop_support("ollama:qwen2.5-coder:7b")

    def test_invalid_model_strings(self):
        """Test that invalid model strings don't crash."""
        assert not is_reasoning_model_without_stop_support("invalid")
        assert not is_reasoning_model_without_stop_support("")


class TestReasoningModelParameterFiltering:
    """Test parameter filtering for reasoning models."""

    def test_o1_mini_removes_all_unsupported_params(self):
        """Test that o1-mini removes all unsupported parameters including reasoning_effort."""
        params = get_model_params("openai:o1-mini", reasoning_effort="high", temperature=0.7, top_p=0.9)

        assert "reasoning_effort" not in params
        assert "temperature" not in params
        assert "top_p" not in params
        assert "stop" not in params
        assert "presence_penalty" not in params
        assert "frequency_penalty" not in params

    def test_o1_keeps_reasoning_effort(self):
        """Test that o1 (not o1-mini) keeps reasoning_effort but removes other params."""
        params = get_model_params("openai:o1", reasoning_effort="high", temperature=0.7)

        assert params.get("reasoning_effort") == "high"
        assert "temperature" not in params
        assert "stop" not in params

    def test_o3_keeps_reasoning_effort(self):
        """Test that o3 models keep reasoning_effort."""
        params = get_model_params("openai:o3-mini", reasoning_effort="medium")

        assert params.get("reasoning_effort") == "medium"
        assert "temperature" not in params

    def test_gpt4_keeps_all_params(self):
        """Test that GPT-4 models keep all parameters."""
        params = get_model_params("openai:gpt-4", temperature=0.7, top_p=0.9)

        assert params.get("temperature") == 0.7
        assert params.get("top_p") == 0.9

    def test_model_id_set_correctly(self):
        """Test that model ID is set correctly in params."""
        params = get_model_params("openai:o1")
        assert params["model"] == "openai/o1"
