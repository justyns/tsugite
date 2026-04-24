"""Tests for reasoning model support and parameter filtering."""

import pytest

from tsugite.models import (
    UnsupportedEffortError,
    get_model_id,
    get_model_kwargs,
    is_reasoning_model_without_stop_support,
    resolve_reasoning_effort,
)


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

    def test_o1_mini_removes_incompatible_params(self):
        """Reasoning models drop temperature/top_p/etc. reasoning_effort passes through
        (runner-level validation handles model-specific gating via ModelInfo)."""
        kwargs = get_model_kwargs("openai:o1-mini", reasoning_effort="high", temperature=0.7, top_p=0.9)

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


class TestOpenAIReasoningEffortLevels:
    """ModelInfo should declare supported_effort_levels for reasoning models."""

    def test_reasoning_models_have_effort_levels(self):
        from tsugite.providers.openai_compat import _OPENAI_MODELS

        for key in ("openai/o1", "openai/o3", "openai/o3-mini", "openai/o4-mini"):
            info = _OPENAI_MODELS.get(key)
            assert info is not None, f"missing: {key}"
            assert info.supported_effort_levels == ["low", "medium", "high"], key

    def test_non_reasoning_models_have_no_effort_levels(self):
        from tsugite.providers.openai_compat import _OPENAI_MODELS

        info = _OPENAI_MODELS.get("openai/gpt-4o")
        assert info is not None
        assert info.supported_effort_levels is None


class TestResolveReasoningEffort:
    """resolve_reasoning_effort() validates against ModelInfo.supported_effort_levels."""

    def _ensure_registered(self):
        # Provider constructors register their models on init.
        from tsugite.providers.claude_code import ClaudeCodeProvider
        from tsugite.providers.model_registry import register_models
        from tsugite.providers.openai_compat import _OPENAI_MODELS

        ClaudeCodeProvider()
        register_models(_OPENAI_MODELS)

    def test_none_effort_returns_none(self):
        self._ensure_registered()
        assert resolve_reasoning_effort("claude_code:opus", None) is None

    def test_supported_value_returned_verbatim(self):
        self._ensure_registered()
        assert resolve_reasoning_effort("claude_code:opus", "xhigh") == "xhigh"

    def test_invalid_value_raises(self):
        self._ensure_registered()
        with pytest.raises(UnsupportedEffortError) as exc:
            resolve_reasoning_effort("claude_code:opus", "ultra")
        assert "xhigh" in str(exc.value)

    def test_model_without_effort_levels_drops_with_warning(self, caplog):
        """gpt-4o doesn't support reasoning_effort; resolve drops it and warns."""
        self._ensure_registered()
        import logging

        with caplog.at_level(logging.WARNING):
            result = resolve_reasoning_effort("openai:gpt-4o", "high")
        assert result is None
        assert any("reasoning_effort" in r.message for r in caplog.records)
