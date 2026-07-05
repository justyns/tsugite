"""Tests for scripts/update_model_registry.py (models.dev-sourced codegen).

The script is codegen, not runtime code, but its capability mapping decides
what lands in the provider registries - these tests pin the models.dev field
mapping so a refresh can't silently regress capabilities.
"""

import importlib.util
from pathlib import Path

from tsugite.providers.base import ModelInfo

_SCRIPT = Path(__file__).parent.parent / "scripts" / "update_model_registry.py"
_spec = importlib.util.spec_from_file_location("update_model_registry", _SCRIPT)
umr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(umr)


def _info(key: str, entry: dict, provider: str) -> ModelInfo:
    """Evaluate the generated ModelInfo(...) constructor string."""
    return eval(umr.entry_to_model_info(key, entry, provider), {"ModelInfo": ModelInfo})  # noqa: S307


def _entry(**overrides) -> dict:
    base = {
        "id": "test-model",
        "reasoning": False,
        "modalities": {"input": ["text"], "output": ["text"]},
        "limit": {"context": 200_000, "output": 64_000},
        "cost": {"input": 5, "output": 25},
    }
    base.update(overrides)
    return base


class TestSource:
    def test_source_is_models_dev(self):
        assert "models.dev" in umr.MODELS_DEV_URL

    def test_capability_regexes_are_gone(self):
        """AC: capability inference comes from structured fields, not name regexes."""
        for name in ("REASONING_PATTERNS", "EFFORT_LEVELS_PATTERNS", "ANTHROPIC_THINKING_PATTERNS"):
            assert not hasattr(umr, name), f"{name} must be removed"


class TestFieldMapping:
    def test_limits_and_costs_map_per_million(self):
        info = _info("m", _entry(), "openai")
        assert info.max_input_tokens == 200_000
        assert info.max_output_tokens == 64_000
        assert info.input_cost_per_million == 5
        assert info.output_cost_per_million == 25

    def test_vision_and_audio_from_input_modalities(self):
        info = _info("m", _entry(modalities={"input": ["text", "image", "audio"], "output": ["text"]}), "openai")
        assert info.supports_vision is True
        assert info.supports_audio is True

    def test_text_only_input_has_no_vision(self):
        info = _info("m", _entry(), "openai")
        assert info.supports_vision is False
        assert info.supports_audio is False

    def test_reasoning_flag_from_structured_field(self):
        info = _info("m", _entry(reasoning=True), "openai")
        assert info.supports_reasoning is True


class TestEffortLevels:
    def test_native_effort_values_used_verbatim(self):
        """Opus 4.8-shaped entry: effort-only reasoning options carry through,
        including xhigh (which the old name-regexes could never produce)."""
        entry = _entry(
            reasoning=True,
            reasoning_options=[{"type": "effort", "values": ["low", "medium", "high", "xhigh", "max"]}],
        )
        info = _info("claude-opus-4-8", entry, "anthropic")
        assert info.supported_effort_levels == ["low", "medium", "high", "xhigh", "max"]

    def test_anthropic_budget_tokens_maps_to_budget_vocab(self):
        """Haiku 4.5-shaped entry: budget_tokens-only models support tsugite's
        effort strings via the provider's _EFFORT_TO_BUDGET translation."""
        entry = _entry(reasoning=True, reasoning_options=[{"type": "budget_tokens", "min": 1024}])
        info = _info("claude-haiku-4-5", entry, "anthropic")
        assert info.supported_effort_levels == ["low", "medium", "high", "max"]

    def test_anthropic_budget_tokens_wins_over_native_effort(self):
        """Opus 4.5-shaped entry (both options): the budget vocabulary includes
        max, which the provider can always emulate via budget_tokens."""
        entry = _entry(
            reasoning=True,
            reasoning_options=[
                {"type": "effort", "values": ["low", "medium", "high"]},
                {"type": "budget_tokens", "min": 1024},
            ],
        )
        info = _info("claude-opus-4-5", entry, "anthropic")
        assert info.supported_effort_levels == ["low", "medium", "high", "max"]

    def test_openai_effort_values_verbatim(self):
        entry = _entry(
            reasoning=True,
            reasoning_options=[{"type": "effort", "values": ["none", "low", "medium", "high", "xhigh"]}],
        )
        info = _info("gpt-5.5", entry, "openai")
        assert info.supported_effort_levels == ["none", "low", "medium", "high", "xhigh"]

    def test_no_reasoning_options_means_no_effort_levels(self):
        info = _info("gpt-4o", _entry(), "openai")
        assert info.supported_effort_levels is None

    def test_toggle_only_reasoning_option_yields_no_effort_levels(self):
        entry = _entry(reasoning=True, reasoning_options=[{"type": "toggle"}])
        info = _info("m", entry, "openai")
        assert info.supported_effort_levels is None


class TestThinkingStyle:
    """thinking_style tells the anthropic provider which API surface drives
    reasoning: budget_tokens extended thinking vs the adaptive/effort surface
    (which rejects budget_tokens and sampling params with 400)."""

    def test_budget_tokens_option_maps_to_budget_style(self):
        entry = _entry(reasoning=True, reasoning_options=[{"type": "budget_tokens", "min": 1024}])
        info = _info("claude-haiku-4-5", entry, "anthropic")
        assert info.thinking_style == "budget_tokens"

    def test_budget_option_wins_when_both_present(self):
        entry = _entry(
            reasoning=True,
            reasoning_options=[
                {"type": "effort", "values": ["low", "medium", "high"]},
                {"type": "budget_tokens", "min": 1024},
            ],
        )
        info = _info("claude-opus-4-5", entry, "anthropic")
        assert info.thinking_style == "budget_tokens"

    def test_effort_only_maps_to_adaptive_style(self):
        entry = _entry(
            reasoning=True,
            reasoning_options=[{"type": "effort", "values": ["low", "medium", "high", "xhigh", "max"]}],
        )
        info = _info("claude-opus-4-8", entry, "anthropic")
        assert info.thinking_style == "adaptive"

    def test_openai_models_get_no_thinking_style(self):
        entry = _entry(
            reasoning=True,
            reasoning_options=[{"type": "effort", "values": ["low", "medium", "high"]}],
        )
        info = _info("o3", entry, "openai")
        assert info.thinking_style is None

    def test_non_reasoning_model_gets_no_thinking_style(self):
        info = _info("m", _entry(), "anthropic")
        assert info.thinking_style is None


class TestShouldSkip:
    def test_skips_non_text_output_models(self):
        entry = _entry(modalities={"input": ["text"], "output": ["image"]})
        assert umr.should_skip("gpt-image-2", entry) is True

    def test_skips_embedding_prefix(self):
        assert umr.should_skip("text-embedding-3-large", _entry()) is True

    def test_keeps_chat_models(self):
        assert umr.should_skip("gpt-4o", _entry()) is False
