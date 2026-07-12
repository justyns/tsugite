"""Tests for model registry — verify no duplicate model keys."""

from tsugite.providers.anthropic import _ANTHROPIC_MODELS
from tsugite.providers.base import ModelInfo
from tsugite.providers.openai_compat import _OPENAI_MODELS


def test_no_duplicate_keys_within_openai():
    keys = list(_OPENAI_MODELS.keys())
    assert len(keys) == len(set(keys)), f"Duplicate keys in _OPENAI_MODELS: {[k for k in keys if keys.count(k) > 1]}"


def test_no_duplicate_keys_within_anthropic():
    keys = list(_ANTHROPIC_MODELS.keys())
    assert len(keys) == len(set(keys)), f"Duplicate keys in _ANTHROPIC_MODELS: {[k for k in keys if keys.count(k) > 1]}"


def test_no_overlap_between_providers():
    overlap = set(_OPENAI_MODELS.keys()) & set(_ANTHROPIC_MODELS.keys())
    assert not overlap, f"Models appear in multiple provider dicts: {overlap}"


def test_all_keys_have_correct_provider_prefix():
    for key in _OPENAI_MODELS:
        assert key.startswith("openai/"), f"OpenAI model key missing prefix: {key}"
    for key in _ANTHROPIC_MODELS:
        assert key.startswith("anthropic/"), f"Anthropic model key missing prefix: {key}"


def test_model_info_has_supported_effort_levels_default_none():
    info = ModelInfo()
    assert info.supported_effort_levels is None


def test_model_info_accepts_supported_effort_levels():
    info = ModelInfo(supported_effort_levels=["low", "medium", "high"])
    assert info.supported_effort_levels == ["low", "medium", "high"]


def test_gpt_5_6_family_registered():
    """gpt-5.6 ships as three named tiers (sol/terra/luna) plus a bare `gpt-5.6`
    alias that routes to Sol. Specs per models.dev / OpenAI's GA announcement:
    1.05M context, 128K output, effort ladder up to `max` (`ultra` is not an
    API effort value in the catalog and is deliberately absent)."""
    pricing = {
        "openai/gpt-5.6": (5.0, 30.0),
        "openai/gpt-5.6-sol": (5.0, 30.0),
        "openai/gpt-5.6-terra": (2.5, 15.0),
        "openai/gpt-5.6-luna": (1.0, 6.0),
    }
    for key, (input_cost, output_cost) in pricing.items():
        info = _OPENAI_MODELS.get(key)
        assert info is not None, f"{key} missing from registry"
        assert info.max_input_tokens == 1_050_000, key
        assert info.max_output_tokens == 128_000, key
        assert info.input_cost_per_million == input_cost, key
        assert info.output_cost_per_million == output_cost, key
        assert info.supports_vision is True, key
        assert info.supports_reasoning is True, key
        assert info.supported_effort_levels == ["none", "low", "medium", "high", "xhigh", "max"], key


def test_gpt_5_6_effort_resolution_end_to_end():
    """`max` effort validates for gpt-5.6 through the shared resolution path."""
    from tsugite.models import resolve_reasoning_effort

    assert resolve_reasoning_effort("openai:gpt-5.6-sol", "max") == "max"


def test_prefix_match_rejects_variants_but_allows_dated_versions():
    """Prefix lookup must not let an unlisted variant inherit a sibling's pricing:
    `o1-mini` is a distinct model from `o1` (~13x cheaper), so it must resolve to None,
    while a dated version (`o1-2024-12-17`) should still prefix-match the base entry."""
    from tsugite.providers.model_registry import _REGISTRY, get_model_info, register_model

    register_model("testreg", "o1", ModelInfo(input_cost_per_million=15.0, output_cost_per_million=60.0))
    try:
        assert get_model_info("testreg", "o1") is not None  # exact
        assert get_model_info("testreg", "o1-2024-12-17") is not None  # dated version -> same family
        assert get_model_info("testreg", "o1-mini") is None  # variant -> must NOT inherit o1 pricing
        assert get_model_info("testreg", "o1-preview") is None
    finally:
        _REGISTRY.pop("testreg/o1", None)
