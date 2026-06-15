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
