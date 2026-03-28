"""Tests for model registry — verify no duplicate model keys."""

from tsugite.providers.anthropic import _ANTHROPIC_MODELS
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
