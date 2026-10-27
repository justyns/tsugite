"""Tests for the cache-usage breakdown derivation."""

from __future__ import annotations

from tsugite.usage.cache import derive_cache_breakdown
from tsugite.usage.store import UsageStore


class TestDeriveCacheBreakdown:
    def test_anthropic_input_tokens_are_already_uncached(self):
        # Anthropic's input_tokens excludes cache tokens. They're additive.
        result = derive_cache_breakdown("anthropic:claude-opus-4-8", 100, 50, 900)

        assert result["cached_read_tokens"] == 900
        assert result["cache_write_tokens"] == 50
        assert result["uncached_input_tokens"] == 100

    def test_claude_code_uses_the_same_additive_scheme_as_anthropic(self):
        result = derive_cache_breakdown("claude_code:opus", 100, 50, 900)

        assert result["cached_read_tokens"] == 900
        assert result["cache_write_tokens"] == 50
        assert result["uncached_input_tokens"] == 100

    def test_acp_uses_the_same_additive_scheme_as_anthropic(self):
        result = derive_cache_breakdown("acp:claude", 100, 2000, 5000)

        assert result["cached_read_tokens"] == 5000
        assert result["cache_write_tokens"] == 2000
        assert result["uncached_input_tokens"] == 100

    def test_openai_compat_input_tokens_include_the_cached_subset(self):
        # OpenAI-style prompt_tokens is inclusive of cached_tokens.
        result = derive_cache_breakdown("openai:gpt-4o", 1000, 0, 900)

        assert result["cached_read_tokens"] == 900
        assert result["cache_write_tokens"] is None
        assert result["uncached_input_tokens"] == 100

    def test_unknown_provider_falls_back_to_the_openai_compat_scheme(self):
        result = derive_cache_breakdown("ollama:qwen2.5-coder:7b", 500, 0, 0)

        assert result["cache_write_tokens"] is None
        assert result["uncached_input_tokens"] == 500

    def test_no_colon_in_model_falls_back_to_the_openai_compat_scheme(self):
        result = derive_cache_breakdown(None, 500, 0, 0)

        assert result["cache_write_tokens"] is None
        assert result["uncached_input_tokens"] == 500

    def test_cache_hit_rate(self):
        result = derive_cache_breakdown("anthropic:claude-opus-4-8", 100, 0, 900)

        assert result["cache_hit_rate"] == 0.9

    def test_cache_hit_rate_counts_cache_writes_as_input_for_additive_providers(self):
        result = derive_cache_breakdown("anthropic:claude-opus-4-8", 100, 800, 100)

        assert result["cache_hit_rate"] == 0.1

    def test_cache_hit_rate_is_read_over_input_for_inclusive_providers(self):
        result = derive_cache_breakdown("openai:gpt-4o", 1000, 0, 250)

        assert result["cache_hit_rate"] == 0.25

    def test_inclusive_provider_with_reads_above_input_is_clamped_and_logged(self, caplog):
        result = derive_cache_breakdown("openai:gpt-4o", 100, 0, 500)

        assert result["uncached_input_tokens"] == 0
        assert "exceed input_tokens" in caplog.text

    def test_cache_hit_rate_is_none_with_no_input(self):
        result = derive_cache_breakdown("anthropic:claude-opus-4-8", 0, 0, 0)

        assert result["cache_hit_rate"] is None


class TestTopModelsCacheBreakdown:
    def test_top_models_includes_the_derived_fields(self, tmp_path):
        store = UsageStore(db_path=tmp_path / "usage.db")
        store.record(
            model="anthropic:claude-opus-4-8", input_tokens=100, cache_creation_tokens=50, cache_read_tokens=900
        )

        row = store.top_models()[0]

        assert row["cached_read_tokens"] == 900
        assert row["cache_write_tokens"] == 50
        assert row["uncached_input_tokens"] == 100
        assert row["cache_hit_rate"] == 900 / 1050
