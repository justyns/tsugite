"""Tests for the provider registry caching behavior."""

import pytest

from tsugite.providers import clear_cache, get_provider


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_cache()
    yield
    clear_cache()


class TestProviderCaching:
    def test_stateless_provider_is_cached(self):
        p1 = get_provider("anthropic")
        p2 = get_provider("anthropic")
        assert p1 is p2

    def test_stateful_provider_not_cached(self):
        p1 = get_provider("claude_code")
        p2 = get_provider("claude_code")
        assert p1 is not p2

    def test_clear_cache_returns_fresh_instance(self):
        p1 = get_provider("anthropic")
        clear_cache()
        p2 = get_provider("anthropic")
        assert p1 is not p2

    def test_cacheable_attribute_on_claude_code(self):
        from tsugite.providers.claude_code import ClaudeCodeProvider

        assert ClaudeCodeProvider.cacheable is False
        assert ClaudeCodeProvider().cacheable is False

    def test_cacheable_default_true_on_anthropic(self):
        p = get_provider("anthropic")
        assert getattr(p, "cacheable", True) is True

    def test_concurrent_claude_code_providers_independent(self):
        """Each get_provider call returns a fresh instance with no shared state."""
        p1 = get_provider("claude_code")
        p2 = get_provider("claude_code")

        # Mutate one instance's state
        p1._session_id = "test-session"
        p1._turn_count = 5

        # Other instance should be unaffected
        assert p2._session_id is None
        assert p2._turn_count == 0

    def test_concurrent_sessions_isolate_all_claude_code_state(self):
        """Issue #321: two daemon sessions on the same agent must each get an
        independent provider instance with no leakage of session_id, cost/cache
        counters, context_window, or compacted flag.

        Pre-1e05c67 (cacheable=False on ClaudeCodeProvider), `get_provider` returned
        a single cached instance to every caller, so session B started a turn
        with session A's `_session_id` still set and would `--resume` into A's
        CLI conversation. Same for the cumulative cost / cache counters used
        for per-turn delta math.
        """
        from tsugite.models import get_provider_and_model

        _, prov_a, _ = get_provider_and_model("claude_code:opus")
        _, prov_b, _ = get_provider_and_model("claude_code:opus")

        assert prov_a is not prov_b, (
            "TsugiteAgent re-uses get_provider_and_model per turn; both sessions must get fresh providers"
        )

        # Simulate session A finishing a turn (subprocess session id, cumulative
        # cost/cache counters, context window, compaction flag all populated).
        prov_a._session_id = "cc-sess-A"
        prov_a._cumulative_cost = 1.50
        prov_a._cache_creation_tokens = 1000
        prov_a._cache_read_tokens = 500
        prov_a._context_window = 100000
        prov_a._compacted = True

        assert prov_b._session_id is None
        assert prov_b._cumulative_cost == 0.0
        assert prov_b._cache_creation_tokens == 0
        assert prov_b._cache_read_tokens == 0
        assert prov_b._context_window is None
        assert prov_b._compacted is False
