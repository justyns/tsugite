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
