"""Tests for tsugite.exceptions helpers."""

from tsugite.exceptions import AgentExecutionError, is_prompt_too_long_error


def test_is_prompt_too_long_error_matches_real_provider_format():
    assert is_prompt_too_long_error("Prompt is too long (subtype=success)")
    assert is_prompt_too_long_error("Prompt is too long (subtype=error_during_execution)")
    assert is_prompt_too_long_error("prompt too long")
    assert is_prompt_too_long_error("context length exceeded")
    assert is_prompt_too_long_error(AgentExecutionError("Prompt is too long (subtype=success)"))


def test_is_prompt_too_long_error_rejects_unrelated_failures():
    assert not is_prompt_too_long_error("rate limit exceeded")
    assert not is_prompt_too_long_error("process ended")
    assert not is_prompt_too_long_error("")
