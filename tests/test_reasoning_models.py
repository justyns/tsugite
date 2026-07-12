"""Tests for reasoning model support and parameter filtering."""

import logging

import pytest

from tsugite.models import (
    RESERVED_MODEL_KWARGS,
    UnsupportedEffortError,
    get_model_id,
    get_model_kwargs,
    is_reasoning_model_without_stop_support,
    resolve_reasoning_effort,
    strip_reserved_model_kwargs,
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


class TestStripReservedModelKwargs:
    """strip_reserved_model_kwargs() drops provider-call args that would collide
    with the explicit keywords splatted alongside **model_kwargs in core/agent.py
    (`acompletion(messages=..., model=..., stream=..., **model_kwargs)`)."""

    def test_reserved_set_is_messages_model_stream(self):
        assert RESERVED_MODEL_KWARGS == frozenset({"messages", "model", "stream"})

    def test_drops_all_reserved_keys(self):
        cleaned = strip_reserved_model_kwargs(
            {"messages": [{"role": "user"}], "model": "evil", "stream": True, "temperature": 0.5}
        )
        assert cleaned == {"temperature": 0.5}

    def test_does_not_mutate_input(self):
        original = {"model": "evil", "temperature": 0.5}
        strip_reserved_model_kwargs(original)
        assert original == {"model": "evil", "temperature": 0.5}

    def test_warns_for_each_dropped_key(self, caplog):
        with caplog.at_level(logging.WARNING):
            strip_reserved_model_kwargs({"model": "evil", "stream": True, "temperature": 0.5})
        dropped = {r.message for r in caplog.records if "Dropping reserved model_kwargs key" in r.message}
        assert any("'model'" in m for m in dropped)
        assert any("'stream'" in m for m in dropped)
        # Non-reserved keys never warn.
        assert not any("'temperature'" in m for m in dropped)

    def test_no_reserved_keys_passes_through_silently(self, caplog):
        with caplog.at_level(logging.WARNING):
            cleaned = strip_reserved_model_kwargs({"temperature": 0.5, "reasoning_effort": "high"})
        assert cleaned == {"temperature": 0.5, "reasoning_effort": "high"}
        assert not any("Dropping reserved" in r.message for r in caplog.records)

    def test_get_model_kwargs_strips_reserved(self):
        """get_model_kwargs (called from core/agent.py) is the second strip layer:
        a reserved key in model_kwargs must never reach acompletion()."""
        kwargs = get_model_kwargs("openai:gpt-4", model="evil", stream=True, temperature=0.7)
        assert "model" not in kwargs
        assert "stream" not in kwargs
        assert kwargs.get("temperature") == 0.7


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
        from tsugite_claude_code.provider import ClaudeCodeProvider

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


class _RecordingProvider:
    """Fake provider that records the `**model_kwargs` reaching acompletion().

    Returns a one-shot `final_answer()` code block so the agent terminates in a
    single turn. The agent calls
    `acompletion(messages=..., model=..., stream=..., **self._model_kwargs)`, so
    a reserved key leaking into model_kwargs would raise "got multiple values for
    keyword argument" right here - which is exactly what the boundary tests guard.
    """

    name = "anthropic"
    cacheable = True

    def __init__(self, effort_levels=None):
        self.calls: list[dict] = []
        self._effort_levels = effort_levels

    async def acompletion(self, messages, model, stream=False, **kwargs):
        from tsugite.providers.base import CompletionResponse, Usage

        self.calls.append(kwargs)
        return CompletionResponse(
            content="finishing up\n```python-exec\nfinal_answer('ok')\n```",
            usage=Usage(total_tokens=5),
            cost=0.0,
        )

    def count_tokens(self, text, model):
        return max(1, len(text) // 4)

    def get_model_info(self, model):
        from tsugite.providers.base import ModelInfo

        return ModelInfo(max_input_tokens=200_000, supported_effort_levels=self._effort_levels)

    async def list_models(self):
        return []

    def set_context(self, **kwargs):
        pass

    def get_state(self):
        return None

    async def stop(self):
        pass


async def _run_agent_capturing_kwargs(
    tmp_path,
    monkeypatch,
    *,
    frontmatter_lines: str,
    model: str = "anthropic:claude-3-haiku-20240307",
    effort_override: str | None = None,
    effort_levels=None,
) -> dict:
    """Drive run_agent_async end-to-end with a recording provider, return the
    captured model_kwargs from the single acompletion call."""
    from unittest.mock import patch

    from tsugite.agent_runner.runner import run_agent_async
    from tsugite.options import ExecutionOptions

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    agent_path = tmp_path / "agent.md"
    # extends: none keeps the agent off the default base template (which would
    # auto-inject spawn_agent / list_available_agents tools the bare test
    # registry doesn't have); we only care about the model_kwargs path here.
    agent_path.write_text(f"---\nname: probe\nextends: none\nmodel: {model}\n{frontmatter_lines}---\nDo the task.\n")

    provider = _RecordingProvider(effort_levels=effort_levels)
    opts = ExecutionOptions(reasoning_effort_override=effort_override) if effort_override else ExecutionOptions()
    model_id = model.split(":", 1)[1]
    with patch(
        "tsugite.models.get_provider_and_model",
        return_value=(provider.name, provider, model_id),
    ):
        await run_agent_async(agent_path, "go", exec_options=opts)

    assert provider.calls, "provider.acompletion was never called"
    return provider.calls[0]


class TestRunnerModelKwargsBoundary:
    """End-to-end (run_agent_async) guards for the merged model_kwargs reaching
    the provider splat: reserved keys must be filtered, reasoning_effort
    precedence must hold. Reverting both strip layers makes
    `test_reserved_frontmatter_key_does_not_raise` fail with
    "got multiple values for keyword argument 'model'"."""

    @pytest.mark.asyncio
    async def test_reserved_frontmatter_key_does_not_raise(self, tmp_path, monkeypatch):
        # Reserved keys baked into agent frontmatter model_kwargs must be stripped
        # before the **splat, not splatted alongside the explicit model=/stream=.
        kwargs = await _run_agent_capturing_kwargs(
            tmp_path,
            monkeypatch,
            frontmatter_lines=("model_kwargs:\n  model: EVIL\n  stream: true\n  messages: EVIL\n  temperature: 0.5\n"),
        )
        assert "model" not in kwargs
        assert "stream" not in kwargs
        assert "messages" not in kwargs
        assert kwargs.get("temperature") == 0.5

    @pytest.mark.asyncio
    async def test_effort_override_beats_agent_model_kwargs(self, tmp_path, monkeypatch):
        # exec_options override is the top-priority knob; it must beat a
        # reasoning_effort baked into agent.model_kwargs (which would otherwise
        # shadow it after the merge).
        kwargs = await _run_agent_capturing_kwargs(
            tmp_path,
            monkeypatch,
            model="claude_code:opus",
            frontmatter_lines="model_kwargs:\n  reasoning_effort: low\n",
            effort_override="high",
            effort_levels=["low", "medium", "high", "max"],
        )
        assert kwargs.get("reasoning_effort") == "high"

    @pytest.mark.asyncio
    async def test_agent_model_kwargs_beats_reasoning_effort_field(self, tmp_path, monkeypatch):
        # agent.model_kwargs.reasoning_effort outranks the agent.reasoning_effort
        # field when no override is supplied.
        kwargs = await _run_agent_capturing_kwargs(
            tmp_path,
            monkeypatch,
            model="claude_code:opus",
            frontmatter_lines="reasoning_effort: low\nmodel_kwargs:\n  reasoning_effort: high\n",
            effort_levels=["low", "medium", "high", "max"],
        )
        assert kwargs.get("reasoning_effort") == "high"

    @pytest.mark.asyncio
    async def test_reasoning_effort_field_used_as_lowest_fallback(self, tmp_path, monkeypatch):
        # With no override and no model_kwargs effort, the agent.reasoning_effort
        # field is the final fallback.
        kwargs = await _run_agent_capturing_kwargs(
            tmp_path,
            monkeypatch,
            model="claude_code:opus",
            frontmatter_lines="reasoning_effort: medium\n",
            effort_levels=["low", "medium", "high", "max"],
        )
        assert kwargs.get("reasoning_effort") == "medium"
