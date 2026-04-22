"""Tests for Claude Code LLM provider."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tsugite.models import _CLAUDE_CODE_MODEL_MAP, get_model_id, parse_model_string

# ── Model parsing tests ──


class TestClaudeCodeModelParams:
    def test_parse_claude_code_model_string(self):
        provider, model, variant = parse_model_string("claude_code:sonnet")
        assert provider == "claude_code"
        assert model == "sonnet"
        assert variant is None

    def test_parse_claude_code_full_model(self):
        provider, model, variant = parse_model_string("claude_code:claude-sonnet-4-6")
        assert provider == "claude_code"
        assert model == "claude-sonnet-4-6"

    def test_model_id_maps_short_names(self):
        assert get_model_id("claude_code:sonnet") == "claude-sonnet-4-6"
        assert get_model_id("claude_code:opus") == "claude-opus-4-7"
        assert get_model_id("claude_code:haiku") == "claude-haiku-4-5-20251001"

    def test_model_id_version_pinned_opus_aliases(self):
        assert get_model_id("claude_code:opus-4-7") == "claude-opus-4-7"
        assert get_model_id("claude_code:opus-4-6") == "claude-opus-4-6"

    def test_model_id_full_id_passthrough(self):
        assert get_model_id("claude_code:claude-sonnet-4-6") == "claude-sonnet-4-6"

    def test_model_map_completeness(self):
        for short, full in _CLAUDE_CODE_MODEL_MAP.items():
            assert get_model_id(f"claude_code:{short}") == full


# ── ClaudeCodeProcess tests ──


class TestClaudeCodeProcess:
    @pytest.fixture
    def process(self):
        from tsugite.core.claude_code import ClaudeCodeProcess

        return ClaudeCodeProcess()

    def _mock_proc(self, events: list[str] | None = None):
        """Create a mock subprocess with optional stdout events.

        Args:
            events: JSON strings to return from stdout.readline(). If None, stdout is a plain AsyncMock.
        """
        mock_proc = AsyncMock()
        mock_proc.stdin = AsyncMock()
        mock_proc.stderr = AsyncMock()
        mock_proc.stderr.readline = AsyncMock(return_value=b"")
        mock_proc.returncode = None
        mock_proc.stdout = AsyncMock()

        if events is not None:
            event_iter = iter(events)

            async def mock_readline():
                try:
                    return (next(event_iter) + "\n").encode()
                except StopIteration:
                    return b""

            mock_proc.stdout.readline = mock_readline

        return mock_proc

    @pytest.mark.asyncio
    async def test_start_launches_subprocess(self, process):
        mock_proc = self._mock_proc()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await process.start(model="sonnet", system_prompt="You are helpful")
            assert mock_exec.called
            cmd_args = mock_exec.call_args[0]
            assert "claude" in cmd_args[0]
            assert "--print" in cmd_args
            assert "--output-format" in cmd_args
            assert "stream-json" in cmd_args[cmd_args.index("--output-format") + 1]
            assert "--verbose" in cmd_args
            assert "--input-format" in cmd_args
            assert "--max-turns" in cmd_args
            assert "1" in cmd_args[cmd_args.index("--max-turns") + 1]
            assert "--model" in cmd_args
            assert "sonnet" in cmd_args[cmd_args.index("--model") + 1]
            assert mock_exec.call_args[1]["stderr"] == asyncio.subprocess.PIPE

        # session_id is None until first send_message (init comes after first user msg)
        assert process.session_id is None
        await process.stop()

    @pytest.mark.asyncio
    async def test_start_unsets_claude_env_vars(self, process):
        mock_proc = self._mock_proc()

        extra_vars = {"ANTHROPIC_API_KEY": "sk-test", "CLAUDECODE": "1", "CLAUDE_CODE_ENTRYPOINT": "cli"}
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            with patch.dict("os.environ", extra_vars):
                await process.start(model="sonnet", system_prompt="test")
                env = mock_exec.call_args[1]["env"]
                assert "ANTHROPIC_API_KEY" not in env
                assert "CLAUDECODE" not in env
                assert "CLAUDE_CODE_ENTRYPOINT" not in env

        await process.stop()

    @pytest.mark.asyncio
    async def test_start_passes_workspace_cv_as_cwd(self, process, tmp_path):
        """Subprocess runs in the workspace bound to the current task."""
        from tsugite.cli.helpers import set_workspace_dir

        mock_proc = self._mock_proc()
        workspace = tmp_path / "agent_ws"
        workspace.mkdir()
        set_workspace_dir(workspace)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await process.start(model="sonnet", system_prompt="test")
            assert mock_exec.call_args[1].get("cwd") == str(workspace), (
                f"expected cwd={workspace!s}, got {mock_exec.call_args[1].get('cwd')!r}"
            )

        await process.stop()

    @pytest.mark.asyncio
    async def test_start_no_cwd_when_workspace_unset(self, process):
        """No workspace bound → don't pass cwd=, let subprocess inherit."""
        mock_proc = self._mock_proc()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await process.start(model="sonnet", system_prompt="test")
            assert mock_exec.call_args[1].get("cwd") is None

        await process.stop()

    @pytest.mark.asyncio
    async def test_send_message_captures_init_event(self, process):
        """Init event arrives after first user message -- session_id gets captured."""
        events = [
            json.dumps({"type": "system", "subtype": "init", "session_id": "captured-session"}),
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "result": "ok",
                    "total_cost_usd": 0.001,
                    "duration_ms": 100,
                    "session_id": "captured-session",
                }
            ),
        ]
        process._process = self._mock_proc(events)

        assert process.session_id is None
        async for _ in process.send_message("hello"):
            pass
        assert process.session_id == "captured-session"

    @pytest.mark.asyncio
    async def test_send_message_parses_assistant_event(self, process):
        """Test parsing the real claude CLI format: assistant event with full text."""
        events = [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Hello world"}],
                        "usage": {"input_tokens": 100, "output_tokens": 50},
                    },
                    "session_id": "s1",
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "result": "Hello world",
                    "total_cost_usd": 0.015,
                    "duration_ms": 1200,
                    "session_id": "s1",
                }
            ),
        ]
        process._process = self._mock_proc(events)
        process._session_id = "s1"

        collected = []
        async for event in process.send_message("test prompt"):
            collected.append(event)

        text_deltas = [e for e in collected if e.get("type") == "text_delta"]
        assert len(text_deltas) == 1
        assert text_deltas[0]["text"] == "Hello world"

        results = [e for e in collected if e.get("type") == "result"]
        assert len(results) == 1
        assert results[0]["text"] == "Hello world"
        assert results[0]["cost_usd"] == 0.015
        assert results[0]["input_tokens"] == 100
        assert results[0]["output_tokens"] == 50

    @pytest.mark.asyncio
    async def test_send_message_parses_content_block_deltas(self, process):
        """Test parsing content_block_delta events (with --include-partial-messages)."""
        events = [
            json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}),
            json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": " world"}}),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello world"}]},
                    "session_id": "s1",
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "result": "Hello world",
                    "total_cost_usd": 0.003,
                    "duration_ms": 500,
                    "session_id": "s1",
                }
            ),
        ]
        process._process = self._mock_proc(events)
        process._session_id = "s1"

        collected = []
        async for event in process.send_message("test"):
            collected.append(event)

        text_deltas = [e for e in collected if e.get("type") == "text_delta"]
        # 2 deltas + 1 from assistant event
        assert len(text_deltas) == 3
        assert text_deltas[0]["text"] == "Hello"
        assert text_deltas[1]["text"] == " world"

    @pytest.mark.asyncio
    async def test_send_message_writes_correct_json(self, process):
        events = [
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "result": "done",
                    "total_cost_usd": 0.001,
                    "duration_ms": 100,
                    "session_id": "s1",
                }
            )
        ]
        mock_proc = self._mock_proc(events)
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock()

        process._process = mock_proc
        process._session_id = "s1"

        async for _ in process.send_message("hello"):
            pass

        # Verify stdin was written with correct JSON
        written = mock_proc.stdin.write.call_args[0][0]
        msg = json.loads(written.decode().strip())
        assert msg["type"] == "user"
        assert msg["message"]["role"] == "user"
        assert msg["message"]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_stop_terminates_process(self, process):
        mock_proc = AsyncMock()
        mock_proc.stdin = AsyncMock()
        mock_proc.returncode = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = AsyncMock()

        process._process = mock_proc
        process._system_prompt_file = None

        await process.stop()
        mock_proc.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_claude_not_found_raises(self, process):
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="Claude Code CLI not found"):
                await process.start(model="sonnet", system_prompt="test")

    @pytest.mark.asyncio
    async def test_subprocess_crash_raises(self, process):
        process._process = self._mock_proc(events=[])  # EOF immediately = crash
        process._session_id = "s1"

        with pytest.raises(RuntimeError, match="Claude Code process ended unexpectedly"):
            async for _ in process.send_message("hello"):
                pass


# ── Agent integration tests ──


class TestClaudeCodeAgentIntegration:
    def test_agent_uses_claude_code_provider(self):
        from tsugite.core.agent import TsugiteAgent
        from tsugite.providers.claude_code import ClaudeCodeProvider

        agent = TsugiteAgent(
            model_string="claude_code:sonnet",
            tools=[],
            instructions="test",
        )
        assert isinstance(agent._provider, ClaudeCodeProvider)
        assert agent._provider_name == "claude_code"

    def test_agent_non_claude_code_provider(self):
        from tsugite.core.agent import TsugiteAgent

        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[],
            instructions="test",
        )
        assert agent._provider_name == "openai"

    @pytest.mark.asyncio
    async def test_agent_run_uses_claude_code_process(self):
        from tsugite.core.agent import TsugiteAgent

        agent = TsugiteAgent(
            model_string="claude_code:sonnet",
            tools=[],
            instructions="test",
            max_turns=2,
        )

        mock_process = AsyncMock()
        mock_process.session_id = "test-session"
        mock_process.compacted = False

        async def mock_send(*args, **kwargs):
            events = [
                {"type": "text_delta", "text": "Thought: simple\n```python\nfinal_answer('42')\n```"},
                {
                    "type": "result",
                    "text": "Thought: simple\n```python\nfinal_answer('42')\n```",
                    "cost_usd": 0.001,
                    "session_id": "test-session",
                    "input_tokens": 500,
                    "output_tokens": 100,
                },
            ]
            for e in events:
                yield e

        mock_process.send_message = mock_send
        mock_process.start = AsyncMock()
        mock_process.stop = AsyncMock()

        with patch("tsugite.core.claude_code.ClaudeCodeProcess", return_value=mock_process):
            result = await agent.run("what is 6*7")

        assert result == "42"
        assert agent.total_tokens == 600
        mock_process.start.assert_called_once()
        mock_process.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_run_emits_tokens_in_final_answer(self):
        from tsugite.core.agent import TsugiteAgent
        from tsugite.events import EventBus, FinalAnswerEvent

        bus = EventBus()
        captured_events = []
        bus.subscribe(lambda e: captured_events.append(e) if isinstance(e, FinalAnswerEvent) else None)

        agent = TsugiteAgent(
            model_string="claude_code:sonnet",
            tools=[],
            instructions="test",
            max_turns=2,
            event_bus=bus,
        )

        mock_process = AsyncMock()
        mock_process.session_id = "test-session"
        mock_process.compacted = False

        async def mock_send(*args, **kwargs):
            events = [
                {"type": "text_delta", "text": "Thought: done\n```python\nfinal_answer('hi')\n```"},
                {
                    "type": "result",
                    "text": "",
                    "cost_usd": 0.01,
                    "session_id": "test-session",
                    "input_tokens": 1000,
                    "output_tokens": 200,
                },
            ]
            for e in events:
                yield e

        mock_process.send_message = mock_send
        mock_process.start = AsyncMock()
        mock_process.stop = AsyncMock()

        with patch("tsugite.core.claude_code.ClaudeCodeProcess", return_value=mock_process):
            await agent.run("hello")

        assert len(captured_events) == 1
        assert captured_events[0].tokens == 1200

    @pytest.mark.asyncio
    async def test_agent_run_emits_cache_tokens_in_cost_summary(self):
        from tsugite.core.agent import TsugiteAgent
        from tsugite.events import CostSummaryEvent, EventBus

        bus = EventBus()
        captured_events = []
        bus.subscribe(lambda e: captured_events.append(e) if isinstance(e, CostSummaryEvent) else None)

        agent = TsugiteAgent(
            model_string="claude_code:sonnet",
            tools=[],
            instructions="test",
            max_turns=2,
            event_bus=bus,
        )

        mock_process = AsyncMock()
        mock_process.session_id = "test-session"
        mock_process.compacted = False

        async def mock_send(*args, **kwargs):
            events = [
                {"type": "text_delta", "text": "Thought: done\n```python\nfinal_answer('hi')\n```"},
                {
                    "type": "result",
                    "text": "",
                    "cost_usd": 0.01,
                    "session_id": "test-session",
                    "input_tokens": 500,
                    "output_tokens": 100,
                    "cache_creation_input_tokens": 300,
                    "cache_read_input_tokens": 200,
                },
            ]
            for e in events:
                yield e

        mock_process.send_message = mock_send
        mock_process.start = AsyncMock()
        mock_process.stop = AsyncMock()

        with patch("tsugite.core.claude_code.ClaudeCodeProcess", return_value=mock_process):
            await agent.run("hello")

        assert len(captured_events) == 1
        assert captured_events[0].cache_creation_input_tokens == 300
        assert captured_events[0].cache_read_input_tokens == 200


# ── Session ID passthrough tests ──


class TestSessionIdPassthrough:
    def test_agent_result_has_provider_state(self):
        from tsugite.core.agent import AgentResult

        result = AgentResult(output="test", provider_state={"session_id": "session-abc"})
        assert result.provider_state["session_id"] == "session-abc"

    def test_agent_result_provider_state_defaults_none(self):
        from tsugite.core.agent import AgentResult

        result = AgentResult(output="test")
        assert result.provider_state is None

    def test_execution_result_has_provider_state(self):
        from tsugite.agent_runner.models import AgentExecutionResult

        result = AgentExecutionResult(response="test", provider_state={"session_id": "session-xyz"})
        assert result.provider_state["session_id"] == "session-xyz"

    def test_execution_result_provider_state_defaults_none(self):
        from tsugite.agent_runner.models import AgentExecutionResult

        result = AgentExecutionResult(response="test")
        assert result.provider_state is None


# ── Context limit resolution tests ──


class TestClaudeCodeContextLimit:
    def test_context_limit_opus_returns_1m(self):
        from tsugite.daemon.memory import get_context_limit

        limit = get_context_limit("claude_code:opus")
        assert limit == 1_000_000

    def test_context_limit_sonnet_returns_1m(self):
        from tsugite.daemon.memory import get_context_limit

        limit = get_context_limit("claude_code:sonnet")
        assert limit == 1_000_000

    def test_context_limit_haiku_returns_200k(self):
        from tsugite.daemon.memory import get_context_limit

        limit = get_context_limit("claude_code:haiku")
        assert limit == 200_000

    def test_context_limit_claude_code_from_registry(self):
        from tsugite.daemon.memory import get_context_limit

        limit = get_context_limit("claude_code:opus")
        assert limit == 1_000_000

    @pytest.mark.asyncio
    async def test_context_window_flows_to_agent_result(self):
        from tsugite.core.agent import TsugiteAgent

        agent = TsugiteAgent(
            model_string="claude_code:sonnet",
            tools=[],
            instructions="test",
            max_turns=2,
        )

        mock_process = AsyncMock()
        mock_process.session_id = "test-session"
        mock_process.compacted = False

        async def mock_send(*args, **kwargs):
            events = [
                {"type": "text_delta", "text": "Thought: done\n```python\nfinal_answer('ok')\n```"},
                {
                    "type": "result",
                    "text": "",
                    "cost_usd": 0.01,
                    "session_id": "test-session",
                    "input_tokens": 500,
                    "output_tokens": 100,
                    "context_window": 200000,
                },
            ]
            for e in events:
                yield e

        mock_process.send_message = mock_send
        mock_process.start = AsyncMock()
        mock_process.stop = AsyncMock()

        with patch("tsugite.core.claude_code.ClaudeCodeProcess", return_value=mock_process):
            result = await agent.run("hello", return_full_result=True)

        assert result.provider_state["context_window"] == 200000


# ── Cost and token tracking bug tests ──


class TestClaudeCodeCostTracking:
    """Tests for cost delta calculation — Claude CLI reports cumulative cost."""

    @pytest.mark.asyncio
    async def test_cost_is_delta_not_cumulative(self):
        """total_cost_usd from Claude CLI is cumulative; provider must return per-turn delta."""
        from tsugite.providers.claude_code import ClaudeCodeProvider

        provider = ClaudeCodeProvider()
        mock_process = AsyncMock()
        mock_process.session_id = "test-session"
        mock_process.compacted = False

        turn_costs = [0.005, 0.010, 0.015]  # Cumulative costs from CLI
        call_count = 0

        original_send = None

        async def mock_send(*args, **kwargs):
            nonlocal call_count
            cost = turn_costs[call_count]
            call_count += 1
            events = [
                {
                    "type": "result",
                    "text": f"answer {call_count}",
                    "cost_usd": cost,
                    "session_id": "test-session",
                    "input_tokens": 500,
                    "output_tokens": 50,
                },
            ]
            for e in events:
                yield e

        mock_process.send_message = mock_send
        mock_process.start = AsyncMock()
        mock_process.stop = AsyncMock()

        with patch("tsugite.core.claude_code.ClaudeCodeProcess", return_value=mock_process):
            # Turn 1
            resp1 = await provider.acompletion(
                messages=[{"role": "system", "content": "test"}, {"role": "user", "content": "q1"}],
                model="sonnet",
                stream=False,
            )
            # Turn 2
            resp2 = await provider.acompletion(
                messages=[{"role": "user", "content": "q2"}], model="sonnet", stream=False
            )
            # Turn 3
            resp3 = await provider.acompletion(
                messages=[{"role": "user", "content": "q3"}], model="sonnet", stream=False
            )
            await provider.stop()

        # Each response cost should be the per-turn delta, not cumulative
        assert resp1.cost == pytest.approx(0.005)
        assert resp2.cost == pytest.approx(0.005)  # 0.010 - 0.005
        assert resp3.cost == pytest.approx(0.005)  # 0.015 - 0.010

    @pytest.mark.asyncio
    async def test_cost_delta_streaming(self):
        """Cost delta works correctly in streaming mode too."""
        from tsugite.providers.claude_code import ClaudeCodeProvider

        provider = ClaudeCodeProvider()
        mock_process = AsyncMock()
        mock_process.session_id = "test-session"
        mock_process.compacted = False

        turn_costs = [0.005, 0.012]
        call_count = 0

        async def mock_send(*args, **kwargs):
            nonlocal call_count
            cost = turn_costs[call_count]
            call_count += 1
            events = [
                {"type": "text_delta", "text": "hello"},
                {
                    "type": "result",
                    "text": "hello",
                    "cost_usd": cost,
                    "session_id": "test-session",
                    "input_tokens": 500,
                    "output_tokens": 50,
                },
            ]
            for e in events:
                yield e

        mock_process.send_message = mock_send
        mock_process.start = AsyncMock()
        mock_process.stop = AsyncMock()

        with patch("tsugite.core.claude_code.ClaudeCodeProcess", return_value=mock_process):
            # Turn 1 streaming
            chunks1 = []
            async for chunk in await provider.acompletion(
                messages=[{"role": "system", "content": "test"}, {"role": "user", "content": "q1"}],
                model="sonnet",
                stream=True,
            ):
                chunks1.append(chunk)

            # Turn 2 streaming
            chunks2 = []
            async for chunk in await provider.acompletion(
                messages=[{"role": "user", "content": "q2"}], model="sonnet", stream=True
            ):
                chunks2.append(chunk)

            await provider.stop()

        final1 = [c for c in chunks1 if c.done][0]
        final2 = [c for c in chunks2 if c.done][0]

        assert final1.cost == pytest.approx(0.005)
        assert final2.cost == pytest.approx(0.007)  # 0.012 - 0.005


class TestClaudeCodeContextWindow:
    """Tests for contextWindow extraction from nested modelUsage."""

    @pytest.mark.asyncio
    async def test_context_window_extracted_from_nested_model_usage(self):
        """modelUsage is keyed by model name; contextWindow is inside the nested dict."""
        from tsugite.core.claude_code import ClaudeCodeProcess

        process = ClaudeCodeProcess()

        events = [
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "result": "ok",
                    "total_cost_usd": 0.005,
                    "duration_ms": 500,
                    "session_id": "s1",
                    "usage": {"input_tokens": 1500, "output_tokens": 5},
                    "modelUsage": {
                        "claude-sonnet-4-6": {
                            "inputTokens": 1500,
                            "outputTokens": 5,
                            "contextWindow": 200000,
                            "maxOutputTokens": 32000,
                        }
                    },
                }
            ),
        ]

        mock_proc = AsyncMock()
        mock_proc.stdin = AsyncMock()
        mock_proc.stderr = AsyncMock()
        mock_proc.stderr.readline = AsyncMock(return_value=b"")
        mock_proc.stdout = AsyncMock()
        event_iter = iter(events)

        async def mock_readline():
            try:
                return (next(event_iter) + "\n").encode()
            except StopIteration:
                return b""

        mock_proc.stdout.readline = mock_readline
        process._process = mock_proc
        process._session_id = "s1"

        collected = []
        async for event in process.send_message("test"):
            collected.append(event)

        results = [e for e in collected if e["type"] == "result"]
        assert len(results) == 1
        assert results[0]["context_window"] == 200000

    @pytest.mark.asyncio
    async def test_context_window_none_when_no_model_usage(self):
        """context_window should be None when modelUsage is missing."""
        from tsugite.core.claude_code import ClaudeCodeProcess

        process = ClaudeCodeProcess()

        events = [
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "result": "ok",
                    "total_cost_usd": 0.005,
                    "duration_ms": 500,
                    "session_id": "s1",
                    "usage": {"input_tokens": 1500, "output_tokens": 5},
                }
            ),
        ]

        mock_proc = AsyncMock()
        mock_proc.stdin = AsyncMock()
        mock_proc.stderr = AsyncMock()
        mock_proc.stderr.readline = AsyncMock(return_value=b"")
        mock_proc.stdout = AsyncMock()
        event_iter = iter(events)

        async def mock_readline():
            try:
                return (next(event_iter) + "\n").encode()
            except StopIteration:
                return b""

        mock_proc.stdout.readline = mock_readline
        process._process = mock_proc
        process._session_id = "s1"

        collected = []
        async for event in process.send_message("test"):
            collected.append(event)

        results = [e for e in collected if e["type"] == "result"]
        assert results[0]["context_window"] is None


class TestClaudeCodeMultiTurnTokens:
    """Tests for multi-turn token tracking through the agent."""

    @pytest.mark.asyncio
    async def test_multi_turn_total_tokens_is_sum(self):
        """total_tokens should be the sum of all turns' tokens."""
        from tsugite.core.agent import TsugiteAgent

        agent = TsugiteAgent(
            model_string="claude_code:sonnet",
            tools=[],
            instructions="test",
            max_turns=5,
        )

        turn_data = [
            {"input": 1500, "output": 50, "code": "x = 1\nprint(x)"},
            {"input": 1550, "output": 50, "code": "final_answer('done')"},
        ]
        call_count = 0

        mock_process = AsyncMock()
        mock_process.session_id = "test-session"
        mock_process.compacted = False

        async def mock_send(*args, **kwargs):
            nonlocal call_count
            td = turn_data[call_count]
            call_count += 1
            code = td["code"]
            events = [
                {"type": "text_delta", "text": f"Thought: step\n```python\n{code}\n```"},
                {
                    "type": "result",
                    "text": "",
                    "cost_usd": 0.005 * call_count,
                    "session_id": "test-session",
                    "input_tokens": td["input"],
                    "output_tokens": td["output"],
                },
            ]
            for e in events:
                yield e

        mock_process.send_message = mock_send
        mock_process.start = AsyncMock()
        mock_process.stop = AsyncMock()

        with patch("tsugite.core.claude_code.ClaudeCodeProcess", return_value=mock_process):
            result = await agent.run("test", return_full_result=True)

        # total_tokens = sum of all turns
        expected_total = (1500 + 50) + (1550 + 50)
        assert agent.total_tokens == expected_total

        # last_input_tokens = last turn's input only
        assert agent.last_input_tokens == 1550
