"""Tests for Claude Code LLM provider."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tsugite.models import _CLAUDE_CODE_MODEL_MAP, get_model_params, parse_model_string


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

    def test_get_model_params_claude_code(self):
        params = get_model_params("claude_code:sonnet")
        assert params["model"] == "sonnet"
        assert params["_provider"] == "claude_code"

    def test_get_model_params_claude_code_opus(self):
        params = get_model_params("claude_code:opus")
        assert params["model"] == "opus"
        assert params["_provider"] == "claude_code"

    def test_get_model_params_claude_code_haiku(self):
        params = get_model_params("claude_code:haiku")
        assert params["model"] == "haiku"
        assert params["_provider"] == "claude_code"

    def test_get_model_params_preserves_kwargs(self):
        params = get_model_params("claude_code:sonnet", temperature=0.5)
        assert params["_provider"] == "claude_code"
        assert params["temperature"] == 0.5

    def test_get_model_params_full_model_id(self):
        params = get_model_params("claude_code:claude-sonnet-4-6")
        assert params["model"] == "claude-sonnet-4-6"
        assert params["_provider"] == "claude_code"

    def test_litellm_model_mapping_short_names(self):
        for short, full in _CLAUDE_CODE_MODEL_MAP.items():
            params = get_model_params(f"claude_code:{short}")
            assert params["_litellm_model"] == full

    def test_litellm_model_mapping_full_id_passthrough(self):
        params = get_model_params("claude_code:claude-sonnet-4-6")
        assert params["_litellm_model"] == "claude-sonnet-4-6"


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
    async def test_start_with_resume_session(self, process):
        mock_proc = self._mock_proc()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await process.start(model="sonnet", system_prompt="test", resume_session="old-session-id")
            cmd_args = mock_exec.call_args[0]
            assert "--resume" in cmd_args
            assert "old-session-id" in cmd_args[cmd_args.index("--resume") + 1]

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
    async def test_send_message_captures_init_event(self, process):
        """Init event arrives after first user message -- session_id gets captured."""
        events = [
            json.dumps({"type": "system", "subtype": "init", "session_id": "captured-session"}),
            json.dumps({"type": "result", "subtype": "success", "result": "ok", "total_cost_usd": 0.001, "duration_ms": 100, "session_id": "captured-session"}),
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
            json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello world"}], "usage": {"input_tokens": 100, "output_tokens": 50}}, "session_id": "s1"}),
            json.dumps({"type": "result", "subtype": "success", "result": "Hello world", "total_cost_usd": 0.015, "duration_ms": 1200, "session_id": "s1"}),
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
            json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello world"}]}, "session_id": "s1"}),
            json.dumps({"type": "result", "subtype": "success", "result": "Hello world", "total_cost_usd": 0.003, "duration_ms": 500, "session_id": "s1"}),
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
        events = [json.dumps({
            "type": "result", "subtype": "success", "result": "done",
            "total_cost_usd": 0.001, "duration_ms": 100, "session_id": "s1",
        })]
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
    def test_agent_detects_claude_code_provider(self):
        from tsugite.core.agent import TsugiteAgent

        agent = TsugiteAgent(
            model_string="claude_code:sonnet",
            tools=[],
            instructions="test",
        )
        assert agent._is_claude_code is True
        assert agent._claude_code_model == "sonnet"

    def test_agent_non_claude_code_provider(self):
        from tsugite.core.agent import TsugiteAgent

        with patch("litellm.acompletion"):
            agent = TsugiteAgent(
                model_string="openai:gpt-4o-mini",
                tools=[],
                instructions="test",
            )
            assert agent._is_claude_code is False

    @pytest.mark.asyncio
    async def test_agent_run_uses_claude_code_process(self):
        from tsugite.core.agent import TsugiteAgent

        agent = TsugiteAgent(
            model_string="claude_code:sonnet",
            tools=[],
            instructions="test",
            max_turns=2,
        )

        # Mock the ClaudeCodeProcess
        mock_process = AsyncMock()
        mock_process.session_id = "test-session"

        # Simulate a response that calls final_answer
        async def mock_send(*args, **kwargs):
            events = [
                {"type": "text_delta", "text": "Thought: simple\n```python\nfinal_answer('42')\n```"},
                {"type": "result", "text": "Thought: simple\n```python\nfinal_answer('42')\n```", "cost_usd": 0.001, "session_id": "test-session", "input_tokens": 500, "output_tokens": 100},
            ]
            for e in events:
                yield e

        mock_process.send_message = mock_send
        mock_process.start = AsyncMock()
        mock_process.stop = AsyncMock()

        with patch("tsugite.core.claude_code.ClaudeCodeProcess", return_value=mock_process):
            result = await agent.run("what is 6*7")

        assert result == "42"
        assert agent._claude_code_last_turn_tokens == 600
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

        async def mock_send(*args, **kwargs):
            events = [
                {"type": "text_delta", "text": "Thought: done\n```python\nfinal_answer('hi')\n```"},
                {"type": "result", "text": "", "cost_usd": 0.01, "session_id": "test-session", "input_tokens": 1000, "output_tokens": 200},
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

        async def mock_send(*args, **kwargs):
            events = [
                {"type": "text_delta", "text": "Thought: done\n```python\nfinal_answer('hi')\n```"},
                {
                    "type": "result", "text": "", "cost_usd": 0.01,
                    "session_id": "test-session",
                    "input_tokens": 500, "output_tokens": 100,
                    "cache_creation_input_tokens": 300, "cache_read_input_tokens": 200,
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
    def test_agent_result_has_session_id(self):
        from tsugite.core.agent import AgentResult

        result = AgentResult(output="test", claude_code_session_id="session-abc")
        assert result.claude_code_session_id == "session-abc"

    def test_agent_result_session_id_defaults_none(self):
        from tsugite.core.agent import AgentResult

        result = AgentResult(output="test")
        assert result.claude_code_session_id is None

    def test_execution_result_has_session_id(self):
        from tsugite.agent_runner.models import AgentExecutionResult

        result = AgentExecutionResult(response="test", claude_code_session_id="session-xyz")
        assert result.claude_code_session_id == "session-xyz"

    def test_execution_result_session_id_defaults_none(self):
        from tsugite.agent_runner.models import AgentExecutionResult

        result = AgentExecutionResult(response="test")
        assert result.claude_code_session_id is None


# ── Context limit resolution tests ──


class TestClaudeCodeContextLimit:
    def test_context_limit_uses_claude_code_override(self):
        from tsugite.daemon.memory import _CLAUDE_CODE_CONTEXT_LIMIT, get_context_limit

        limit = get_context_limit("claude_code:opus")
        assert limit == _CLAUDE_CODE_CONTEXT_LIMIT

    def test_context_limit_claude_code_ignores_litellm(self):
        from tsugite.daemon.memory import _CLAUDE_CODE_CONTEXT_LIMIT, get_context_limit

        with patch("litellm.get_model_info", side_effect=Exception("should not be called")):
            limit = get_context_limit("claude_code:opus")
            assert limit == _CLAUDE_CODE_CONTEXT_LIMIT

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

        async def mock_send(*args, **kwargs):
            events = [
                {"type": "text_delta", "text": "Thought: done\n```python\nfinal_answer('ok')\n```"},
                {
                    "type": "result", "text": "", "cost_usd": 0.01,
                    "session_id": "test-session",
                    "input_tokens": 500, "output_tokens": 100,
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

        assert result.context_window == 200000
