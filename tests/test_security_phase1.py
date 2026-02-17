"""Tests for Phase 1 security changes: parent_only flag, audit events, run_safe removal."""

import json

import pytest

from tsugite.events import EventType, ToolCallEvent, ToolResultEvent
from tsugite.tools import get_tool, list_tools, tool


# ============================================================================
# parent_only flag
# ============================================================================


class TestParentOnlyFlag:
    def test_tool_without_parent_only(self, reset_tool_registry):
        @tool
        def normal_tool():
            """A normal tool."""
            return "ok"

        info = get_tool("normal_tool")
        assert info.parent_only is False

    def test_tool_with_parent_only(self, reset_tool_registry):
        @tool(parent_only=True)
        def special_tool():
            """Runs in parent only."""
            return "ok"

        info = get_tool("special_tool")
        assert info.parent_only is True

    def test_parent_only_on_interactive_tools(self):
        """ask_user and ask_user_batch should be marked parent_only via function attribute."""
        from tsugite.tools.interactive import ask_user, ask_user_batch

        assert getattr(ask_user, "_parent_only", False) is True
        assert getattr(ask_user_batch, "_parent_only", False) is True

    def test_parent_only_on_spawn_agent(self):
        import importlib

        import tsugite.tools.agents as agents_mod

        importlib.reload(agents_mod)
        assert getattr(agents_mod.spawn_agent, "_parent_only", False) is True

    def test_parent_only_on_load_skill(self):
        from tsugite.tools.skills import load_skill

        assert getattr(load_skill, "_parent_only", False) is True


# ============================================================================
# Audit events
# ============================================================================


class TestAuditEvents:
    def test_tool_call_event_serializes(self):
        event = ToolCallEvent(tool_name="read_file", arguments={"path": "/tmp/x"}, step=1)
        assert event.event_type == EventType.TOOL_CALL
        assert event.tool_name == "read_file"
        assert event.arguments == {"path": "/tmp/x"}
        assert event.step == 1

    def test_tool_result_event_success(self):
        event = ToolResultEvent(tool_name="read_file", success=True, result_summary="contents...", duration_ms=42)
        assert event.event_type == EventType.TOOL_RESULT
        assert event.success is True
        assert event.duration_ms == 42

    def test_tool_result_event_failure(self):
        event = ToolResultEvent(tool_name="run", success=False, result_summary="command failed", duration_ms=100)
        assert event.success is False
        assert event.result_summary == "command failed"

    def test_tool_call_event_default_arguments(self):
        event = ToolCallEvent(tool_name="get_system_info")
        assert event.arguments == {}
        assert event.step is None

    def test_audit_events_emitted_by_tool_wrapper(self, reset_tool_registry):
        """Tool wrapper emits ToolCallEvent before and ToolResultEvent after execution."""
        from tsugite.core.agent import TsugiteAgent
        from tsugite.core.tools import Tool
        from tsugite.events import EventBus

        collected = []

        def handler(event):
            collected.append(event)

        bus = EventBus()
        bus.subscribe(handler)

        # Create a simple tool
        async def dummy_read(path: str = ".") -> str:
            return "file contents"

        tools = [
            Tool(
                name="dummy_read",
                description="Read a file",
                parameters={"path": {"type": "string", "description": "path"}},
                function=dummy_read,
            )
        ]

        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=tools,
            event_bus=bus,
        )

        # Call the tool wrapper directly
        wrapper = agent.executor.namespace.get("dummy_read")
        assert wrapper is not None
        result = wrapper(path="test.txt")
        assert result == "file contents"

        call_events = [e for e in collected if isinstance(e, ToolCallEvent)]
        result_events = [e for e in collected if isinstance(e, ToolResultEvent)]

        assert len(call_events) == 1
        assert call_events[0].tool_name == "dummy_read"
        assert call_events[0].arguments == {"path": "test.txt"}

        assert len(result_events) == 1
        assert result_events[0].tool_name == "dummy_read"
        assert result_events[0].success is True
        assert result_events[0].duration_ms >= 0

    def test_audit_events_on_tool_error(self, reset_tool_registry):
        """Tool errors emit ToolResultEvent with success=False."""
        from tsugite.core.agent import TsugiteAgent
        from tsugite.core.tools import Tool
        from tsugite.events import EventBus

        collected = []

        def handler(event):
            collected.append(event)

        bus = EventBus()
        bus.subscribe(handler)

        async def failing_tool() -> str:
            raise RuntimeError("boom")

        tools = [
            Tool(
                name="failing",
                description="Always fails",
                parameters={},
                function=failing_tool,
            )
        ]

        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=tools,
            event_bus=bus,
        )

        wrapper = agent.executor.namespace.get("failing")
        with pytest.raises(RuntimeError, match="boom"):
            wrapper()

        result_events = [e for e in collected if isinstance(e, ToolResultEvent)]
        assert len(result_events) == 1
        assert result_events[0].success is False
        assert "boom" in result_events[0].result_summary


# ============================================================================
# JSONL handler for audit events
# ============================================================================


class TestJSONLAuditEvents:
    def test_tool_call_event_emitted_as_jsonl(self, capsys):
        from tsugite.ui.jsonl import JSONLUIHandler

        handler = JSONLUIHandler()
        event = ToolCallEvent(tool_name="read_file", arguments={"path": "/tmp"}, step=2)
        handler.handle_event(event)

        output = capsys.readouterr().out.strip()
        data = json.loads(output)
        assert data["type"] == "tool_call"
        assert data["tool"] == "read_file"
        assert data["arguments"] == {"path": "/tmp"}

    def test_tool_result_event_emitted_as_jsonl(self, capsys):
        from tsugite.ui.jsonl import JSONLUIHandler

        handler = JSONLUIHandler()
        event = ToolResultEvent(tool_name="run", success=True, duration_ms=55, result_summary="ok")
        handler.handle_event(event)

        output = capsys.readouterr().out.strip()
        data = json.loads(output)
        assert data["type"] == "tool_result_audit"
        assert data["tool"] == "run"
        assert data["success"] is True
        assert data["duration_ms"] == 55


# ============================================================================
# run_safe removal
# ============================================================================


class TestRunSafeRemoved:
    def test_no_run_safe_in_registry(self):
        """run_safe should not exist in the tool registry."""
        from tsugite.tools import _ensure_tools_loaded

        _ensure_tools_loaded()
        assert "run_safe" not in list_tools()

    def test_run_has_no_blocklist_check(self, reset_tool_registry):
        """run() should not check for dangerous patterns."""
        from tsugite.tools.shell import run

        # These patterns were previously blocked
        # Now they should just try to execute (will fail naturally if command doesn't exist)
        # We just verify no ValueError is raised for the pattern check
        import subprocess

        # Mock execute_shell_command to avoid actual execution
        from unittest.mock import patch

        with patch("tsugite.tools.shell.execute_shell_command", return_value="ok"):
            result = run("echo rm -rf /", timeout=5)
            assert result == "ok"


# ============================================================================
# spawn_agent auto-injection removed
# ============================================================================


class TestSpawnAgentAutoInjection:
    @pytest.fixture(autouse=True)
    def _reload_tools(self):
        """Force-reload tool modules so the full registry is available."""
        import importlib

        import tsugite.tools as tools_mod
        import tsugite.tools.agents as agents_mod
        import tsugite.tools.fs as fs_mod
        import tsugite.tools.interactive as interactive_mod
        import tsugite.tools.shell as shell_mod

        tools_mod._tools_loaded = False
        importlib.reload(fs_mod)
        importlib.reload(shell_mod)
        importlib.reload(agents_mod)
        importlib.reload(interactive_mod)
        tools_mod._tools_loaded = True

    def test_spawn_agent_not_auto_injected(self, tmp_path, monkeypatch):
        """Agent without spawn_agent in tools list should not have it added."""
        monkeypatch.chdir(tmp_path)

        from tsugite.agent_preparation import AgentPreparer
        from tsugite.md_agents import parse_agent

        agent_content = """---
name: no_spawn
extends: none
model: openai:gpt-4o-mini
tools: [read_file, write_file]
---

# Agent
{{ user_prompt }}
"""
        agent_file = tmp_path / "no_spawn.md"
        agent_file.write_text(agent_content)

        agent = parse_agent(agent_content, str(agent_file))
        preparer = AgentPreparer()
        prepared = preparer.prepare(agent, "test task")

        tool_names = [t.name for t in prepared.tools]
        assert "spawn_agent" not in tool_names

    def test_spawn_agent_included_when_explicit(self, tmp_path, monkeypatch):
        """Agent with spawn_agent explicitly in tools should have it."""
        monkeypatch.chdir(tmp_path)

        from tsugite.agent_preparation import AgentPreparer
        from tsugite.md_agents import parse_agent

        agent_content = """---
name: with_spawn
extends: none
model: openai:gpt-4o-mini
tools: [read_file, spawn_agent]
---

# Agent
{{ user_prompt }}
"""
        agent_file = tmp_path / "with_spawn.md"
        agent_file.write_text(agent_content)

        agent = parse_agent(agent_content, str(agent_file))
        preparer = AgentPreparer()
        prepared = preparer.prepare(agent, "test task")

        tool_names = [t.name for t in prepared.tools]
        assert "spawn_agent" in tool_names
