"""Tests for JSONL UI handler."""

import json

from tsugite.ui.base import UIEvent
from tsugite.ui.jsonl import JSONLUIHandler


def test_jsonl_task_start(capsys):
    """Test TASK_START event emits init JSONL."""
    handler = JSONLUIHandler()
    handler.handle_event(UIEvent.TASK_START, {"agent": "test_agent", "model": "gpt-4"})

    output = capsys.readouterr().out
    lines = output.strip().split("\n")

    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["type"] == "init"
    assert event["agent"] == "test_agent"
    assert event["model"] == "gpt-4"


def test_jsonl_step_start(capsys):
    """Test STEP_START event emits turn_start JSONL."""
    handler = JSONLUIHandler()
    handler.handle_event(UIEvent.STEP_START, {"step": 1})

    output = capsys.readouterr().out
    lines = output.strip().split("\n")

    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["type"] == "turn_start"
    assert event["turn"] == 1


def test_jsonl_thought(capsys):
    """Test LLM_MESSAGE event emits thought JSONL."""
    handler = JSONLUIHandler()
    handler.handle_event(UIEvent.LLM_MESSAGE, {"content": "Thinking about the problem..."})

    output = capsys.readouterr().out
    lines = output.strip().split("\n")

    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["type"] == "thought"
    assert event["content"] == "Thinking about the problem..."


def test_jsonl_code_execution(capsys):
    """Test CODE_EXECUTION event emits code JSONL."""
    handler = JSONLUIHandler()
    handler.handle_event(UIEvent.CODE_EXECUTION, {"code": "print('hello')"})

    output = capsys.readouterr().out
    lines = output.strip().split("\n")

    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["type"] == "code"
    assert event["content"] == "print('hello')"


def test_jsonl_tool_call(capsys):
    """Test TOOL_CALL event emits tool_call JSONL."""
    handler = JSONLUIHandler()
    handler.handle_event(UIEvent.TOOL_CALL, {"tool": "read_file", "args": {"path": "test.txt"}})

    output = capsys.readouterr().out
    lines = output.strip().split("\n")

    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["type"] == "tool_call"
    assert event["tool"] == "read_file"
    assert event["args"] == {"path": "test.txt"}


def test_jsonl_observation_success(capsys):
    """Test OBSERVATION event with success emits tool_result JSONL."""
    handler = JSONLUIHandler()
    handler.handle_event(UIEvent.OBSERVATION, {"tool": "read_file", "observation": "file contents"})

    output = capsys.readouterr().out
    lines = output.strip().split("\n")

    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["type"] == "tool_result"
    assert event["tool"] == "read_file"
    assert event["success"] is True
    assert event["output"] == "file contents"


def test_jsonl_observation_error(capsys):
    """Test OBSERVATION event with error emits tool_result JSONL."""
    handler = JSONLUIHandler()
    handler.handle_event(UIEvent.OBSERVATION, {"tool": "read_file", "error": "File not found"})

    output = capsys.readouterr().out
    lines = output.strip().split("\n")

    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["type"] == "tool_result"
    assert event["tool"] == "read_file"
    assert event["success"] is False
    assert event["error"] == "File not found"


def test_jsonl_final_answer(capsys):
    """Test FINAL_ANSWER event emits final_result JSONL."""
    handler = JSONLUIHandler()
    handler.handle_event(UIEvent.FINAL_ANSWER, {"answer": "The result", "turns": 3, "tokens": 150, "cost": 0.002})

    output = capsys.readouterr().out
    lines = output.strip().split("\n")

    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["type"] == "final_result"
    assert event["result"] == "The result"
    assert event["turns"] == 3
    assert event["tokens"] == 150
    assert event["cost"] == 0.002


def test_jsonl_error(capsys):
    """Test ERROR event emits error JSONL."""
    handler = JSONLUIHandler()
    handler.handle_event(UIEvent.ERROR, {"error": "Something went wrong", "step": 2})

    output = capsys.readouterr().out
    lines = output.strip().split("\n")

    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["type"] == "error"
    assert event["error"] == "Something went wrong"
    assert event["step"] == 2


def test_jsonl_multiple_events(capsys):
    """Test multiple events produce multiple JSONL lines."""
    handler = JSONLUIHandler()

    handler.handle_event(UIEvent.TASK_START, {"agent": "test", "model": "gpt-4"})
    handler.handle_event(UIEvent.STEP_START, {"step": 1})
    handler.handle_event(UIEvent.LLM_MESSAGE, {"content": "Thinking..."})
    handler.handle_event(UIEvent.TOOL_CALL, {"tool": "read_file", "args": {}})
    handler.handle_event(UIEvent.FINAL_ANSWER, {"answer": "Done"})

    output = capsys.readouterr().out
    lines = output.strip().split("\n")

    assert len(lines) == 5
    assert json.loads(lines[0])["type"] == "init"
    assert json.loads(lines[1])["type"] == "turn_start"
    assert json.loads(lines[2])["type"] == "thought"
    assert json.loads(lines[3])["type"] == "tool_call"
    assert json.loads(lines[4])["type"] == "final_result"
