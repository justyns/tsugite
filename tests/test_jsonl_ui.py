"""Tests for JSONL UI handler."""

import json

from tsugite.events import (
    CodeExecutionEvent,
    ErrorEvent,
    FinalAnswerEvent,
    LLMMessageEvent,
    LLMWaitProgressEvent,
    ObservationEvent,
    StepStartEvent,
    TaskStartEvent,
)
from tsugite.ui.jsonl import JSONLUIHandler


def test_jsonl_task_start(capsys):
    """Test TASK_START event emits init JSONL."""
    handler = JSONLUIHandler()
    event = TaskStartEvent(task="test_agent", model="gpt-4")
    handler.handle_event(event)

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
    event = StepStartEvent(step=1)
    handler.handle_event(event)

    output = capsys.readouterr().out
    lines = output.strip().split("\n")

    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["type"] == "turn_start"
    assert event["turn"] == 1


def test_jsonl_thought(capsys):
    """Test LLM_MESSAGE event emits thought JSONL."""
    handler = JSONLUIHandler()
    event = LLMMessageEvent(content="Thinking about the problem...")
    handler.handle_event(event)

    output = capsys.readouterr().out
    lines = output.strip().split("\n")

    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["type"] == "thought"
    assert event["content"] == "Thinking about the problem..."


def test_jsonl_code_execution(capsys):
    """Test CODE_EXECUTION event emits code JSONL."""
    handler = JSONLUIHandler()
    event = CodeExecutionEvent(code="print('hello')")
    handler.handle_event(event)

    output = capsys.readouterr().out
    lines = output.strip().split("\n")

    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["type"] == "code"
    assert event["content"] == "print('hello')"


def test_jsonl_observation_success(capsys):
    """Test OBSERVATION event with success emits tool_result JSONL."""
    handler = JSONLUIHandler()
    event = ObservationEvent(tool="read_file", observation="file contents")
    handler.handle_event(event)

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
    event = ObservationEvent(tool="read_file", observation="File not found", success=False)
    handler.handle_event(event)

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
    event = FinalAnswerEvent(answer="The result", turns=3, tokens=150, cost=0.002)
    handler.handle_event(event)

    output = capsys.readouterr().out
    lines = output.strip().split("\n")

    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["type"] == "final_result"
    assert event["result"] == "The result"
    assert event["turns"] == 3
    assert event["tokens"] == 150
    assert event["cost"] == 0.002
    assert event["result_data"] is None


def test_jsonl_final_answer_with_structured_data(capsys):
    handler = JSONLUIHandler()
    handler.handle_event(
        FinalAnswerEvent(
            answer="{'status': 'ok'}",
            answer_data={"status": "ok", "items": [1, 2]},
            turns=1,
            tokens=10,
            cost=0.001,
        )
    )
    ev = json.loads(capsys.readouterr().out.strip().split("\n")[0])
    assert ev["type"] == "final_result"
    assert ev["result_data"] == {"status": "ok", "items": [1, 2]}
    assert ev["result"] == "{'status': 'ok'}"


def test_jsonl_final_answer_string_no_data(capsys):
    handler = JSONLUIHandler()
    handler.handle_event(FinalAnswerEvent(answer="plain string"))
    ev = json.loads(capsys.readouterr().out.strip().split("\n")[0])
    assert ev["result_data"] is None
    assert ev["result"] == "plain string"


def test_jsonl_error(capsys):
    """Test ERROR event emits error JSONL."""
    handler = JSONLUIHandler()
    event = ErrorEvent(error="Something went wrong", step=2)
    handler.handle_event(event)

    output = capsys.readouterr().out
    lines = output.strip().split("\n")

    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["type"] == "error"
    assert event["error"] == "Something went wrong"
    assert event["step"] == 2


def test_jsonl_llm_wait_progress(capsys):
    """LLM_WAIT_PROGRESS heartbeat surfaces as llm_wait_progress JSONL with elapsed_seconds."""
    handler = JSONLUIHandler()
    handler.handle_event(LLMWaitProgressEvent(elapsed_seconds=42))

    output = capsys.readouterr().out
    lines = output.strip().split("\n")

    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["type"] == "llm_wait_progress"
    assert event["elapsed_seconds"] == 42


def test_jsonl_multiple_events(capsys):
    """Test multiple events produce multiple JSONL lines."""
    handler = JSONLUIHandler()

    handler.handle_event(TaskStartEvent(task="test", model="gpt-4"))
    handler.handle_event(StepStartEvent(step=1))
    handler.handle_event(LLMMessageEvent(content="Thinking..."))
    handler.handle_event(FinalAnswerEvent(answer="Done"))

    output = capsys.readouterr().out
    lines = output.strip().split("\n")

    assert len(lines) == 4
    assert json.loads(lines[0])["type"] == "init"
    assert json.loads(lines[1])["type"] == "turn_start"
    assert json.loads(lines[2])["type"] == "thought"
    assert json.loads(lines[3])["type"] == "final_result"
