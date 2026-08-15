"""Tests for JSONL UI handler."""

import json
from io import StringIO
from types import SimpleNamespace

from rich.console import Console

from tsugite.agent_runner.helpers import clear_multistep_ui_context, set_multistep_ui_context
from tsugite.events import (
    CodeExecutionEvent,
    ErrorEvent,
    FinalAnswerEvent,
    LLMMessageEvent,
    LLMWaitProgressEvent,
    ModelResponseEvent,
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


def test_jsonl_model_response(capsys):
    """MODEL_RESPONSE emits the settled parse with the same type and field
    names as the persisted history event (one reducer path for live+replay)."""
    handler = JSONLUIHandler()
    handler.handle_event(
        ModelResponseEvent(thought="pre-fence prose", content_blocks={"a.md": "body"}, tail="post-fence prose")
    )

    event = json.loads(capsys.readouterr().out.strip())
    assert event["type"] == "model_response"
    assert event["thought"] == "pre-fence prose"
    assert event["content_blocks"] == {"a.md": "body"}
    assert event["tail"] == "post-fence prose"


def test_jsonl_model_response_carries_usage(capsys):
    """The frame maps the turn's usage dump through unchanged, so the live path
    surfaces the same cache split (reads/writes) that replay reads off history."""
    handler = JSONLUIHandler()
    handler.handle_event(
        ModelResponseEvent(
            thought="x",
            usage={"total_tokens": 115, "cache_read_input_tokens": 80, "cache_creation_input_tokens": 20},
        )
    )

    event = json.loads(capsys.readouterr().out.strip())
    assert event["usage"] == {
        "total_tokens": 115,
        "cache_read_input_tokens": 80,
        "cache_creation_input_tokens": 20,
    }


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


def test_jsonl_stream_chunk_and_complete(capsys):
    """Streaming deltas map to stream_chunk/stream_complete frames so the web
    UI can render tokens as they arrive (the daemon SSE handlers subclass this)."""
    from tsugite.events import StreamChunkEvent, StreamCompleteEvent

    handler = JSONLUIHandler()
    handler.handle_event(StreamChunkEvent(chunk="Hel"))
    handler.handle_event(StreamChunkEvent(chunk="lo"))
    handler.handle_event(StreamCompleteEvent())

    lines = [json.loads(line) for line in capsys.readouterr().out.strip().split("\n")]
    assert [e["type"] for e in lines] == ["stream_chunk", "stream_chunk", "stream_complete"]
    assert [e.get("chunk") for e in lines[:2]] == ["Hel", "lo"]


def test_multistep_context_is_skipped_on_handlers_that_do_not_track_it():
    """The runner calls set/clear_multistep_context on whatever handler is wired up,
    and only the Rich console handler defines them.

    The bare `object()` is the case that matters: the built-in handlers share a base
    class, so testing those alone would also pass if the methods were added to that
    base, leaving the daemon's CompositeUIHandler and plugin handlers crashing.
    """
    from tsugite.ui.repl_handler import ReplUIHandler

    for handler in (JSONLUIHandler(), ReplUIHandler(Console(file=StringIO())), object()):
        custom_logger = SimpleNamespace(ui_handler=handler)
        set_multistep_ui_context(custom_logger, 1, "research", 3)
        clear_multistep_ui_context(custom_logger)


def test_multistep_context_still_reaches_the_handler_that_tracks_it():
    """The guard must skip only what is absent - the console handler's real
    implementation has to keep being called."""
    from tsugite.ui.base import CustomUIHandler

    handler = CustomUIHandler(Console(file=StringIO()))
    custom_logger = SimpleNamespace(ui_handler=handler)

    set_multistep_ui_context(custom_logger, 2, "research", 4)
    assert handler.state.multistep_context == {"step_number": 2, "step_name": "research", "total_steps": 4}

    clear_multistep_ui_context(custom_logger)
    assert handler.state.multistep_context is None
