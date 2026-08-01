"""JSONL UI handler for subprocess-based subagent communication."""

import json
from typing import Any, Dict

from tsugite.events import (
    CodeExecutionEvent,
    ContentBlockEvent,
    ErrorEvent,
    FileReadEvent,
    FileWriteEvent,
    FinalAnswerEvent,
    InfoEvent,
    LLMMessageEvent,
    LLMWaitProgressEvent,
    ModelResponseEvent,
    ObservationEvent,
    PromptSnapshotEvent,
    ReasoningContentEvent,
    ReasoningTokensEvent,
    SecretAccessEvent,
    SkillLoadedEvent,
    SkillLoadFailedEvent,
    SkillUnloadedEvent,
    StepStartEvent,
    StreamChunkEvent,
    StreamCompleteEvent,
    TaskStartEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from tsugite.ui.dispatch import EventDispatchMixin, handles

# Event types that share the same file-io payload shape
_FILE_IO_EVENTS = {FileReadEvent: "file_read", FileWriteEvent: "file_write"}


class JSONLUIHandler(EventDispatchMixin):
    """Emit UI events as JSONL to stdout for subprocess communication.

    This handler converts all UI events to line-delimited JSON objects,
    enabling parent agents to monitor subagent progress in real-time.

    JSONL Protocol Schema:
    ----------------------
    Each line is a JSON object with a "type" field and type-specific data.

    Event Type Mappings:
    - TaskStartEvent      → {"type": "init", "agent": str, "model": str}
    - StepStartEvent      → {"type": "turn_start", "turn": int}
    - LLMMessageEvent     → {"type": "thought", "content": str}
    - StreamChunkEvent    → {"type": "stream_chunk", "chunk": str}
    - StreamCompleteEvent → {"type": "stream_complete"}
    - CodeExecutionEvent  → {"type": "code", "content": str}
    - ObservationEvent    → {"type": "tool_result", "tool": str, "success": bool, "output"?: str, "error"?: str}
    - FinalAnswerEvent    → {"type": "final_result", "result": str, "result_data": Any|null, "turns": int, "tokens": int, "cost": float}
    - ErrorEvent          → {"type": "error", "error": str, "step": int}
    - FileReadEvent       → {"type": "file_read", "path": str, "line_count": int, "byte_count": int, "operation": str}
    - FileWriteEvent      → {"type": "file_write", "path": str, "line_count": int, "byte_count": int, "operation": str}
    - SkillLoadedEvent    → {"type": "skill_loaded", "name": str, "description": str}
    - SkillLoadFailedEvent→ {"type": "warning", "message": "Failed to load skill '{name}': {error}"}
    - SkillUnloadedEvent  → {"type": "skill_unloaded", "name": str}
    - SecretAccessEvent   → {"type": "secret_access", "name": str}

    Success/Failure Patterns:
    - Successful tool: {"type": "tool_result", "tool": "read_file", "success": true, "output": "..."}
    - Failed tool: {"type": "tool_result", "tool": "read_file", "success": false, "error": "..."}
    """

    @handles(TaskStartEvent)
    def _handle_task_start(self, event: TaskStartEvent) -> None:
        self._emit("init", {"agent": event.task, "model": event.model})

    @handles(StepStartEvent)
    def _handle_step_start(self, event: StepStartEvent) -> None:
        self._emit("turn_start", {"turn": event.step})

    @handles(LLMMessageEvent)
    def _handle_llm_message(self, event: LLMMessageEvent) -> None:
        if event.content.strip():
            self._emit("thought", {"content": event.content})

    @handles(StreamChunkEvent)
    def _handle_stream_chunk(self, event: StreamChunkEvent) -> None:
        self._emit("stream_chunk", {"chunk": event.chunk})

    @handles(StreamCompleteEvent)
    def _handle_stream_complete(self, event: StreamCompleteEvent) -> None:
        self._emit("stream_complete", {})

    @handles(CodeExecutionEvent)
    def _handle_code_execution(self, event: CodeExecutionEvent) -> None:
        if event.code:
            self._emit("code", {"content": event.code})

    @handles(ObservationEvent)
    def _handle_observation(self, event: ObservationEvent) -> None:
        if event.success:
            self._emit("tool_result", {"tool": event.tool or "unknown", "success": True, "output": event.observation})
        else:
            self._emit(
                "tool_result",
                {"tool": event.tool or "unknown", "success": False, "error": event.error or event.observation},
            )

    @handles(ContentBlockEvent)
    def _handle_content_block(self, event: ContentBlockEvent) -> None:
        self._emit("content_block", {"name": event.name, "content": event.content})

    @handles(ModelResponseEvent)
    def _handle_model_response(self, event: ModelResponseEvent) -> None:
        # The settled parse of one model turn; shares its type and field names
        # with the persisted model_response event so the timeline reducer
        # renders live frames and replayed history through one code path.
        self._emit(
            "model_response",
            {
                "thought": event.thought,
                "content_blocks": event.content_blocks,
                "tail": event.tail,
                "usage": event.usage,
            },
        )

    @handles(FinalAnswerEvent)
    def _handle_final_answer(self, event: FinalAnswerEvent) -> None:
        self._emit(
            "final_result",
            {
                "result": event.answer,
                "result_data": event.answer_data,
                "turns": event.turns,
                "tokens": event.tokens,
                "cost": event.cost,
            },
        )

    @handles(ErrorEvent)
    def _handle_error(self, event: ErrorEvent) -> None:
        self._emit("error", {"error": event.error, "step": event.step})

    @handles(SkillLoadedEvent)
    def _handle_skill_loaded(self, event: SkillLoadedEvent) -> None:
        payload = {"name": event.skill_name, "description": event.description or ""}
        if event.session_id:
            payload["session_id"] = event.session_id
        self._emit("skill_loaded", payload)

    @handles(SkillLoadFailedEvent)
    def _handle_skill_load_failed(self, event: SkillLoadFailedEvent) -> None:
        self._emit("warning", {"message": f"Failed to load skill '{event.skill_name}': {event.error_message}"})

    @handles(SkillUnloadedEvent)
    def _handle_skill_unloaded(self, event: SkillUnloadedEvent) -> None:
        payload = {"name": event.skill_name}
        if event.session_id:
            payload["session_id"] = event.session_id
        self._emit("skill_unloaded", payload)

    @handles(FileReadEvent, FileWriteEvent)
    def _handle_file_io(self, event) -> None:
        self._emit(
            _FILE_IO_EVENTS[type(event)],
            {
                "path": event.path,
                "line_count": event.line_count,
                "byte_count": event.byte_count,
                "operation": event.operation,
            },
        )

    @handles(SecretAccessEvent)
    def _handle_secret_access(self, event: SecretAccessEvent) -> None:
        self._emit("secret_access", {"name": event.name})

    @handles(ToolCallEvent)
    def _handle_tool_call(self, event: ToolCallEvent) -> None:
        self._emit("tool_call", {"tool": event.tool_name, "arguments": event.arguments, "step": event.step})

    @handles(ToolResultEvent)
    def _handle_tool_result(self, event: ToolResultEvent) -> None:
        self._emit(
            "tool_result_audit",
            {
                "tool": event.tool_name,
                "success": event.success,
                "duration_ms": event.duration_ms,
                "summary": event.result_summary,
                "step": event.step,
            },
        )

    @handles(InfoEvent)
    def _handle_info(self, event: InfoEvent) -> None:
        self._emit("info", {"message": event.message})

    @handles(PromptSnapshotEvent)
    def _handle_prompt_snapshot(self, event: PromptSnapshotEvent) -> None:
        self._emit("prompt_snapshot", {"token_breakdown": event.token_breakdown})

    @handles(ReasoningContentEvent)
    def _handle_reasoning_content(self, event: ReasoningContentEvent) -> None:
        self._emit("reasoning_content", {"content": event.content, "step": event.step})

    @handles(ReasoningTokensEvent)
    def _handle_reasoning_tokens(self, event: ReasoningTokensEvent) -> None:
        self._emit("reasoning_tokens", {"tokens": event.tokens, "step": event.step})

    @handles(LLMWaitProgressEvent)
    def _handle_llm_wait_progress(self, event: LLMWaitProgressEvent) -> None:
        self._emit("llm_wait_progress", {"elapsed_seconds": event.elapsed_seconds})

    def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Print JSONL event to stdout.

        Args:
            event_type: The event type string
            data: Event-specific data dictionary
        """
        event = {"type": event_type, **data}
        print(json.dumps(event), flush=True)

    def update_progress(self, description: str) -> None:
        """No-op for progress updates in JSONL mode."""
        pass

    def progress_context(self):
        """No-op context manager for compatibility."""
        from contextlib import nullcontext

        return nullcontext()
