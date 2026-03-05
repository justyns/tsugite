"""JSONL UI handler for subprocess-based subagent communication."""

import json
from typing import Any, Dict

from tsugite.events import (
    BaseEvent,
    CodeExecutionEvent,
    ErrorEvent,
    FileReadEvent,
    FileWriteEvent,
    FinalAnswerEvent,
    InfoEvent,
    LLMMessageEvent,
    ObservationEvent,
    SkillLoadedEvent,
    SkillLoadFailedEvent,
    SkillUnloadedEvent,
    StepStartEvent,
    TaskStartEvent,
    ToolCallEvent,
    ToolResultEvent,
)

# Event types that share the same file-io payload shape
_FILE_IO_EVENTS = {FileReadEvent: "file_read", FileWriteEvent: "file_write"}


class JSONLUIHandler:
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
    - CodeExecutionEvent  → {"type": "code", "content": str}
    - ObservationEvent    → {"type": "tool_result", "tool": str, "success": bool, "output"?: str, "error"?: str}
    - FinalAnswerEvent    → {"type": "final_result", "result": str, "turns": int, "tokens": int, "cost": float}
    - ErrorEvent          → {"type": "error", "error": str, "step": int}
    - FileReadEvent       → {"type": "file_read", "path": str, "line_count": int, "byte_count": int, "operation": str}
    - FileWriteEvent      → {"type": "file_write", "path": str, "line_count": int, "byte_count": int, "operation": str}
    - SkillLoadedEvent    → {"type": "skill_loaded", "name": str, "description": str}
    - SkillLoadFailedEvent→ {"type": "warning", "message": "Failed to load skill '{name}': {error}"}
    - SkillUnloadedEvent  → {"type": "skill_unloaded", "name": str}

    Success/Failure Patterns:
    - Successful tool: {"type": "tool_result", "tool": "read_file", "success": true, "output": "..."}
    - Failed tool: {"type": "tool_result", "tool": "read_file", "success": false, "error": "..."}
    """

    _DISPATCH: dict[type, str] = {
        TaskStartEvent: "_handle_task_start",
        StepStartEvent: "_handle_step_start",
        LLMMessageEvent: "_handle_llm_message",
        CodeExecutionEvent: "_handle_code_execution",
        ObservationEvent: "_handle_observation",
        FinalAnswerEvent: "_handle_final_answer",
        ErrorEvent: "_handle_error",
        SkillLoadedEvent: "_handle_skill_loaded",
        SkillLoadFailedEvent: "_handle_skill_load_failed",
        SkillUnloadedEvent: "_handle_skill_unloaded",
        FileReadEvent: "_handle_file_io",
        FileWriteEvent: "_handle_file_io",
        ToolCallEvent: "_handle_tool_call",
        ToolResultEvent: "_handle_tool_result",
        InfoEvent: "_handle_info",
    }

    def handle_event(self, event: BaseEvent) -> None:
        """Convert UI event to JSONL and print to stdout."""
        handler_name = self._DISPATCH.get(type(event))
        if handler_name:
            getattr(self, handler_name)(event)

    def _handle_task_start(self, event: TaskStartEvent) -> None:
        self._emit("init", {"agent": event.task, "model": event.model})

    def _handle_step_start(self, event: StepStartEvent) -> None:
        self._emit("turn_start", {"turn": event.step})

    def _handle_llm_message(self, event: LLMMessageEvent) -> None:
        if event.content.strip():
            self._emit("thought", {"content": event.content})

    def _handle_code_execution(self, event: CodeExecutionEvent) -> None:
        if event.code:
            self._emit("code", {"content": event.code})

    def _handle_observation(self, event: ObservationEvent) -> None:
        if event.success:
            self._emit("tool_result", {"tool": event.tool or "unknown", "success": True, "output": event.observation})
        else:
            self._emit(
                "tool_result",
                {"tool": event.tool or "unknown", "success": False, "error": event.error or event.observation},
            )

    def _handle_final_answer(self, event: FinalAnswerEvent) -> None:
        self._emit(
            "final_result",
            {"result": event.answer, "turns": event.turns, "tokens": event.tokens, "cost": event.cost},
        )

    def _handle_error(self, event: ErrorEvent) -> None:
        self._emit("error", {"error": event.error, "step": event.step})

    def _handle_skill_loaded(self, event: SkillLoadedEvent) -> None:
        self._emit("skill_loaded", {"name": event.skill_name, "description": event.description or ""})

    def _handle_skill_load_failed(self, event: SkillLoadFailedEvent) -> None:
        self._emit("warning", {"message": f"Failed to load skill '{event.skill_name}': {event.error_message}"})

    def _handle_skill_unloaded(self, event: SkillUnloadedEvent) -> None:
        self._emit("skill_unloaded", {"name": event.skill_name})

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

    def _handle_tool_call(self, event: ToolCallEvent) -> None:
        self._emit("tool_call", {"tool": event.tool_name, "arguments": event.arguments, "step": event.step})

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

    def _handle_info(self, event: InfoEvent) -> None:
        self._emit("info", {"message": event.message})

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
