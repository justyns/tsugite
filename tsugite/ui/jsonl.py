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

    def handle_event(self, event: BaseEvent) -> None:
        """Convert UI event to JSONL and print to stdout.

        Args:
            event: The Pydantic event
        """
        if isinstance(event, TaskStartEvent):
            self._emit("init", {"agent": event.task, "model": event.model})

        elif isinstance(event, StepStartEvent):
            self._emit("turn_start", {"turn": event.step})

        elif isinstance(event, LLMMessageEvent):
            if event.content.strip():
                self._emit("thought", {"content": event.content})

        elif isinstance(event, CodeExecutionEvent):
            if event.code:
                self._emit("code", {"content": event.code})

        elif isinstance(event, ObservationEvent):
            if event.success:
                self._emit(
                    "tool_result", {"tool": event.tool or "unknown", "success": True, "output": event.observation}
                )
            else:
                self._emit(
                    "tool_result",
                    {"tool": event.tool or "unknown", "success": False, "error": event.error or event.observation},
                )

        elif isinstance(event, FinalAnswerEvent):
            self._emit(
                "final_result",
                {
                    "result": event.answer,
                    "turns": event.turns,
                    "tokens": event.tokens,
                    "cost": event.cost,
                },
            )

        elif isinstance(event, ErrorEvent):
            self._emit("error", {"error": event.error, "step": event.step})

        elif isinstance(event, SkillLoadedEvent):
            self._emit("skill_loaded", {"name": event.skill_name, "description": event.description or ""})

        elif isinstance(event, SkillLoadFailedEvent):
            self._emit("warning", {"message": f"Failed to load skill '{event.skill_name}': {event.error_message}"})

        elif isinstance(event, SkillUnloadedEvent):
            self._emit("skill_unloaded", {"name": event.skill_name})

        elif type(event) in _FILE_IO_EVENTS:
            self._emit(
                _FILE_IO_EVENTS[type(event)],
                {
                    "path": event.path,
                    "line_count": event.line_count,
                    "byte_count": event.byte_count,
                    "operation": event.operation,
                },
            )

        elif isinstance(event, ToolCallEvent):
            self._emit("tool_call", {"tool": event.tool_name, "arguments": event.arguments, "step": event.step})

        elif isinstance(event, ToolResultEvent):
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

        elif isinstance(event, InfoEvent):
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
