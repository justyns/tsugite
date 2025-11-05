"""JSONL UI handler for subprocess-based subagent communication."""

import json
from typing import Any, Dict

from tsugite.events import (
    BaseEvent,
    CodeExecutionEvent,
    ErrorEvent,
    ExecutionResultEvent,
    FinalAnswerEvent,
    LLMMessageEvent,
    ObservationEvent,
    SkillLoadedEvent,
    SkillUnloadedEvent,
    StepStartEvent,
    TaskStartEvent,
    ToolCallEvent,
)


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
    - ToolCallEvent       → {"type": "tool_call", "tool": str, "args": dict}
    - ObservationEvent    → {"type": "tool_result", "tool": str, "success": bool, "output"?: str, "error"?: str}
    - ExecutionResultEvent→ {"type": "tool_result", "tool": "code_execution", "success": bool, "output"?: str, "error"?: str}
    - FinalAnswerEvent    → {"type": "final_result", "result": str, "turns": int, "tokens": int, "cost": float}
    - ErrorEvent          → {"type": "error", "error": str, "step": int}
    - SkillLoadedEvent    → {"type": "info", "message": "Loaded skill: {name} ({description})"}
    - SkillUnloadedEvent  → {"type": "info", "message": "Unloaded skill: {name}"}

    Success/Failure Patterns:
    - Successful tool: {"type": "tool_result", "tool": "read_file", "success": true, "output": "..."}
    - Failed tool: {"type": "tool_result", "tool": "read_file", "success": false, "error": "..."}
    - Code execution success: {"type": "tool_result", "tool": "code_execution", "success": true, "output": "logs\\nresult"}
    - Code execution failure: {"type": "tool_result", "tool": "code_execution", "success": false, "error": "..."}
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

        elif isinstance(event, ToolCallEvent):
            self._emit("tool_call", {"tool": event.tool, "args": event.args})

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

        elif isinstance(event, ExecutionResultEvent):
            if event.success:
                # Combine logs and output for backward compatibility
                output = event.output or ""
                if event.logs:
                    logs_str = "\n".join(event.logs)
                    output = f"{logs_str}\n{output}" if output else logs_str
                self._emit("tool_result", {"tool": "code_execution", "success": True, "output": output})
            else:
                self._emit("tool_result", {"tool": "code_execution", "success": False, "error": event.error or ""})

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
            info_msg = f"Loaded skill: {event.skill_name}"
            if event.description:
                info_msg += f" ({event.description})"
            self._emit("info", {"message": info_msg})

        elif isinstance(event, SkillUnloadedEvent):
            self._emit("info", {"message": f"Unloaded skill: {event.skill_name}"})

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
