"""JSONL UI handler for subprocess-based subagent communication."""

import json
from typing import Any, Dict

from tsugite.ui.base import UIEvent


class JSONLUIHandler:
    """Emit UI events as JSONL to stdout for subprocess communication.

    This handler converts all UI events to line-delimited JSON objects,
    enabling parent agents to monitor subagent progress in real-time.
    """

    def handle_event(self, event: UIEvent, data: Dict[str, Any]) -> None:
        """Convert UI event to JSONL and print to stdout.

        Args:
            event: The UI event type
            data: Event-specific data dictionary
        """
        if event == UIEvent.TASK_START:
            self._emit("init", {"agent": data.get("agent"), "model": data.get("model")})

        elif event == UIEvent.STEP_START:
            self._emit("turn_start", {"turn": data.get("step")})

        elif event == UIEvent.LLM_MESSAGE:
            content = data.get("content", "")
            if content.strip():
                self._emit("thought", {"content": content})

        elif event == UIEvent.CODE_EXECUTION:
            code = data.get("code", "")
            if code:
                self._emit("code", {"content": code})

        elif event == UIEvent.TOOL_CALL:
            self._emit("tool_call", {"tool": data.get("tool", "unknown"), "args": data.get("args", {})})

        elif event == UIEvent.OBSERVATION:
            observation = data.get("observation", "")
            error = data.get("error")

            if error:
                self._emit("tool_result", {"tool": data.get("tool", "unknown"), "success": False, "error": error})
            else:
                self._emit("tool_result", {"tool": data.get("tool", "unknown"), "success": True, "output": observation})

        elif event == UIEvent.EXECUTION_RESULT:
            content = data.get("content", "")
            error = data.get("error")

            if error:
                self._emit("tool_result", {"tool": "code_execution", "success": False, "error": error})
            else:
                self._emit("tool_result", {"tool": "code_execution", "success": True, "output": content})

        elif event == UIEvent.FINAL_ANSWER:
            self._emit(
                "final_result",
                {
                    "result": data.get("answer", ""),
                    "turns": data.get("turns"),
                    "tokens": data.get("tokens"),
                    "cost": data.get("cost"),
                },
            )

        elif event == UIEvent.ERROR:
            self._emit("error", {"error": data.get("error", ""), "step": data.get("step")})

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
