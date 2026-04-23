"""Per-event session storage model.

The session JSONL is an append-only log of `Event` records. Each event has an
open `data` payload — adding a new event type means choosing a new string for
`type`; no model changes required. Schema drift can't silently drop records
because Event accepts any type string.

Standard event types (documented for reference; runtime treats `type` as opaque):

- session_start: agent, model, machine, workspace, parent_session
- user_input: text, attachments
- model_request: provider, payload (messages|input string), tool_names, model
- model_response: provider, raw_content, usage, cost, stop_reason, state_delta
- code_execution: code, output, error, duration_ms, tools_called
- tool_invocation: name, args, output, error, duration_ms, call_id
- format_error: reason, rejected_content
- attachment_added / attachment_removed: name, sha256, type, source
- skill_added / skill_removed: name
- hook_execution: phase, command, exit_code, stdout, stderr, duration_ms
- compaction: summary, replaced_count, retained_count, reason
- session_end: status (success|error|interrupted), error_message
"""

from datetime import datetime
from typing import Annotated, Any, Dict

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def _parse_iso(v):
    if isinstance(v, str):
        return datetime.fromisoformat(v.replace("Z", "+00:00"))
    return v


ISODatetime = Annotated[datetime, BeforeValidator(_parse_iso)]


class Event(BaseModel):
    """One record in a session JSONL file."""

    model_config = ConfigDict(extra="allow")

    type: str = Field(..., description="Event type string")
    ts: ISODatetime = Field(..., description="When the event occurred")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event-specific payload")
