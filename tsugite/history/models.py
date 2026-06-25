"""Per-event session storage model.

The session JSONL is an append-only log of `Event` records. Each event has an
open `data` payload — adding a new event type means choosing a new string for
`type`; no model changes required. Schema drift can't silently drop records
because Event accepts any type string.

Standard event types (documented for reference; runtime treats `type` as opaque):

- session_start: agent, model, workspace, parent_session
- user_input: text, attachments
- model_request: provider, model, message_count, tool_names
- model_response: provider, raw_content, usage, cost, stop_reason, state_delta
- code_execution: code, output, error, duration_ms, tools_called
- tool_invocation: name, args, output, error, duration_ms, call_id
- format_error: reason, rejected_content
- skill_added / skill_removed: name
- hook_execution: phase, command, exit_code, stdout, stderr, duration_ms
- compaction: summary, replaced_count, retained_count, reason, range_start, range_end, source_session_id
- compacted_into: new_session_id, reason, replaced_count, retained_count
- session_end: status (success|error|interrupted), error_message
"""

from datetime import datetime, timezone
from typing import Annotated, Any, Dict, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def _parse_iso(v):
    if isinstance(v, str):
        return datetime.fromisoformat(v.replace("Z", "+00:00"))
    return v


def iso_utc(dt: Optional[datetime] = None) -> str:
    """Serialize a datetime as a fixed-precision UTC ISO-8601 string for storage.

    Always microsecond precision and UTC so lexicographic comparison equals
    chronological order (sqlite ORDER BY on these columns relies on it). A naive
    datetime is assumed to be UTC; an aware one is converted.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat(timespec="microseconds")


def dedup_model_request_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Drop a model_request's full ``messages`` array, keeping just its count.

    Reconstruction rebuilds the prompt from the other events, so the array is redundant
    on disk. Shared by new writes and by the legacy importer so both produce the same shape.
    """
    if "messages" not in data:
        return data
    out = {k: v for k, v in data.items() if k != "messages"}
    out["message_count"] = len(data["messages"])
    return out


ISODatetime = Annotated[datetime, BeforeValidator(_parse_iso)]


class Event(BaseModel):
    """One record in a session JSONL file."""

    model_config = ConfigDict(extra="allow")

    id: Optional[int] = Field(default=None, description="Storage rowid, populated on read; never written")
    type: str = Field(..., description="Event type string")
    ts: ISODatetime = Field(..., description="When the event occurred")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event-specific payload")
