"""Per-event session storage model.

The session JSONL is an append-only log of `Event` records. Each event has an
open `data` payload — adding a new event type means choosing a new string for
`type`; no model changes required. Schema drift can't silently drop records
because Event accepts any type string.

Standard event types (documented for reference; runtime treats `type` as opaque):

- session_start: agent, model, workspace, parent_session
- user_input: text, attachments
- model_request: provider, model, message_count, tool_names
- model_response: provider, raw_content, thought, content_blocks, tail, usage, cost, stop_reason, state_delta
  (thought/content_blocks/tail = the settled parse; thought present marks a parsed event,
  older events without it are normalized on read)
- code_execution: code, output, error, duration_ms, tools_called, tool_calls
- reasoning: content (the turn's reasoning/thinking summary; never replayed into model context)
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


class SessionSummary:
    """Aggregates derived from an event log: agent, model, totals, status."""

    def __init__(self):
        self.agent: Optional[str] = None
        self.model: Optional[str] = None
        self.workspace: Optional[str] = None
        self.created_at: Optional[datetime] = None
        self.parent_session: Optional[str] = None
        self.status: Optional[str] = None
        self.error_message: Optional[str] = None
        self.turn_count: int = 0  # number of user_input events
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        self.total_duration_ms: int = 0
        self.functions_called: set[str] = set()
        self.last_response_text: str = ""

    @classmethod
    def from_events(cls, events: Iterable[Event]) -> "SessionSummary":
        s = cls()
        for event in events:
            data = event.data
            if event.type == "session_start":
                s.agent = data.get("agent")
                s.model = data.get("model")
                s.workspace = data.get("workspace")
                s.created_at = event.ts
                s.parent_session = data.get("parent_session")
            elif event.type == "user_input":
                s.turn_count += 1
            elif event.type == "model_response":
                usage = data.get("usage") or {}
                if isinstance(usage, dict):
                    s.total_tokens += int(usage.get("total_tokens") or 0)
                cost = data.get("cost")
                if cost:
                    s.total_cost += float(cost)
                s.last_response_text = data.get("raw_content", s.last_response_text)
            elif event.type == "code_execution":
                s.total_duration_ms += int(data.get("duration_ms") or 0)
                for fn in data.get("tools_called") or []:
                    s.functions_called.add(fn)
            elif event.type == "tool_invocation":
                name = data.get("name")
                if name:
                    s.functions_called.add(name)
                s.total_duration_ms += int(data.get("duration_ms") or 0)
            elif event.type == "session_end":
                s.status = data.get("status")
                s.error_message = data.get("error_message")
        return s
