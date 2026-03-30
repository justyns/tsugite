"""Pydantic models for session storage V2.

This module defines the data models for the turn-based session storage format.
Key features:
- Stores context once with hash references, tracks changes as deltas
- Stores full messages per turn for exact reconstruction
- Excludes system prompt (rebuild from code)
"""

from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

CompactionReason = Literal["token_threshold", "prompt_too_long", "scheduled", "manual"]


def _parse_iso_datetime(v):
    if v is None:
        return None
    if isinstance(v, str):
        return datetime.fromisoformat(v.replace("Z", "+00:00"))
    return v


ISODatetime = Annotated[datetime, BeforeValidator(_parse_iso_datetime)]


class AttachmentRef(BaseModel):
    """Reference to an attachment in context.

    For file-based attachments, content is cached by hash.
    For URL attachments, the URL is stored directly (provider fetches).
    """

    model_config = ConfigDict(extra="forbid")

    hash: Optional[str] = Field(default=None, description="SHA256 hash for cached content")
    url: Optional[str] = Field(default=None, description="URL for provider-fetched content")
    original_path: Optional[str] = Field(default=None, description="Original file path (fallback if cache cleared)")
    type: str = Field(..., description="Content type: text, image, audio, document")
    source: str = Field(..., description="Source type: file or url")
    mime_type: Optional[str] = Field(default=None, description="MIME type for correct data URI")


class SessionMeta(BaseModel):
    """Metadata for a session (first record in JSONL file)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    type: Literal["session_meta"] = "session_meta"
    workspace: Optional[str] = Field(default=None, description="Workspace name if applicable")
    agent: str = Field(..., description="Agent name used in this session")
    model: str = Field(..., description="Model identifier (provider:model format)")
    machine: str = Field(..., description="Hostname/machine name where session occurred")
    created_at: ISODatetime = Field(..., description="Session creation timestamp")
    compacted_from: Optional[str] = Field(default=None, description="Previous session ID if compacted")


class ContextSnapshot(BaseModel):
    """Initial context state for a session."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["context"] = "context"
    attachments: Dict[str, AttachmentRef] = Field(
        default_factory=dict, description="Attachment name -> reference mapping"
    )
    skills: List[str] = Field(default_factory=list, description="Loaded skill names")
    hash: str = Field(..., description="Overall context hash for change detection")


class ContextUpdate(BaseModel):
    """Delta update to context (recorded when context changes)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["context_update"] = "context_update"
    changed: Dict[str, AttachmentRef] = Field(default_factory=dict, description="Changed/added attachments")
    removed: List[str] = Field(default_factory=list, description="Removed attachment names")
    added_skills: List[str] = Field(default_factory=list, description="Newly loaded skills")
    removed_skills: List[str] = Field(default_factory=list, description="Unloaded skills")
    timestamp: ISODatetime = Field(..., description="When this update occurred")
    hash: str = Field(..., description="New context hash after this update")


class Turn(BaseModel):
    """A single conversation turn with full message history."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    type: Literal["turn"] = "turn"
    messages: List[Dict[str, Any]] = Field(..., description="Full LiteLLM messages (may have array content)")
    final_answer: Optional[str] = Field(default=None, description="Final answer if provided")
    user_summary: Optional[str] = Field(default=None, description="First user message truncated (for display)")
    tokens: Optional[int] = Field(default=None, description="Total tokens used in this turn")
    cost: Optional[float] = Field(default=None, description="Estimated cost for this turn")
    timestamp: ISODatetime = Field(..., description="When this turn occurred")
    model: Optional[str] = Field(default=None, description="Model used (may change mid-session)")
    duration_ms: Optional[int] = Field(default=None, description="Execution duration in milliseconds")
    functions_called: List[str] = Field(default_factory=list, description="Tool/function names called")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Channel routing metadata (source, channel_id, user_id, reply_to)"
    )


class CompactionSummary(BaseModel):
    """Summary from a compacted session."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["compaction_summary"] = "compaction_summary"
    summary: str = Field(..., description="LLM-generated summary of previous conversation")
    previous_turns: int = Field(..., description="Number of turns in the compacted session")
    retained_turns: int = Field(default=0, description="Number of recent turns kept verbatim after compaction")
    compaction_reason: Optional[CompactionReason] = Field(default=None)


class HookExecution(BaseModel):
    """Record of a hook execution with its output and exit code."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["hook_execution"] = "hook_execution"
    phase: str = Field(..., description="Hook phase: post_tool, pre_message, pre_compact, post_compact")
    name: Optional[str] = Field(default=None, description="Hook name or capture_as variable")
    command: str = Field(..., description="Rendered command that was executed")
    exit_code: int = Field(..., description="Process exit code")
    stdout: Optional[str] = Field(default=None, description="Standard output")
    stderr: Optional[str] = Field(default=None, description="Standard error")
    duration_ms: Optional[int] = Field(default=None, description="Execution duration in milliseconds")
    timestamp: ISODatetime = Field(..., description="When this hook was executed")


class SessionStatus(BaseModel):
    """Final status of an agent run, appended as last record."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["session_status"] = "session_status"
    status: Literal["success", "error", "interrupted"] = Field(..., description="Run outcome")
    error_message: Optional[str] = Field(default=None, description="Error details if failed")
    timestamp: ISODatetime = Field(..., description="When status was recorded")


# Type alias for any record type
SessionRecord = SessionMeta | ContextSnapshot | ContextUpdate | Turn | CompactionSummary | HookExecution | SessionStatus
