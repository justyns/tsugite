"""Pydantic models for session storage V2.

This module defines the data models for the turn-based session storage format.
Key features:
- Stores context once with hash references, tracks changes as deltas
- Stores full messages per turn for exact reconstruction
- Excludes system prompt (rebuild from code)
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    """Metadata for a session (first record in JSONL file).

    Contains information about the session itself, not the conversation content.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    type: Literal["session_meta"] = "session_meta"
    workspace: Optional[str] = Field(default=None, description="Workspace name if applicable")
    agent: str = Field(..., description="Agent name used in this session")
    model: str = Field(..., description="Model identifier (provider:model format)")
    machine: str = Field(..., description="Hostname/machine name where session occurred")
    created_at: datetime = Field(..., description="Session creation timestamp")
    compacted_from: Optional[str] = Field(default=None, description="Previous session ID if compacted")

    @field_validator("created_at", mode="before")
    @classmethod
    def parse_datetime(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v


class ContextSnapshot(BaseModel):
    """Initial context state for a session.

    Captures all attachments and skills at session start.
    Content is stored by reference (hash or URL), not inline.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["context"] = "context"
    attachments: Dict[str, AttachmentRef] = Field(
        default_factory=dict, description="Attachment name -> reference mapping"
    )
    skills: List[str] = Field(default_factory=list, description="Loaded skill names")
    hash: str = Field(..., description="Overall context hash for change detection")


class ContextUpdate(BaseModel):
    """Delta update to context (recorded when context changes).

    Only records what changed, not the full context.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["context_update"] = "context_update"
    changed: Dict[str, AttachmentRef] = Field(default_factory=dict, description="Changed/added attachments")
    removed: List[str] = Field(default_factory=list, description="Removed attachment names")
    added_skills: List[str] = Field(default_factory=list, description="Newly loaded skills")
    removed_skills: List[str] = Field(default_factory=list, description="Unloaded skills")
    timestamp: datetime = Field(..., description="When this update occurred")
    hash: str = Field(..., description="New context hash after this update")

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v


class Turn(BaseModel):
    """A single conversation turn with full message history.

    Contains the complete LiteLLM message array for exact reconstruction.
    Messages may have array content for multi-modal (images, etc).
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    type: Literal["turn"] = "turn"
    messages: List[Dict[str, Any]] = Field(..., description="Full LiteLLM messages (may have array content)")
    final_answer: Optional[str] = Field(default=None, description="Final answer if provided")
    user_summary: Optional[str] = Field(default=None, description="First user message truncated (for display)")
    tokens: Optional[int] = Field(default=None, description="Total tokens used in this turn")
    cost: Optional[float] = Field(default=None, description="Estimated cost for this turn")
    timestamp: datetime = Field(..., description="When this turn occurred")
    model: Optional[str] = Field(default=None, description="Model used (may change mid-session)")
    duration_ms: Optional[int] = Field(default=None, description="Execution duration in milliseconds")
    functions_called: List[str] = Field(default_factory=list, description="Tool/function names called")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Channel routing metadata (source, channel_id, user_id, reply_to)"
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v


class CompactionSummary(BaseModel):
    """Summary from a compacted session.

    When a session is compacted, this record preserves a summary of
    the previous conversation for context.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["compaction_summary"] = "compaction_summary"
    summary: str = Field(..., description="LLM-generated summary of previous conversation")
    previous_turns: int = Field(..., description="Number of turns in the compacted session")


# Type alias for any record type
SessionRecord = SessionMeta | ContextSnapshot | ContextUpdate | Turn | CompactionSummary
