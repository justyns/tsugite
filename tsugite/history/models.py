"""Pydantic models for history and conversation data structures."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ConversationMetadata(BaseModel):
    """Metadata for a conversation session.

    This is the first record in each conversation JSONL file, containing
    basic information about the conversation.
    """

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for backward/forward compatibility
        str_strip_whitespace=True,  # Auto-strip whitespace from strings
    )

    type: str = Field(default="metadata", description="Record type identifier")
    id: str = Field(..., description="Unique conversation identifier")
    agent: str = Field(..., description="Agent name used in this conversation")
    model: str = Field(..., description="Model identifier (provider:model format)")
    machine: str = Field(..., description="Hostname/machine name where conversation occurred")
    created_at: datetime = Field(..., description="Conversation creation timestamp")
    timestamp: Optional[datetime] = Field(default=None, description="Alias for created_at (for backward compatibility)")

    @field_validator("created_at", "timestamp", mode="before")
    @classmethod
    def parse_datetime(cls, v):
        """Parse ISO format strings to datetime objects."""
        if v is None:
            return None
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v

    @model_validator(mode="after")
    def set_timestamp_default(self):
        """Set timestamp to created_at if not provided."""
        if self.timestamp is None:
            self.timestamp = self.created_at
        return self


class Turn(BaseModel):
    """A single conversation turn (user message + assistant response).

    Represents one complete interaction in the conversation, including
    the user's input, assistant's response, and metadata about tool usage
    and cost.
    """

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for future extensions
        str_strip_whitespace=True,
    )

    type: str = Field(default="turn", description="Record type identifier")
    timestamp: datetime = Field(..., description="When this turn occurred")
    user: str = Field(..., description="User's input message")
    assistant: str = Field(..., description="Assistant's response")
    tools: list[str] = Field(default_factory=list, description="Tools used in this turn")
    tokens: Optional[int] = Field(default=None, description="Total tokens used in this turn")
    cost: Optional[float] = Field(default=None, description="Estimated cost for this turn")

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_datetime(cls, v):
        """Parse ISO format strings to datetime objects."""
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v


class IndexEntry(BaseModel):
    """Entry in the conversation index for fast lookups.

    The index maps conversation IDs to summary metadata, allowing quick
    queries without reading entire JSONL files.
    """

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields
        str_strip_whitespace=True,
    )

    agent: str = Field(..., description="Agent name")
    model: str = Field(..., description="Model identifier")
    machine: str = Field(..., description="Hostname/machine name")
    created_at: datetime = Field(..., description="When conversation was created")
    updated_at: datetime = Field(..., description="Last update timestamp")
    turn_count: int = Field(default=0, description="Total number of turns in conversation")
    total_tokens: Optional[int] = Field(default=None, description="Cumulative token count")
    total_cost: Optional[float] = Field(default=None, description="Cumulative cost")

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def parse_datetime(cls, v):
        """Parse ISO format strings to datetime objects."""
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v
