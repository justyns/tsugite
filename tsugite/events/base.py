"""Base event model and event type enum."""

from datetime import datetime, timezone
from enum import IntEnum

from pydantic import BaseModel, Field


class EventType(IntEnum):
    """Event type enumeration."""

    # Core execution events
    TASK_START = 1
    STEP_START = 2
    CODE_EXECUTION = 3
    TOOL_CALL = 4
    OBSERVATION = 5
    SKILL_LOADED = 6
    SKILL_UNLOADED = 7
    ERROR = 8
    FINAL_ANSWER = 9

    # LLM events
    LLM_MESSAGE = 10
    EXECUTION_RESULT = 11
    EXECUTION_LOGS = 12
    REASONING_CONTENT = 13
    REASONING_TOKENS = 14

    # Meta events
    COST_SUMMARY = 15
    STREAM_CHUNK = 16
    STREAM_COMPLETE = 17
    INFO = 20

    # New progress events
    DEBUG_MESSAGE = 21
    WARNING = 22
    STEP_PROGRESS = 23


class BaseEvent(BaseModel):
    """Base class for all UI events."""

    model_config = {"frozen": True, "use_enum_values": False}

    event_type: EventType = Field(frozen=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
