"""All event classes consolidated in one module.

Error Handling Patterns:
------------------------
1. Tool Results (ObservationEvent):
   - Success: ObservationEvent(success=True, observation="result", tool="tool_name")
   - Failure: ObservationEvent(success=False, error="error msg", tool="tool_name")

2. General/Fatal Errors (ErrorEvent):
   - ErrorEvent(error="error msg", error_type="Error Type", step=N)
   - Used for: Format errors, max turns exceeded, critical failures
"""

from typing import Any, Dict, Optional

from pydantic import Field

from .base import BaseEvent, EventType

# ============================================================================
# Execution Events
# ============================================================================


class TaskStartEvent(BaseEvent):
    """Agent execution starts."""

    event_type: EventType = Field(default=EventType.TASK_START, frozen=True)
    task: str
    model: str


class StepStartEvent(BaseEvent):
    """New reasoning turn.

    Args:
        step: Current step number
        max_turns: Maximum turns allowed
        recovering_from_error: True if previous turn had an error and LLM is attempting recovery
    """

    event_type: EventType = Field(default=EventType.STEP_START, frozen=True)
    step: int = Field(ge=1)
    max_turns: Optional[int] = Field(default=None, ge=1)
    recovering_from_error: bool = False


class CodeExecutionEvent(BaseEvent):
    """Code being executed."""

    event_type: EventType = Field(default=EventType.CODE_EXECUTION, frozen=True)
    code: str
    language: str = "python"


class ObservationEvent(BaseEvent):
    """Observation from tool execution or code execution.

    Usage:
    - Tool success: ObservationEvent(success=True, observation="result", tool="tool_name")
    - Tool failure: ObservationEvent(success=False, error="error", tool="tool_name")
    - Code execution: ObservationEvent(observation="output", tool=None)
    """

    event_type: EventType = Field(default=EventType.OBSERVATION, frozen=True)
    observation: str = ""
    tool: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


class SkillLoadedEvent(BaseEvent):
    """Skill loaded into agent context."""

    event_type: EventType = Field(default=EventType.SKILL_LOADED, frozen=True)
    skill_name: str
    description: Optional[str] = None


class SkillUnloadedEvent(BaseEvent):
    """Skill unloaded from agent context."""

    event_type: EventType = Field(default=EventType.SKILL_UNLOADED, frozen=True)
    skill_name: str


class SkillLoadFailedEvent(BaseEvent):
    """Skill failed to load."""

    event_type: EventType = Field(default=EventType.SKILL_LOAD_FAILED, frozen=True)
    skill_name: str
    error_message: str


class FinalAnswerEvent(BaseEvent):
    """Agent completed."""

    event_type: EventType = Field(default=EventType.FINAL_ANSWER, frozen=True)
    answer: str
    turns: Optional[int] = Field(default=None, ge=1)
    tokens: Optional[int] = Field(default=None, ge=0)
    cost: Optional[float] = Field(default=None, ge=0)


# ============================================================================
# LLM Events
# ============================================================================


class LLMMessageEvent(BaseEvent):
    """Reasoning/thought process."""

    event_type: EventType = Field(default=EventType.LLM_MESSAGE, frozen=True)
    content: str
    title: Optional[str] = None
    step: Optional[int] = Field(default=None, ge=1)


class ReasoningContentEvent(BaseEvent):
    """Model reasoning (Claude, Deepseek)."""

    event_type: EventType = Field(default=EventType.REASONING_CONTENT, frozen=True)
    content: str
    step: Optional[int] = Field(default=None, ge=1)


class ReasoningTokensEvent(BaseEvent):
    """Reasoning token count (o1, o3)."""

    event_type: EventType = Field(default=EventType.REASONING_TOKENS, frozen=True)
    tokens: int = Field(ge=0)
    step: Optional[int] = Field(default=None, ge=1)


# ============================================================================
# Meta Events
# ============================================================================


class InfoEvent(BaseEvent):
    """Informational message."""

    event_type: EventType = Field(default=EventType.INFO, frozen=True)
    message: str


class ErrorEvent(BaseEvent):
    """Execution error.

    Args:
        error: Error message
        error_type: Type of error (e.g., "Execution Error", "Format Error")
        step: Turn/step number where error occurred
        traceback: Optional traceback information
        suppress_from_ui: If True, hide from UI unless verbose/debug is enabled.
            Used for recoverable errors (tool failures) that the LLM will self-correct.
            Fatal errors (max turns, format errors) should set this to False.
    """

    event_type: EventType = Field(default=EventType.ERROR, frozen=True)
    error: str
    error_type: Optional[str] = None
    step: Optional[int] = Field(default=None, ge=1)
    traceback: Optional[str] = None
    suppress_from_ui: bool = False


class CostSummaryEvent(BaseEvent):
    """Token/cost metrics with prompt caching support.

    Prompt caching fields (supported by OpenAI, Anthropic, AWS Bedrock, Deepseek):
    - cached_tokens: Total cached tokens read (unified across providers)
    - cache_creation_input_tokens: Tokens used to create cache (Anthropic-specific)
    - cache_read_input_tokens: Tokens read from cache (Anthropic-specific)

    For Anthropic, both creation and read tokens are also included in cached_tokens.

    Cumulative fields (for chat mode):
    - cumulative_tokens: Total tokens across all turns in this conversation
    - cumulative_cost: Total cost across all turns in this conversation
    """

    event_type: EventType = Field(default=EventType.COST_SUMMARY, frozen=True)
    tokens: Optional[int] = Field(default=None, ge=0)
    cost: Optional[float] = Field(default=None, ge=0)
    model: Optional[str] = None
    duration_seconds: Optional[float] = Field(default=None, ge=0)

    # Prompt caching fields
    cached_tokens: Optional[int] = Field(default=None, ge=0)
    cache_creation_input_tokens: Optional[int] = Field(default=None, ge=0)
    cache_read_input_tokens: Optional[int] = Field(default=None, ge=0)

    # Cumulative tracking (for chat mode)
    cumulative_tokens: Optional[int] = Field(default=None, ge=0)
    cumulative_cost: Optional[float] = Field(default=None, ge=0)


class StreamChunkEvent(BaseEvent):
    """Streaming response chunk."""

    event_type: EventType = Field(default=EventType.STREAM_CHUNK, frozen=True)
    chunk: str


class StreamCompleteEvent(BaseEvent):
    """Streaming finished."""

    event_type: EventType = Field(default=EventType.STREAM_COMPLETE, frozen=True)
    complete: bool = True


# ============================================================================
# Progress Events
# ============================================================================


class DebugMessageEvent(BaseEvent):
    """Debug output."""

    event_type: EventType = Field(default=EventType.DEBUG_MESSAGE, frozen=True)
    message: str
    context: Optional[Dict[str, Any]] = None


class WarningEvent(BaseEvent):
    """Warning message."""

    event_type: EventType = Field(default=EventType.WARNING, frozen=True)
    message: str
    category: Optional[str] = None


class StepProgressEvent(BaseEvent):
    """Progress update."""

    event_type: EventType = Field(default=EventType.STEP_PROGRESS, frozen=True)
    message: str
    step: Optional[int] = Field(default=None, ge=1)
    total: Optional[int] = Field(default=None, ge=1)
    percentage: Optional[float] = Field(default=None, ge=0, le=100)


class FileReadEvent(BaseEvent):
    """File read notification with metadata."""

    event_type: EventType = Field(default=EventType.FILE_READ, frozen=True)
    path: str
    line_count: int = Field(ge=0)
    byte_count: int = Field(ge=0)
    operation: str  # "prefetch", "attachment", "tool_call", "auto_context"


class FileWriteEvent(BaseEvent):
    """File write notification with metadata."""

    event_type: EventType = Field(default=EventType.FILE_WRITE, frozen=True)
    path: str
    line_count: int = Field(ge=0)
    byte_count: int = Field(ge=0)
    operation: str  # "tool_call"


# ============================================================================
# Audit Events
# ============================================================================


class ToolCallEvent(BaseEvent):
    """Tool invocation started (audit trail)."""

    event_type: EventType = Field(default=EventType.TOOL_CALL, frozen=True)
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    step: Optional[int] = Field(default=None, ge=1)


class ToolResultEvent(BaseEvent):
    """Tool invocation completed (audit trail)."""

    event_type: EventType = Field(default=EventType.TOOL_RESULT, frozen=True)
    tool_name: str
    success: bool = True
    result_summary: str = ""
    duration_ms: Optional[int] = Field(default=None, ge=0)
    step: Optional[int] = Field(default=None, ge=1)
