"""Data models for agent execution results."""

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class AgentSkippedError(Exception):
    """Raised when an agent's run_if guard evaluates to false."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class AgentExecutionResult(BaseModel):
    """Result from agent execution with metrics and metadata.

    This model provides a structured way to return agent execution results
    with optional metrics and debugging information. It replaces the fragile
    7-tuple return value that was previously used.

    Attributes:
        response: The agent's final response string
        token_count: Total tokens used (prompt + completion)
        cost: Total cost in dollars
        step_count: Number of execution steps (think-code-observe cycles)
        execution_steps: List of execution step details (for debugging)
        system_message: The system prompt used (for debugging)
        attachments: List of Attachment objects for cached context
    """

    response: str
    token_count: Optional[int] = None
    cost: Optional[float] = None
    step_count: int = 0
    execution_steps: List[Any] = Field(default_factory=list)
    system_message: Optional[str] = None
    attachments: List[Any] = Field(
        default_factory=list
    )  # List of Attachment objects (using Any for Pydantic compatibility)
    provider_state: Optional[dict] = None
    last_input_tokens: Optional[int] = None
    # Cache-token totals accumulated across the turn's model calls. The daemon
    # usage-store call reads these (falling back to provider_state) so OpenAI-family
    # turns - whose get_state() carries no cache - still record their cache reads.
    cache_creation_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    session_id: Optional[str] = None  # ID of the SessionStorage the agent recorded events to

    def __str__(self) -> str:
        """Allow result to be used as string for backward compatibility.

        Returns:
            The response string
        """
        return self.response
