"""Data models for agent execution results."""

from typing import Any, List, Optional, Tuple

from pydantic import BaseModel, Field


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
        attachments: List of (name, content) tuples for cached context
    """

    response: str
    token_count: Optional[int] = None
    cost: Optional[float] = None
    step_count: int = 0
    execution_steps: List[Any] = Field(default_factory=list)
    system_message: Optional[str] = None
    attachments: List[Tuple[str, str]] = Field(default_factory=list)

    def __str__(self) -> str:
        """Allow result to be used as string for backward compatibility.

        Returns:
            The response string
        """
        return self.response
