"""Custom exception classes for Tsugite."""

from typing import Any, List, Optional


class StateSerializationError(RuntimeError):
    """Raised when per-session state cannot be serialized to JSON.

    Attributes:
        session_id: Session whose state failed to persist (may be None for ad-hoc executors).
        key: The offending state key.
        reason: Short classification, e.g. "not-json-serializable" or "size-cap".
    """

    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        key: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        super().__init__(message)
        self.session_id = session_id
        self.key = key
        self.reason = reason


class AgentExecutionError(RuntimeError):
    """Exception raised when agent execution fails.

    Includes execution details for debugging and analysis.

    Attributes:
        message: Error message
        execution_steps: List of step results from agent execution
        token_usage: Token usage count (if available)
        cost: Cost of execution (if available)
        step_count: Number of steps taken
    """

    def __init__(
        self,
        message: str,
        execution_steps: Optional[List[Any]] = None,
        token_usage: Optional[int] = None,
        cost: Optional[float] = None,
        step_count: int = 0,
        partial_output: Optional[str] = None,
    ):
        super().__init__(message)
        self.execution_steps = execution_steps or []
        self.token_usage = token_usage
        self.cost = cost
        self.step_count = step_count
        self.partial_output = partial_output
