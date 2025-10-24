"""Custom exception classes for Tsugite."""

from typing import Any, List, Optional


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
    ):
        """Initialize AgentExecutionError.

        Args:
            message: Error message
            execution_steps: List of step results from execution
            token_usage: Token usage count
            cost: Cost of execution
            step_count: Number of steps taken
        """
        super().__init__(message)
        self.execution_steps = execution_steps or []
        self.token_usage = token_usage
        self.cost = cost
        self.step_count = step_count
