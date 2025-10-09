"""Code execution backends for agents.

Provides local execution using Python's exec().
WARNING: Not secure! Only use for development.

Maintains state (variables persist between runs).
"""

import io
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ExecutionResult:
    """Result from code execution."""

    output: str
    error: Optional[str]
    stdout: str
    stderr: str
    final_answer: Optional[Any] = None


class CodeExecutor(ABC):
    """Abstract interface for code execution.

    Defines the contract for code executors used by agents.
    """

    @abstractmethod
    async def execute(self, code: str) -> ExecutionResult:
        """Execute Python code and return results.

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult with output, errors, etc.
        """
        pass

    @abstractmethod
    async def send_variables(self, variables: Dict[str, Any]):
        """Inject variables into execution namespace.

        This is used for multi-step agents to pass results between steps.

        Args:
            variables: Dict of {name: value} to make available in code
        """
        pass


class LocalExecutor(CodeExecutor):
    """Simple local code executor using Python's exec().

    WARNING: This is NOT secure! Only use for development.

    Uses Python's built-in exec() function. State persists between
    runs by maintaining a shared namespace dict.

    Example:
        executor = LocalExecutor()

        # First run
        result = await executor.execute("x = 5")

        # Second run - x still exists!
        result = await executor.execute("print(x + 3)")  # Prints: 8
    """

    def __init__(self):
        """Initialize executor with empty namespace."""
        self.namespace = {}
        self._final_answer_value = None

        # Inject final_answer function into namespace
        def final_answer(value):
            self._final_answer_value = value
            print(f"__FINAL_ANSWER__: {value}")

        self.namespace["final_answer"] = final_answer

    async def execute(self, code: str) -> ExecutionResult:
        """Execute code using exec().

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult with output, error, stdout, stderr, and final_answer
        """
        # Reset final answer
        self._final_answer_value = None

        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            # Redirect output
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # Execute code in shared namespace
            exec(code, self.namespace)

            # Get output
            output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()

            return ExecutionResult(
                output=output,
                error=None,
                stdout=output,
                stderr=stderr_output,
                final_answer=self._final_answer_value,
            )

        except Exception as e:
            # Code execution failed
            error_msg = f"{type(e).__name__}: {str(e)}"
            return ExecutionResult(
                output=stdout_capture.getvalue(),
                error=error_msg,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue() + "\n" + error_msg,
                final_answer=None,
            )

        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    async def send_variables(self, variables: Dict[str, Any]):
        """Inject variables into namespace.

        Simply updates the shared namespace dict.

        Args:
            variables: Dict of {name: value} to inject
        """
        self.namespace.update(variables)
