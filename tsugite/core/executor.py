"""Code execution backends for agents.

Provides local execution using Python's exec().
WARNING: Not secure! Only use for development.

Maintains state (variables persist between runs).
"""

import ast
import io
import pprint
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

PPRINT_WIDTH = 100


@dataclass
class ExecutionResult:
    """Result from code execution."""

    output: str
    error: Optional[str]
    stdout: str
    stderr: str
    final_answer: Optional[Any] = None
    tools_called: List[str] = field(default_factory=list)


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
        self._tools_called = []

        # Inject final_answer function into namespace
        def final_answer(value):
            self._final_answer_value = value
            # Don't print here - the UI handler will display it properly via FINAL_ANSWER event

        self.namespace["final_answer"] = final_answer

    def _split_code_for_last_expr(self, code: str) -> tuple[str, Optional[str]]:
        """Split code into setup and last expression if applicable.

        If the last statement is an expression, return (setup, last_expr).
        Otherwise, return (code, None).

        Args:
            code: Python code to analyze

        Returns:
            Tuple of (setup_code, last_expression_or_none)
        """
        try:
            tree = ast.parse(code)
            if not tree.body:
                return (code, None)

            # Check if last statement is an expression
            last_node = tree.body[-1]
            if not isinstance(last_node, ast.Expr):
                return (code, None)

            if len(tree.body) == 1:
                setup_code = ""
                last_expr = ast.unparse(last_node.value)
            else:
                setup_tree = ast.Module(body=tree.body[:-1], type_ignores=[])
                setup_code = ast.unparse(setup_tree)
                last_expr = ast.unparse(last_node.value)

            return (setup_code, last_expr)

        except SyntaxError:
            return (code, None)

    def _format_value(self, value: Any) -> str:
        """Format a value for display.

        Uses pprint for complex objects, repr for simple ones.

        Args:
            value: Value to format

        Returns:
            Formatted string representation
        """
        if isinstance(value, (dict, list, tuple, set)):
            return pprint.pformat(value, width=PPRINT_WIDTH, compact=False)
        return repr(value)

    def _check_code_safety(self, code: str) -> Optional[str]:
        """Check code for anti-patterns before execution.

        Detects common mistakes where LLMs use built-in Python functions
        instead of the provided tools.

        Args:
            code: Python code to check

        Returns:
            Error message string if violations found, None if code is safe
        """
        import re

        if re.search(r"\bopen\s*\(", code):
            code_without_strings = re.sub(r'["\'].*?["\']', "", code)
            code_without_comments = re.sub(r"#.*$", "", code_without_strings, flags=re.MULTILINE)

            if re.search(r"\bopen\s*\(", code_without_comments):
                return """Code Safety Check Failed: Detected use of 'open()' for file operations.

Please use the provided tools instead:
  - read_file(path) - to read file contents
  - write_file(path, content) - to write to files

Example:
  # Instead of:
  with open('file.txt') as f:
      content = f.read()

  # Use:
  content = read_file('file.txt')

  # Instead of:
  with open('output.txt', 'w') as f:
      f.write(data)

  # Use:
  write_file('output.txt', data)"""

        return None

    async def execute(self, code: str) -> ExecutionResult:
        """Execute code using exec().

        Automatically displays the value of the last expression (REPL-like behavior).

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult with output, error, stdout, stderr, final_answer, and tools_called
        """
        self._final_answer_value = None
        self._tools_called = []

        safety_error = self._check_code_safety(code)
        if safety_error:
            return ExecutionResult(
                output="",
                error=safety_error,
                stdout="",
                stderr=safety_error,
                final_answer=None,
                tools_called=[],
            )

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            setup_code, last_expr = self._split_code_for_last_expr(code)

            if last_expr:
                if setup_code.strip():
                    exec(setup_code, self.namespace)

                result = eval(last_expr, self.namespace)

                if result is not None:
                    formatted = self._format_value(result)
                    print(formatted)
            else:
                exec(code, self.namespace)

            output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()

            return ExecutionResult(
                output=output,
                error=None,
                stdout=output,
                stderr=stderr_output,
                final_answer=self._final_answer_value,
                tools_called=self._tools_called.copy(),
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            return ExecutionResult(
                output=stdout_capture.getvalue(),
                error=error_msg,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue() + "\n" + error_msg,
                final_answer=None,
                tools_called=self._tools_called.copy(),
            )

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    async def send_variables(self, variables: Dict[str, Any]):
        """Inject variables into namespace.

        Simply updates the shared namespace dict.

        Args:
            variables: Dict of {name: value} to inject
        """
        self.namespace.update(variables)
