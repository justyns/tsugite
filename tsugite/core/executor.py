"""Code execution backends for agents.

Provides local execution using Python's exec().
WARNING: Not secure! Only use for development.

Maintains state (variables persist between runs).
"""

import ast
import io
import pprint
import sys
from dataclasses import dataclass, field
from pathlib import Path
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
    variables_set: Dict[str, str] = field(default_factory=dict)  # name -> "type(size)"
    loaded_skills: Dict[str, str] = field(default_factory=dict)  # name -> content
    truncated: bool = False
    truncated_to: Optional[str] = None  # Path to full output if truncated

    def to_xml(self, duration_ms: int = 0, max_output_kb: int = 50) -> str:
        """Convert execution result to structured XML format.

        Args:
            duration_ms: Execution duration in milliseconds
            max_output_kb: Maximum output size in KB before truncation

        Returns:
            XML string with execution result
        """
        from xml.sax.saxutils import escape

        # Check for truncation first (before building attrs)
        output = self.output or ""
        max_bytes = max_output_kb * 1024
        if len(output) > max_bytes:
            output = output[:max_bytes]
            self.truncated = True

        status = "error" if self.error else "success"
        attrs = f'status="{status}"'
        if duration_ms:
            attrs += f' duration_ms="{duration_ms}"'
        if self.truncated:
            attrs += ' truncated="true"'

        parts = [f"<tsugite_execution_result {attrs}>"]
        parts.append(f"<output>{escape(output)}</output>")

        if self.error:
            parts.append(f"<error>{escape(self.error)}</error>")
            # Include traceback from stderr (last 10 lines)
            if self.stderr:
                tb_lines = self.stderr.strip().split("\n")[-10:]
                parts.append(f"<traceback>{escape(chr(10).join(tb_lines))}</traceback>")

        if self.truncated_to:
            parts.append(f"<truncated_to>{escape(self.truncated_to)}</truncated_to>")

        if self.variables_set:
            var_list = ", ".join(f"{escape(k)}={escape(v)}" for k, v in self.variables_set.items())
            parts.append(f"<variables_set>{var_list}</variables_set>")

        if self.final_answer is not None:
            parts.append(f"<final_answer>{escape(str(self.final_answer))}</final_answer>")

        parts.append("</tsugite_execution_result>")
        return "\n".join(parts)


def _summarize_variable(value: Any) -> str:
    """Summarize a variable's type and size for display.

    Args:
        value: The variable value to summarize

    Returns:
        Summary string like "dict(3 keys)" or "list(5 items)"
    """
    t = type(value).__name__
    if isinstance(value, dict):
        return f"{t}({len(value)} keys)"
    elif isinstance(value, (list, tuple, set, frozenset)):
        return f"{t}({len(value)} items)"
    elif isinstance(value, str):
        return f"{t}({len(value)} chars)"
    elif isinstance(value, bytes):
        return f"{t}({len(value)} bytes)"
    elif hasattr(value, "shape"):  # numpy/pandas
        return f"{t}(shape={value.shape})"
    elif hasattr(value, "__len__"):
        try:
            return f"{t}({len(value)} items)"
        except Exception:
            pass
    return t


class LocalExecutor:
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

    def __init__(
        self,
        workspace_dir: Optional[Path] = None,
        event_bus: Optional[Any] = None,
        path_context: Optional[Any] = None,
    ):
        """Initialize executor with empty namespace.

        Args:
            workspace_dir: Optional workspace directory (for reference, CWD set at CLI level)
            event_bus: Optional event bus for emitting events (used by send_message)
            path_context: Optional PathContext with invoked_from, workspace_dir, effective_cwd
        """
        self.namespace = {}
        self._final_answer_value = None
        self._tools_called = []
        self._loaded_skills_for_turn: Dict[str, str] = {}
        self.workspace_dir = workspace_dir
        self.event_bus = event_bus
        self.path_context = path_context

        # Inject final_answer function into namespace
        # Accept any arg name (value, result, etc.) since LLMs may vary
        def final_answer(*args, **kwargs):
            if args:
                self._final_answer_value = args[0]
            elif kwargs:
                self._final_answer_value = next(iter(kwargs.values()))

        self.namespace["final_answer"] = final_answer

        # Inject send_message function for progress updates
        def send_message(*args, **kwargs):
            if args:
                msg = args[0]
            elif kwargs:
                msg = kwargs.get("message") or next(iter(kwargs.values()))
            else:
                msg = ""

            if self.event_bus:
                from tsugite.events import InfoEvent

                self.event_bus.emit(InfoEvent(message=str(msg)))
            return f"Message sent: {msg}"

        self.namespace["send_message"] = send_message

        # Inject path context variables for workspace-aware code
        if path_context:
            self.namespace["WORKSPACE_DIR"] = str(path_context.workspace_dir) if path_context.workspace_dir else None
            self.namespace["INVOKED_FROM"] = str(path_context.invoked_from) if path_context.invoked_from else None
        else:
            self.namespace["WORKSPACE_DIR"] = None
            self.namespace["INVOKED_FROM"] = None

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
        CWD is managed at CLI level - no directory changes here.

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult with output, error, stdout, stderr, final_answer, tools_called, and variables_set
        """
        self._final_answer_value = None
        self._tools_called = []
        self._loaded_skills_for_turn = {}

        # Set executor on skill manager so load_skill() can track
        from tsugite.tools.skills import get_skill_manager

        skill_manager = get_skill_manager()
        skill_manager.set_executor(self)

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

        # Track variables before execution
        namespace_before = set(self.namespace.keys())

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

            # Track new variables (exclude _ prefixed private vars)
            variables_set = self._get_new_variables(namespace_before)

            return ExecutionResult(
                output=output,
                error=None,
                stdout=output,
                stderr=stderr_output,
                final_answer=self._final_answer_value,
                tools_called=self._tools_called.copy(),
                variables_set=variables_set,
                loaded_skills=self._loaded_skills_for_turn.copy(),
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"

            # Still capture any variables set before error
            variables_set = self._get_new_variables(namespace_before)

            return ExecutionResult(
                output=stdout_capture.getvalue(),
                error=error_msg,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue() + "\n" + error_msg,
                final_answer=None,
                tools_called=self._tools_called.copy(),
                variables_set=variables_set,
                loaded_skills=self._loaded_skills_for_turn.copy(),
            )

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _get_new_variables(self, namespace_before: set) -> Dict[str, str]:
        """Get new variables created since namespace_before snapshot.

        Args:
            namespace_before: Set of variable names before execution

        Returns:
            Dict of {variable_name: summary} for new variables
        """
        namespace_after = set(self.namespace.keys())
        new_vars = namespace_after - namespace_before

        variables_set = {}
        for var_name in new_vars:
            # Skip private variables (starting with _)
            if var_name.startswith("_"):
                continue
            try:
                variables_set[var_name] = _summarize_variable(self.namespace[var_name])
            except Exception:
                variables_set[var_name] = type(self.namespace[var_name]).__name__
        return variables_set

    async def send_variables(self, variables: Dict[str, Any]):
        """Inject variables into namespace.

        Simply updates the shared namespace dict.

        Args:
            variables: Dict of {name: value} to inject
        """
        self.namespace.update(variables)

    def register_loaded_skill(self, name: str, content: str):
        """Register a skill loaded during this execution turn.

        Called by load_skill() tool to track skills for observation embedding.

        Args:
            name: Skill name
            content: Rendered skill content
        """
        self._loaded_skills_for_turn[name] = content
