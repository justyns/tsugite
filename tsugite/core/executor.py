"""Code execution backends for agents.

Provides local execution using Python's exec().
WARNING: Not secure! Only use for development.

Each turn runs in a fresh Python namespace. Only values assigned to the
injected `state` object persist across turns.
"""

import ast
import io
import pprint
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tsugite.core.state import load_state, save_state
from tsugite.exceptions import StateSerializationError

PPRINT_WIDTH = 100


def _looks_html_escaped(source: str) -> bool:
    """True if `source` is HTML-entity-escaped XML (observation content leaked into exec)."""
    i = 0
    for ch in source:
        if not ch.isspace():
            break
        i += 1
    return source.startswith("&lt;", i) or source.startswith("&amp;lt;", i)


# Tools with special executor handling (not injected via the normal tool wrapper path).
# These are implemented directly in the executor because they need event_bus access
# or special completion signaling.
EXECUTOR_BUILTIN_TOOLS = frozenset({"return_value", "final_answer", "send_message", "react_to_message"})


@dataclass
class ExecutionResult:
    """Result from code execution."""

    output: str
    error: Optional[str]
    stdout: str
    stderr: str
    return_value: Optional[Any] = None
    tools_called: List[str] = field(default_factory=list)
    variables_set: Dict[str, str] = field(default_factory=dict)  # name -> "type(size)"
    state_keys: Dict[str, str] = field(default_factory=dict)  # persisted state: name -> "type(size)"
    loaded_skills: Dict[str, str] = field(default_factory=dict)  # name -> content
    unloaded_skills: List[str] = field(default_factory=list)  # names unloaded this turn
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

        from tsugite.secrets.registry import get_registry

        _mask = get_registry().mask

        # Check for truncation first (before building attrs)
        output = _mask(self.output or "")
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
            parts.append(f"<error>{escape(_mask(self.error))}</error>")
            # Include traceback from stderr (last 10 lines)
            if self.stderr:
                tb_lines = _mask(self.stderr).strip().split("\n")[-10:]
                parts.append(f"<traceback>{escape(chr(10).join(tb_lines))}</traceback>")

        if self.truncated_to:
            parts.append(f"<truncated_to>{escape(self.truncated_to)}</truncated_to>")

        if self.variables_set:
            var_list = ", ".join(f"{escape(k)}={escape(_mask(v))}" for k, v in self.variables_set.items())
            parts.append(f"<variables_set>{var_list}</variables_set>")

        if self.state_keys:
            state_list = ", ".join(f"{escape(k)}={escape(_mask(v))}" for k, v in self.state_keys.items())
            parts.append(f"<state>{state_list}</state>")

        if self.return_value is not None:
            parts.append(f"<return_value>{escape(_mask(str(self.return_value)))}</return_value>")

        parts.append("</tsugite_execution_result>")
        return "\n".join(parts)


def _summarize_mapping(items) -> Dict[str, str]:
    """Summarize a (name, value) iterable as {name: type-and-size} for display."""
    out: Dict[str, str] = {}
    for name, value in items:
        try:
            out[name] = _summarize_variable(value)
        except Exception:
            out[name] = type(value).__name__
    return out


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

    Each call to ``execute()`` runs in a fresh namespace. Use the injected
    ``state`` dict to persist values across turns; all other bindings are
    discarded when the turn ends.

    Example:
        executor = LocalExecutor()

        await executor.execute("state['x'] = 5")
        await executor.execute("print(state['x'] + 3)")  # prints 8

        # But plain locals do NOT persist:
        await executor.execute("y = 10")
        await executor.execute("print(y)")  # NameError
    """

    def __init__(
        self,
        workspace_dir: Optional[Path] = None,
        event_bus: Optional[Any] = None,
        path_context: Optional[Any] = None,
        state_path: Optional[Path] = None,
        session_id: Optional[str] = None,
    ):
        """Initialize executor.

        Args:
            workspace_dir: Optional workspace directory (for reference, CWD set at CLI level)
            event_bus: Optional event bus for emitting events (used by send_message)
            path_context: Optional PathContext with invoked_from, workspace_dir, effective_cwd
            state_path: Optional path to a JSON file for persisting `state` across turns.
                When None, state is ephemeral (in-memory only).
            session_id: Optional session identifier, used in StateSerializationError messages.
        """
        self._return_value = None
        self._tools_called = []
        self._loaded_skills_for_turn: Dict[str, str] = {}
        self._unloaded_skills_for_turn: List[str] = []
        self.workspace_dir = workspace_dir
        self.event_bus = event_bus
        self.path_context = path_context
        self._state_path = state_path
        self._session_id = session_id
        self._state: Dict[str, Any] = load_state(state_path) if state_path else {}
        self._tool_functions: Dict[str, Callable[..., Any]] = {}
        self._sticky_injections: Dict[str, Any] = {}
        self._content_blocks: Dict[str, str] = {}

        self.namespace: Dict[str, Any] = self._build_turn_namespace()

    def _build_turn_namespace(self) -> Dict[str, Any]:
        """Construct a fresh namespace populated with built-ins, tools, state, and sticky injections."""
        ns: Dict[str, Any] = {}

        def return_value(*args, **kwargs):
            if args:
                self._return_value = args[0]
            elif kwargs:
                self._return_value = next(iter(kwargs.values()))

        ns["return_value"] = return_value
        # final_answer is kept as a backward-compat alias for older agent
        # markdown files. New agents should use return_value().
        ns["final_answer"] = return_value

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

        ns["send_message"] = send_message

        def react_to_message(emoji="", message_id=None):
            if self.event_bus:
                from tsugite.events import ReactionEvent

                self.event_bus.emit(ReactionEvent(emoji=str(emoji), message_id=message_id))
            return f"Reacted with {emoji}"

        ns["react_to_message"] = react_to_message

        def _blocked_open(*args, **kwargs):
            raise RuntimeError(
                "open() is not available. Use the provided tools instead:\n"
                "  - read_file(path) to read file contents\n"
                "  - write_file(path, content) to write to files"
            )

        ns["open"] = _blocked_open

        if self.path_context:
            ns["WORKSPACE_DIR"] = str(self.path_context.workspace_dir) if self.path_context.workspace_dir else None
            ns["INVOKED_FROM"] = str(self.path_context.invoked_from) if self.path_context.invoked_from else None
        else:
            ns["WORKSPACE_DIR"] = None
            ns["INVOKED_FROM"] = None

        ns.update(self._tool_functions)
        ns.update(self._sticky_injections)
        ns.update(self._content_blocks)
        ns["state"] = self._state
        return ns

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

    async def execute(self, code: str) -> ExecutionResult:
        """Execute code using exec().

        Automatically displays the value of the last expression (REPL-like behavior).
        CWD is managed at CLI level - no directory changes here.

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult with output, error, stdout, stderr, return_value, tools_called, and variables_set
        """
        self._return_value = None
        self._tools_called = []
        self._loaded_skills_for_turn = {}
        self._unloaded_skills_for_turn = []

        # TODO: This _probably_ isn't needed, but leaving for now as an extra safeguard
        if _looks_html_escaped(code):
            return ExecutionResult(
                output="",
                error=(
                    "Refusing to exec HTML-entity-escaped source - the `&lt;` prefix "
                    "indicates XML observation content was fed into exec(). The "
                    "xml.sax.saxutils.escape() pass is for LLM-facing XML only."
                ),
                stdout="",
                stderr="",
            )

        # Set executor on skill manager so load_skill() can track
        from tsugite.tools.skills import get_skill_manager

        skill_manager = get_skill_manager()
        skill_manager.set_executor(self)

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        old_stdout = sys.stdout
        old_stderr = sys.stderr

        self.namespace = self._build_turn_namespace()
        namespace_before = set(self.namespace.keys())

        exec_error: Optional[str] = None
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

        except Exception as e:
            exec_error = f"{type(e).__name__}: {str(e)}"

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        variables_set = self._get_new_variables(namespace_before)
        state_keys = self._summarize_state_keys()
        save_error = self._save_state()

        error_msg = exec_error
        if save_error is not None:
            error_msg = f"{exec_error}\n{save_error}" if exec_error else save_error

        stderr_output = stderr_capture.getvalue()
        if exec_error:
            stderr_output = stderr_output + "\n" + exec_error
        if save_error:
            stderr_output = stderr_output + "\n" + save_error

        stdout_output = stdout_capture.getvalue()
        return ExecutionResult(
            output=stdout_output,
            error=error_msg,
            stdout=stdout_output,
            stderr=stderr_output,
            return_value=None if exec_error else self._return_value,
            tools_called=self._tools_called.copy(),
            variables_set=variables_set,
            state_keys=state_keys,
            loaded_skills=self._loaded_skills_for_turn.copy(),
            unloaded_skills=list(self._unloaded_skills_for_turn),
        )

    def _get_new_variables(self, namespace_before: set) -> Dict[str, str]:
        new_vars = set(self.namespace.keys()) - namespace_before
        return _summarize_mapping((name, self.namespace[name]) for name in new_vars if not name.startswith("_"))

    def _summarize_state_keys(self) -> Dict[str, str]:
        return _summarize_mapping(self._state.items())

    def _save_state(self) -> Optional[str]:
        """Persist session state. Returns an error message string on failure, else None."""
        if self._state_path is None:
            return None
        try:
            save_state(self._state, self._state_path, session_id=self._session_id)
        except StateSerializationError as e:
            return f"StateSerializationError: {e}"
        return None

    async def send_variables(self, variables: Dict[str, Any]):
        """Register harness-level variables that are re-injected at the start of every turn.

        These are intended for caller-supplied inputs (e.g. multi-step agent parameters);
        they are not serialized with session state.
        """
        self._sticky_injections.update(variables)
        self.namespace.update(variables)

    async def inject_content_blocks(self, blocks: Dict[str, str]):
        """Replace the content-block variables available to the next turn.

        Content blocks are scoped to the turn that declared them; earlier
        turns' block names do not carry forward.
        """
        self._content_blocks = dict(blocks)
        self.namespace.update(blocks)

    def register_tools(self, tools: Dict[str, Callable[..., Any]]):
        """Register tool functions that should be re-injected into the namespace every turn.

        Called by the agent after tool setup; tool wrappers are not serialized into state.
        """
        self._tool_functions.update(tools)
        self.namespace.update(tools)

    def register_loaded_skill(self, name: str, content: str):
        """Register a skill loaded during this execution turn.

        Called by load_skill() tool to track skills for observation embedding.

        Args:
            name: Skill name
            content: Rendered skill content
        """
        self._loaded_skills_for_turn[name] = content

    def register_unloaded_skill(self, name: str):
        """Record that a skill was unloaded during this execution turn.

        Called by unload_skill() tool so the daemon can drop the name from
        session-level sticky state after the turn completes.
        """
        if name not in self._unloaded_skills_for_turn:
            self._unloaded_skills_for_turn.append(name)
