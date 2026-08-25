"""Code execution backends for agents.

Provides local execution using Python's exec().
WARNING: Not secure! Only use for development.

Each turn runs in a fresh Python namespace. Only values assigned to the
injected `state` object persist across turns.
"""

import ast
import asyncio
import contextlib
import io
import os
import pprint
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, runtime_checkable

from tsugite.core.state import load_state, save_state
from tsugite.exceptions import StateSerializationError
from tsugite.prompt_xml import El
from tsugite.secrets.redaction import redact_sensitive_obj

PPRINT_WIDTH = 100

# CWD is process-global. Concurrent execute() calls in different threads must
# serialize their chdir+exec+restore window; the lock is held across the whole
# user-code exec(), so heavy parallel LocalExecutor turns serialize end-to-end.
# Accepted trade-off: silent disagreement between raw `os.getcwd()` and tool
# path resolution is worse than the serialization. A background spawn returns
# immediately so a parent holding the lock doesn't starve a nested child.
_chdir_lock = threading.Lock()


@contextlib.contextmanager
def _locked_chdir(target: Optional[Path]) -> Iterator[None]:
    """chdir to target under `_chdir_lock`; restore CWD and release on exit.

    No-op when target is None.
    """
    if target is None:
        yield
        return
    with _chdir_lock:
        previous = os.getcwd()
        os.chdir(str(target))
        try:
            yield
        finally:
            os.chdir(previous)


def _looks_html_escaped(source: str) -> bool:
    """True if `source` is HTML-entity-escaped XML (observation content leaked into exec)."""
    stripped = source.lstrip()
    return stripped.startswith(("&lt;", "&amp;lt;"))


# Tools with special executor handling (not injected via the normal tool wrapper path).
# These are implemented directly in the executor because they need event_bus access
# or special completion signaling.
EXECUTOR_BUILTIN_TOOLS = frozenset({"return_value", "final_answer", "send_message"})

# Max execution output shown to the model before truncation. Shared so the live
# observation (ExecutionResult.to_xml) and the replayed one (history reconstruction)
# truncate at the same boundary and stay byte-stable.
MAX_EXECUTION_OUTPUT_KB = 50

# Per-call record caps for ExecutionResult.tool_calls (persisted with the
# code_execution event): arguments and outputs are display detail for the UI,
# not model input, so they stay small.
TOOL_CALL_ARG_MAX = 500
TOOL_CALL_OUTPUT_MAX = 2000
# Titles are LLM-authored and nothing else bounds them before history.
GROUP_TITLE_MAX = 200


def _cap_text(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def _jsonable_call_args(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-safe, capped copy of a call's kwargs (non-scalars fall back to repr)."""
    safe: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if v is None or isinstance(v, (int, float, bool)):
            safe[k] = v
        elif isinstance(v, str):
            safe[k] = _cap_text(v, TOOL_CALL_ARG_MAX)
        else:
            safe[k] = _cap_text(repr(v), TOOL_CALL_ARG_MAX)
    return safe


def truncate_observation(text: str, max_output_kb: int = MAX_EXECUTION_OUTPUT_KB) -> tuple[str, bool]:
    """Clip text to the observation cap, returning (text, was_truncated).

    Replay clips the same way: the event holds the full output but the live turn
    only showed this much, so replaying it whole would diverge from what the
    model saw.
    """
    max_bytes = max_output_kb * 1024
    if len(text) > max_bytes:
        return text[:max_bytes], True
    return text, False


def build_execution_result(
    *,
    output: str,
    error: Optional[str] = None,
    traceback: Optional[str] = None,
    truncated_to: Optional[str] = None,
    variables_set: Optional[Dict[str, str]] = None,
    state_keys: Optional[Dict[str, str]] = None,
    return_value: Optional[str] = None,
    duration_ms: Optional[int] = None,
    truncated: bool = False,
    ts: Optional[str] = None,
) -> El:
    """The `<tsugite_execution_result>` envelope, shared by the live turn and replay.

    Callers mask secrets before handing values in.
    """

    def _pairs(mapping: Dict[str, str]) -> str:
        return ", ".join(f"{k}={v}" for k, v in mapping.items())

    children = [El("output", [output], inline=True)]
    if error:
        children.append(El("error", [error], inline=True))
        if traceback:
            children.append(El("traceback", [traceback], inline=True))
    if truncated_to:
        children.append(El("truncated_to", [truncated_to], inline=True))
    if variables_set:
        children.append(El("variables_set", [_pairs(variables_set)], inline=True))
    if state_keys:
        children.append(El("state", [_pairs(state_keys)], inline=True))
    if return_value is not None:
        children.append(El("return_value", [return_value], inline=True))

    return El(
        "tsugite_execution_result",
        children,
        {
            "status": "error" if error else "success",
            "duration_ms": duration_ms or None,
            "truncated": "true" if truncated else None,
            "ts": ts or None,
        },
    )


@dataclass
class ExecutionResult:
    """Result from code execution."""

    output: str
    error: Optional[str]
    stdout: str
    stderr: str
    return_value: Optional[Any] = None
    tools_called: List[str] = field(default_factory=list)
    # Per-call records ({tool, arguments, success, duration_ms, output|error}),
    # capped via TOOL_CALL_ARG_MAX / TOOL_CALL_OUTPUT_MAX.
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    # Named `tsu_group` sections opened this turn, in the order they opened:
    # {group_id, title, parent_group_id, success, duration_ms, error?}.
    groups: List[Dict[str, Any]] = field(default_factory=list)
    variables_set: Dict[str, str] = field(default_factory=dict)  # name -> "type(size)"
    state_keys: Dict[str, str] = field(default_factory=dict)  # persisted state: name -> "type(size)"
    loaded_skills: Dict[str, str] = field(default_factory=dict)  # name -> content
    unloaded_skills: List[str] = field(default_factory=list)  # names unloaded this turn
    truncated: bool = False
    truncated_to: Optional[str] = None  # Path to full output if truncated
    last_statement_type: Optional[str] = None  # "expr" | "statement" | None (unparsed) - history replay metadata

    def to_xml(self, duration_ms: int = 0, max_output_kb: int = MAX_EXECUTION_OUTPUT_KB) -> str:
        """Render the result as the XML observation replayed to the model."""
        from tsugite.secrets.registry import get_registry

        _mask = get_registry().mask

        # Check for truncation first (before building attrs)
        output, clipped = truncate_observation(_mask(self.output or ""), max_output_kb)
        self.truncated = self.truncated or clipped

        # An explicit return_value() is the other way a turn hands back megabytes;
        # the last-expression path prints instead, so it already rides `output`.
        return_value = None if self.return_value is None else _mask(str(self.return_value))
        if return_value is not None:
            return_value, clipped = truncate_observation(return_value, max_output_kb)
            self.truncated = self.truncated or clipped

        traceback = None
        if self.error and self.stderr:
            traceback = "\n".join(_mask(self.stderr).strip().split("\n")[-10:])

        return build_execution_result(
            output=output,
            error=_mask(self.error) if self.error else None,
            traceback=traceback,
            truncated_to=self.truncated_to,
            variables_set={k: _mask(v) for k, v in self.variables_set.items()} or None,
            state_keys={k: _mask(v) for k, v in self.state_keys.items()} or None,
            return_value=return_value,
            duration_ms=duration_ms,
            truncated=self.truncated,
        ).render()


@runtime_checkable
class Executor(Protocol):
    """Shared turn-execution surface of every executor backend.

    Backends register tools differently (in-process namespace injection vs IPC
    stub generation across a process/host boundary), but they all expose the same
    `set_tools` entry point so the agent stays backend-agnostic.
    """

    def set_tools(self, tools: List[Any], event_bus: Optional[Any] = None) -> None:
        """Register the tools available to executed code for this run."""
        ...

    async def execute(self, code: str) -> ExecutionResult:
        """Run a turn of code and return its result."""
        ...

    async def send_variables(self, variables: Dict[str, Any]) -> None:
        """Register harness-level variables re-injected at the start of every turn."""
        ...

    async def inject_content_blocks(self, blocks: Dict[str, str]) -> None:
        """Replace the content-block variables available to the next turn."""
        ...

    def register_loaded_skill(self, name: str, content: str) -> None:
        """Record a skill loaded during the current turn."""
        ...

    def register_unloaded_skill(self, name: str) -> None:
        """Record a skill unloaded during the current turn."""
        ...


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
    """Summarize a variable's type and size for display, e.g. "dict(3 keys)"."""
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


def run_async_in_sync_context(coro):
    """Run an async coroutine from synchronous user code, handling the event loop.

    In-process tool wrappers are sync but tools are async; when called from inside
    the agent's running loop we run the coroutine in a dedicated thread with its own
    loop (copying contextvars so the interaction backend etc. propagate).
    """
    import concurrent.futures
    import contextvars

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        ctx = contextvars.copy_context()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:

            def run_coro():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()

            return pool.submit(ctx.run, run_coro).result()
    return asyncio.run(coro)


def convert_positional_to_kwargs(tool_obj, args, kwargs):
    """Map positional args onto keyword args using the tool's signature."""
    if not args:
        return

    import inspect

    try:
        param_names = list(inspect.signature(tool_obj.function).parameters.keys())
    except (ValueError, TypeError):
        raise TypeError(
            f"Tool '{tool_obj.name}' must be called with keyword arguments, not positional. "
            f"Example: {tool_obj.name}(param1=value1, param2=value2)"
        )

    for i, arg in enumerate(args):
        if i >= len(param_names):
            raise TypeError(
                f"Tool '{tool_obj.name}' takes at most {len(param_names)} "
                f"positional arguments but {len(args)} were given"
            )
        param_name = param_names[i]
        if param_name in kwargs:
            raise TypeError(f"Tool '{tool_obj.name}' got multiple values for argument '{param_name}'")
        kwargs[param_name] = arg


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
        self._tool_calls: List[Dict[str, Any]] = []
        self._group_stack: List[str] = []
        self._groups: List[Dict[str, Any]] = []
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

        target = workspace_dir
        if target is None and path_context is not None:
            target = path_context.effective_cwd or path_context.workspace_dir
        self._chdir_target_resolved: Optional[Path] = Path(target).resolve() if target is not None else None

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
        ns["tsu_group"] = self._tsu_group

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
        # Content blocks are model-authored; never let one shadow a tool / builtin
        # / state (a block named read_file would replace the callable with a str).
        for name, value in self._content_blocks.items():
            if name in ns or name == "state":
                continue
            ns[name] = value
        ns["state"] = self._state
        return ns

    def _split_code_for_last_expr(self, code: str) -> tuple[str, Optional[str]]:
        """Split code into (setup, last_expression) when the last statement is an
        expression, so it can be eval'd for REPL-style display. Otherwise (code, None)."""
        try:
            tree = ast.parse(code)
            if not tree.body:
                return (code, None)

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

    @staticmethod
    def _classify_last_statement(code: str) -> Optional[str]:
        """Classify the last top-level statement for history replay metadata.

        "expr" when it's a bare expression (a REPL-style trailing value, including a
        ``return_value(...)`` call), "statement" otherwise, None when the code won't
        parse. Lets history replay trust the executed shape instead of regex-scraping
        raw_content.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None
        if not tree.body:
            return None
        return "expr" if isinstance(tree.body[-1], ast.Expr) else "statement"

    def _format_value(self, value: Any) -> str:
        """Format a value for display: pprint for containers, repr for everything else."""
        if isinstance(value, (dict, list, tuple, set)):
            return pprint.pformat(value, width=PPRINT_WIDTH, compact=False)
        return repr(value)

    def _workspace_chdir_target(self) -> Optional[Path]:
        """Target for the per-execute chdir, or None if CWD already matches (or no workspace)."""
        target = self._chdir_target_resolved
        if target is None:
            return None
        try:
            if target == Path.cwd().resolve():
                return None
        except FileNotFoundError:
            pass
        return target

    async def execute(self, code: str) -> ExecutionResult:
        """Execute code using exec().

        Automatically displays the value of the last expression (REPL-like behavior).
        When a workspace is bound, chdir to it under a process-wide lock so raw
        Python file APIs (os.getcwd, open, Path.cwd) resolve against the workspace.
        """
        self._return_value = None
        self._tools_called = []
        self._tool_calls = []
        self._group_stack = []
        self._groups = []
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

            with _locked_chdir(self._workspace_chdir_target()):
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

        # Capture a `state = {...}` rebind, not just in-place mutation. The
        # namespace's `state` may now point at a new object; adopt it so the
        # rebind persists (SubprocessExecutor does the same via namespace.get).
        # Ignore a rebind to a non-dict - keep the prior state rather than crash
        # the save/summarize path.
        rebound = self.namespace.get("state")
        if isinstance(rebound, dict) and rebound is not self._state:
            self._state = rebound

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
            tool_calls=[dict(c) for c in self._tool_calls],
            groups=[dict(g) for g in self._groups],
            variables_set=variables_set,
            state_keys=state_keys,
            loaded_skills=self._loaded_skills_for_turn.copy(),
            unloaded_skills=list(self._unloaded_skills_for_turn),
            last_statement_type=self._classify_last_statement(code),
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
        # Don't let a block shadow a live tool/builtin/state in the current
        # namespace (mirrors _build_turn_namespace's guard).
        for name, value in blocks.items():
            if name in self._tool_functions or name in self._sticky_injections or name == "state":
                continue
            self.namespace[name] = value

    def set_tools(self, tools: List[Any], event_bus: Optional[Any] = None):
        """Wrap tools as in-process namespace functions the executed code can call."""
        if event_bus is not None:
            self.event_bus = event_bus
        wrappers = {t.name: self._make_tool_wrapper(t) for t in tools if t.name not in EXECUTOR_BUILTIN_TOOLS}
        self.register_tools(wrappers)

    @contextmanager
    def _tsu_group(self, title: str):
        """Bracket the block as a named group on the event stream and the execution result."""
        from tsugite.events import ExecutionGroupEndEvent, ExecutionGroupStartEvent

        record: Dict[str, Any] = {
            "group_id": uuid.uuid4().hex[:12],
            "title": _cap_text(str(title), GROUP_TITLE_MAX),
            "parent_group_id": self._current_group(),
        }
        self._groups.append(record)
        if self.event_bus:
            self.event_bus.emit(
                ExecutionGroupStartEvent(
                    group_id=record["group_id"],
                    title=record["title"],
                    parent_group_id=record["parent_group_id"],
                )
            )
        self._group_stack.append(record["group_id"])
        t0 = time.perf_counter()
        try:
            yield
            record["success"] = True
        except BaseException as exc:
            record["success"] = False
            record["error"] = _cap_text(f"{type(exc).__name__}: {exc}", TOOL_CALL_ARG_MAX)
            raise
        finally:
            self._group_stack.pop()
            record["duration_ms"] = int((time.perf_counter() - t0) * 1000)
            if self.event_bus:
                self.event_bus.emit(
                    ExecutionGroupEndEvent(
                        group_id=record["group_id"],
                        success=record.get("success", False),
                        duration_ms=record["duration_ms"],
                        error=record.get("error"),
                    )
                )

    def _current_group(self) -> Optional[str]:
        return self._group_stack[-1] if self._group_stack else None

    def _make_tool_wrapper(self, tool_obj):
        """Build the sync namespace wrapper that calls a tool's async execute()."""
        from tsugite.events import ToolCallEvent, ToolResultEvent

        def tool_wrapper(*args, **kwargs):
            self._tools_called.append(tool_obj.name)
            convert_positional_to_kwargs(tool_obj, args, kwargs)
            # Redact BEFORE _jsonable_call_args: it flattens nested values into
            # capped repr() strings, and a token baked into one of those is no
            # longer reachable by key or path.
            audit_args = redact_sensitive_obj(kwargs, getattr(tool_obj, "sensitive_paths", ()))
            group = self._current_group()
            record: Dict[str, Any] = {"tool": tool_obj.name, "arguments": _jsonable_call_args(audit_args)}
            if group:
                record["group_id"] = group
            self._tool_calls.append(record)
            if self.event_bus:
                self.event_bus.emit(ToolCallEvent(tool_name=tool_obj.name, arguments=audit_args, group_id=group))
            t0 = time.perf_counter()
            try:
                result = run_async_in_sync_context(tool_obj.execute(**kwargs))
                record["success"] = True
                record["duration_ms"] = int((time.perf_counter() - t0) * 1000)
                record["output"] = _cap_text(str(result), TOOL_CALL_OUTPUT_MAX) if result is not None else ""
                if self.event_bus:
                    self.event_bus.emit(
                        ToolResultEvent(
                            tool_name=tool_obj.name,
                            success=True,
                            result_summary=str(result)[:200] if result is not None else "",
                            duration_ms=record["duration_ms"],
                        )
                    )
                return result
            except Exception as exc:
                record["success"] = False
                record["duration_ms"] = int((time.perf_counter() - t0) * 1000)
                record["error"] = _cap_text(str(exc), TOOL_CALL_OUTPUT_MAX)
                if self.event_bus:
                    self.event_bus.emit(
                        ToolResultEvent(
                            tool_name=tool_obj.name,
                            success=False,
                            result_summary=str(exc)[:200],
                            duration_ms=record["duration_ms"],
                        )
                    )
                raise

        tool_wrapper.__name__ = tool_obj.name
        tool_wrapper.__doc__ = tool_obj.description
        if hasattr(tool_obj.function, "__signature__"):
            tool_wrapper.__signature__ = tool_obj.function.__signature__
        if hasattr(tool_obj.function, "__annotations__"):
            tool_wrapper.__annotations__ = tool_obj.function.__annotations__
        return tool_wrapper

    def register_tools(self, tools: Dict[str, Callable[..., Any]]):
        """Register tool functions that should be re-injected into the namespace every turn.

        Called by the agent after tool setup; tool wrappers are not serialized into state.
        """
        self._tool_functions.update(tools)
        self.namespace.update(tools)

    def register_loaded_skill(self, name: str, content: str):
        """Register a skill loaded during this turn. Called by the load_skill() tool."""
        self._loaded_skills_for_turn[name] = content

    def register_unloaded_skill(self, name: str):
        """Record that a skill was unloaded during this execution turn.

        Called by unload_skill() tool so the daemon can drop the name from
        session-level sticky state after the turn completes.
        """
        if name not in self._unloaded_skills_for_turn:
            self._unloaded_skills_for_turn.append(name)
