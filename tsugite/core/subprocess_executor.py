"""Subprocess-based code executor with IPC for tool calls.

Runs LLM-generated Python in a child process. Parent-only tools
(ask_user, spawn_agent, etc.) and final_answer/send_message are
dispatched via IPC. Non-parent-only tools run directly in the child.

State between turns is serialized as JSON (not pickle — pickle is a
sandbox escape via __reduce__).
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from .executor import ExecutionResult

logger = logging.getLogger(__name__)

# Helpers re-exported so the harness can import them without duplicating code
_HARNESS_IMPORTS = textwrap.dedent("""\
    import ast
    import io
    import json
    import os
    import pprint
    import re
    import sys
    import time
    import traceback
""")

_IPC_HELPER = textwrap.dedent("""\
    _REQ_PATH = os.environ["_TSUGITE_REQ_PATH"]
    _RESP_PATH = os.environ["_TSUGITE_RESP_PATH"]
    _req_file = open(_REQ_PATH, "w", buffering=1)  # line-buffered
    _resp_file = open(_RESP_PATH, "r")
    _call_id_counter = 0
    _tools_called = []

    def _ipc_call(msg_type, **kwargs):
        global _call_id_counter
        _call_id_counter += 1
        msg = {"type": msg_type, "call_id": _call_id_counter, **kwargs}
        _req_file.write(json.dumps(msg) + "\\n")
        _req_file.flush()
        if msg_type == "audit":
            return None
        resp_line = _resp_file.readline()
        if not resp_line:
            raise RuntimeError("IPC: parent closed connection")
        resp = json.loads(resp_line)
        if resp.get("error"):
            raise RuntimeError(f"IPC tool error: {resp['error']}")
        return resp.get("result")

    def _ipc_audit(event, tool, **kwargs):
        _ipc_call("audit", event=event, tool=tool, **kwargs)
""")

_FINAL_ANSWER_STUB = textwrap.dedent("""\
    _final_answer_value = None

    def final_answer(*args, **kwargs):
        global _final_answer_value
        if args:
            _final_answer_value = args[0]
        elif kwargs:
            _final_answer_value = next(iter(kwargs.values()))
        _ipc_call("tool_call", name="final_answer", kwargs={"value": _final_answer_value})
""")

_SEND_MESSAGE_STUB = textwrap.dedent("""\
    def send_message(*args, **kwargs):
        if args:
            msg = args[0]
        elif kwargs:
            msg = kwargs.get("message") or next(iter(kwargs.values()))
        else:
            msg = ""
        _ipc_call("tool_call", name="send_message", kwargs={"message": str(msg)})
        return f"Message sent: {msg}"
""")

# Mirrors LocalExecutor._split_code_for_last_expr — duplicated because this runs
# as a string template inside the sandboxed subprocess (can't import from parent).
_SPLIT_CODE_FN = textwrap.dedent("""\
    def _split_code_for_last_expr(code):
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
""")

_TIMED_AUDIT_WRAPPER = textwrap.dedent("""\
    def _timed_audit_call(tool_name, fn, kwargs):
        _tools_called.append(tool_name)
        _ipc_audit("tool_call", tool_name, args=kwargs)
        t0 = time.time()
        try:
            result = fn(**kwargs)
            _ipc_audit("tool_result", tool_name, success=True, duration_ms=int((time.time() - t0) * 1000))
            return result
        except Exception:
            _ipc_audit("tool_result", tool_name, success=False, duration_ms=int((time.time() - t0) * 1000))
            raise

    def _run_maybe_async(fn, kwargs):
        import asyncio as _asyncio
        import inspect as _inspect
        if not _inspect.iscoroutinefunction(fn):
            return fn(**kwargs)
        try:
            loop = _asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(_asyncio.run, fn(**kwargs)).result()
        return _asyncio.run(fn(**kwargs))
""")


def _build_parent_only_tool_stub(name: str) -> str:
    """Generate an IPC stub function for a parent-only tool."""
    return textwrap.dedent(f"""\
    def {name}(**kwargs):
        return _timed_audit_call("{name}", lambda **kw: _ipc_call("tool_call", name="{name}", kwargs=kw), kwargs)
    """)


def _build_local_tool_stub(name: str, module_path: str) -> str:
    """Generate a wrapper for a non-parent-only tool that runs locally in the child."""
    return textwrap.dedent(f"""\
    from {module_path} import {name} as _raw_{name}
    def {name}(**kwargs):
        return _timed_audit_call("{name}", lambda **kw: _run_maybe_async(_raw_{name}, kw), kwargs)
    """)


class SubprocessExecutor:
    """Execute code in a subprocess with IPC for parent-only tools.

    Same interface as LocalExecutor — returns ExecutionResult.
    State persists between turns via JSON serialization.
    """

    def __init__(
        self,
        workspace_dir: Optional[Path] = None,
        event_bus: Optional[Any] = None,
        path_context: Optional[Any] = None,
        sandbox_config: Optional[Any] = None,
    ):
        self.workspace_dir = workspace_dir
        self.event_bus = event_bus
        self.path_context = path_context
        self.sandbox_config = sandbox_config

        self._state: Dict[str, Any] = {}
        self._tools: List[Any] = []
        self._tool_map: Dict[str, Any] = {}
        self._parent_only_tools: set = set()
        self._local_tools: Dict[str, str] = {}  # name -> module_path
        self._final_answer_value = None
        self._tools_called: List[str] = []
        self._loaded_skills_for_turn: Dict[str, str] = {}

        self._tmpdir = tempfile.mkdtemp(prefix="tsugite_sub_")
        self._proxy = None
        self._proxy_socket: Optional[Path] = None

        # Inject path context
        if path_context:
            self._state["WORKSPACE_DIR"] = str(path_context.workspace_dir) if path_context.workspace_dir else None
            self._state["INVOKED_FROM"] = str(path_context.invoked_from) if path_context.invoked_from else None
        else:
            self._state["WORKSPACE_DIR"] = None
            self._state["INVOKED_FROM"] = None

    def set_tools(self, tools: List[Any], event_bus: Optional[Any] = None):
        """Register tools. Called by TsugiteAgent._inject_tools_into_executor.

        Looks up each tool in the tsugite registry to get the real parent_only
        flag and importable module path. Tool objects passed in are wrappers
        from create_tool_from_tsugite() whose __module__ is tsugite.core.tools
        and which don't carry _parent_only — so we can't rely on the wrapper.
        """
        if event_bus:
            self.event_bus = event_bus
        self._tools = tools
        self._tool_map = {t.name: t for t in tools}

        from tsugite.tools import _ensure_tools_loaded, _tools as registry

        _ensure_tools_loaded()

        for t in tools:
            if t.name in ("final_answer", "send_message"):
                continue

            registry_info = registry.get(t.name)
            if registry_info and registry_info.parent_only:
                self._parent_only_tools.add(t.name)
            elif registry_info:
                mod = registry_info.func.__module__
                if mod.startswith("tsugite.tools."):
                    self._local_tools[t.name] = mod
                else:
                    self._parent_only_tools.add(t.name)
            else:
                # Not in registry (MCP tool, custom function) — route via IPC
                self._parent_only_tools.add(t.name)

    def _build_harness(self, code: str) -> str:
        """Generate the Python harness script for a single turn."""
        state_path = os.path.join(self._tmpdir, "state.json")
        result_path = os.path.join(self._tmpdir, "result.json")

        # Build tool stubs
        tool_stubs = []
        for t in self._tools:
            if t.name in ("final_answer", "send_message"):
                continue
            if t.name in self._parent_only_tools:
                tool_stubs.append(_build_parent_only_tool_stub(t.name))
            elif t.name in self._local_tools:
                tool_stubs.append(_build_local_tool_stub(t.name, self._local_tools[t.name]))

        # Escape the user code for embedding
        code_escaped = json.dumps(code)

        harness = f"""\
{_HARNESS_IMPORTS}
{_IPC_HELPER}
{_TIMED_AUDIT_WRAPPER}
{_FINAL_ANSWER_STUB}
{_SEND_MESSAGE_STUB}
{_SPLIT_CODE_FN}

PPRINT_WIDTH = 100
STATE_PATH = {json.dumps(state_path)}
RESULT_PATH = {json.dumps(result_path)}

# Tool stubs
{"".join(tool_stubs)}

# Load state
namespace = {{}}
if os.path.exists(STATE_PATH):
    with open(STATE_PATH, "r") as f:
        namespace = json.load(f)

# Inject builtins into namespace
namespace["final_answer"] = final_answer
namespace["send_message"] = send_message
{self._inject_tool_names_code()}

code = {code_escaped}

# Capture stdout/stderr
stdout_capture = io.StringIO()
stderr_capture = io.StringIO()
old_stdout = sys.stdout
old_stderr = sys.stderr
namespace_before = set(namespace.keys())

try:
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture

    setup_code, last_expr = _split_code_for_last_expr(code)
    if last_expr:
        if setup_code.strip():
            exec(setup_code, namespace)
        val = eval(last_expr, namespace)
        if val is not None:
            formatted = pprint.pformat(val, width=PPRINT_WIDTH, compact=False) if isinstance(val, (dict, list, tuple, set)) else repr(val)
            print(formatted)
    else:
        exec(code, namespace)

    output = stdout_capture.getvalue()
    stderr_output = stderr_capture.getvalue()

    # Track new variables (mirrors _summarize_variable from executor.py — duplicated
    # because this runs as a string template in the sandboxed subprocess)
    namespace_after = set(namespace.keys())
    new_vars = namespace_after - namespace_before
    variables_set = {{}}
    for var_name in new_vars:
        if var_name.startswith("_"):
            continue
        try:
            v = namespace[var_name]
            t = type(v).__name__
            if isinstance(v, dict):
                variables_set[var_name] = f"{{t}}({{len(v)}} keys)"
            elif isinstance(v, (list, tuple, set, frozenset)):
                variables_set[var_name] = f"{{t}}({{len(v)}} items)"
            elif isinstance(v, str):
                variables_set[var_name] = f"{{t}}({{len(v)}} chars)"
            elif isinstance(v, bytes):
                variables_set[var_name] = f"{{t}}({{len(v)}} bytes)"
            else:
                variables_set[var_name] = t
        except Exception:
            variables_set[var_name] = type(namespace[var_name]).__name__

    result = {{
        "output": output,
        "error": None,
        "stdout": output,
        "stderr": stderr_output,
        "final_answer": _final_answer_value,
        "tools_called": _tools_called[:],
        "variables_set": variables_set,
    }}

except Exception as e:
    error_msg = f"{{type(e).__name__}}: {{str(e)}}"
    variables_set = {{}}
    namespace_after = set(namespace.keys())
    new_vars = namespace_after - namespace_before
    for var_name in new_vars:
        if var_name.startswith("_"):
            continue
        try:
            variables_set[var_name] = type(namespace[var_name]).__name__
        except Exception:
            pass

    result = {{
        "output": stdout_capture.getvalue(),
        "error": error_msg,
        "stdout": stdout_capture.getvalue(),
        "stderr": stderr_capture.getvalue() + "\\n" + error_msg,
        "final_answer": None,
        "tools_called": _tools_called[:],
        "variables_set": variables_set,
    }}

finally:
    sys.stdout = old_stdout
    sys.stderr = old_stderr

# Save state (JSON-serializable vars only)
save_state = {{}}
skip_keys = {{"final_answer", "send_message", "__builtins__"}}
for k, v in namespace.items():
    if k.startswith("_") or k in skip_keys:
        continue
    try:
        json.dumps(v)
        save_state[k] = v
    except (TypeError, ValueError, OverflowError):
        pass  # skip non-serializable

# Also skip tool function names
tool_names = {json.dumps([t.name for t in self._tools])}
for tn in tool_names:
    save_state.pop(tn, None)

with open(STATE_PATH, "w") as f:
    json.dump(save_state, f)

# Write result
with open(RESULT_PATH, "w") as f:
    json.dump(result, f)
"""
        return harness

    def _inject_tool_names_code(self) -> str:
        """Generate code to inject tool function references into namespace."""
        lines = []
        for t in self._tools:
            if t.name in ("final_answer", "send_message"):
                continue
            lines.append(f'namespace["{t.name}"] = {t.name}')
        return "\n".join(lines)

    async def execute(self, code: str) -> ExecutionResult:
        """Execute code in a subprocess."""
        self._final_answer_value = None
        self._tools_called = []
        self._loaded_skills_for_turn = {}

        from .executor import LocalExecutor

        safety_error = LocalExecutor._check_code_safety(code)
        if safety_error:
            return ExecutionResult(
                output="",
                error=safety_error,
                stdout="",
                stderr=safety_error,
                final_answer=None,
                tools_called=[],
            )

        harness_code = self._build_harness(code)
        harness_path = os.path.join(self._tmpdir, "harness.py")
        result_path = os.path.join(self._tmpdir, "result.json")

        with open(harness_path, "w") as f:
            f.write(harness_code)

        # Remove stale result file
        if os.path.exists(result_path):
            os.unlink(result_path)

        # Create named FIFOs for IPC (works through bwrap bind mounts)
        req_fifo = os.path.join(self._tmpdir, "req.fifo")
        resp_fifo = os.path.join(self._tmpdir, "resp.fifo")
        for fifo in (req_fifo, resp_fifo):
            if os.path.exists(fifo):
                os.unlink(fifo)
            os.mkfifo(fifo)

        env = os.environ.copy()
        env["_TSUGITE_REQ_PATH"] = req_fifo
        env["_TSUGITE_RESP_PATH"] = resp_fifo

        # Build command: either plain python or bwrap-wrapped
        inner_cmd = [sys.executable, harness_path]
        if self.sandbox_config:
            from .sandbox import BubblewrapSandbox

            # Start proxy on first execution (if network is allowed)
            if not self.sandbox_config.no_network and not self._proxy:
                await self._start_proxy()

            sandbox = BubblewrapSandbox(
                config=self.sandbox_config,
                proxy_socket=self._proxy_socket,
                workspace_dir=Path(self.workspace_dir) if self.workspace_dir else None,
                state_dir=Path(self._tmpdir),
            )
            cmd = sandbox.build_command(inner_cmd)
        else:
            cmd = inner_cmd

        try:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as e:
            return ExecutionResult(
                output="",
                error=f"Failed to spawn subprocess: {e}",
                stdout="",
                stderr=str(e),
            )

        # Open FIFOs in a thread with timeout — FIFO open() blocks until both ends
        # connect, so if the child crashes before opening its ends we'd block forever.
        loop = asyncio.get_running_loop()
        try:
            req_file = await asyncio.wait_for(loop.run_in_executor(None, lambda: open(req_fifo, "r")), timeout=30)
            resp_file = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: open(resp_fifo, "w", buffering=1)), timeout=30
            )
        except (asyncio.TimeoutError, OSError) as e:
            proc.kill()
            proc.wait()
            stderr_output = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            return ExecutionResult(
                output="",
                error=f"Child process failed to connect to IPC FIFOs: {e}",
                stdout="",
                stderr=stderr_output,
            )

        try:
            await self._ipc_loop(proc, req_file, resp_file)
        except Exception as e:
            logger.warning("IPC loop error: %s", e)
        finally:
            try:
                req_file.close()
            except Exception:
                pass
            try:
                resp_file.close()
            except Exception:
                pass

        # Wait for process to finish
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        # Read result
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                result_data = json.load(f)
            return ExecutionResult(
                output=result_data.get("output", ""),
                error=result_data.get("error"),
                stdout=result_data.get("stdout", ""),
                stderr=result_data.get("stderr", ""),
                final_answer=result_data.get("final_answer"),
                tools_called=result_data.get("tools_called", []),
                variables_set=result_data.get("variables_set", {}),
                loaded_skills=self._loaded_skills_for_turn.copy(),
            )
        else:
            # Child crashed before writing result
            stderr_output = ""
            try:
                stderr_output = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            except Exception:
                pass
            return ExecutionResult(
                output="",
                error=f"Subprocess exited with code {proc.returncode}",
                stdout="",
                stderr=stderr_output,
                final_answer=None,
                tools_called=[],
            )

    async def _ipc_loop(self, proc: subprocess.Popen, req_file, resp_file):
        """Read IPC messages from child, dispatch tool calls, write responses."""
        loop = asyncio.get_running_loop()

        while True:
            # Check if child is still alive
            if proc.poll() is not None:
                # Drain remaining messages
                try:
                    while True:
                        line = await asyncio.wait_for(
                            loop.run_in_executor(None, req_file.readline), timeout=0.1
                        )
                        if not line:
                            break
                        await self._handle_ipc_message(json.loads(line), resp_file)
                except (asyncio.TimeoutError, Exception):
                    pass
                break

            try:
                line = await asyncio.wait_for(
                    loop.run_in_executor(None, req_file.readline), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            if not line:
                break

            try:
                msg = json.loads(line.strip())
            except json.JSONDecodeError:
                logger.warning("IPC: invalid JSON from child: %r", line)
                continue

            await self._handle_ipc_message(msg, resp_file)

    async def _handle_ipc_message(self, msg: dict, resp_file):
        """Handle a single IPC message from the child."""
        msg_type = msg.get("type")
        call_id = msg.get("call_id")

        if msg_type == "audit":
            self._handle_audit(msg)
            return

        if msg_type == "tool_call":
            name = msg.get("name")
            kwargs = msg.get("kwargs", {})

            if name == "final_answer":
                self._final_answer_value = kwargs.get("value")
                resp = {"call_id": call_id, "result": None, "error": None}
            elif name == "send_message":
                message = kwargs.get("message", "")
                if self.event_bus:
                    from tsugite.events import InfoEvent

                    self.event_bus.emit(InfoEvent(message=str(message)))
                resp = {"call_id": call_id, "result": f"Message sent: {message}", "error": None}
            elif name in self._tool_map:
                tool = self._tool_map[name]
                try:
                    result = await tool.execute(**kwargs)
                    result = self._ensure_json_serializable(result)
                    resp = {"call_id": call_id, "result": result, "error": None}
                except Exception as e:
                    resp = {"call_id": call_id, "result": None, "error": str(e)}
            else:
                resp = {"call_id": call_id, "result": None, "error": f"Unknown tool: {name}"}

            resp_file.write(json.dumps(resp) + "\n")
            resp_file.flush()

    @staticmethod
    def _ensure_json_serializable(value: Any) -> Any:
        """Return value if JSON-serializable, otherwise return str(value)."""
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)

    def _handle_audit(self, msg: dict):
        """Handle an audit event from the child."""
        if not self.event_bus:
            return

        from tsugite.events import ToolCallEvent, ToolResultEvent

        event_type = msg.get("event")
        tool_name = msg.get("tool", "")

        if event_type == "tool_call":
            self.event_bus.emit(ToolCallEvent(tool_name=tool_name, arguments=msg.get("args", {})))
        elif event_type == "tool_result":
            self.event_bus.emit(
                ToolResultEvent(
                    tool_name=tool_name,
                    success=msg.get("success", True),
                    result_summary="",
                    duration_ms=msg.get("duration_ms"),
                )
            )

    async def _start_proxy(self):
        """Start the HTTP CONNECT proxy for sandbox network access."""
        from .proxy import ConnectProxy

        self._proxy_socket = Path(self._tmpdir) / "proxy.sock"
        allowed = self.sandbox_config.allowed_domains if self.sandbox_config else None
        self._proxy = ConnectProxy(socket_path=self._proxy_socket, allowed_domains=allowed or None)
        await self._proxy.start()
        logger.info("Started proxy at %s (domains=%s)", self._proxy_socket, allowed)

    async def _stop_proxy(self):
        """Stop the proxy if running."""
        if self._proxy:
            await self._proxy.stop()
            self._proxy = None
            self._proxy_socket = None

    async def send_variables(self, variables: Dict[str, Any]):
        """Inject variables into state for next turn."""
        for k, v in variables.items():
            try:
                json.dumps(v)
                self._state[k] = v
            except (TypeError, ValueError, OverflowError):
                logger.warning("SubprocessExecutor: skipping non-serializable variable %r", k)

        # Write state file so next turn picks it up
        state_path = os.path.join(self._tmpdir, "state.json")
        with open(state_path, "w") as f:
            json.dump(self._state, f)

    def register_loaded_skill(self, name: str, content: str):
        """Register a skill loaded during this execution turn."""
        self._loaded_skills_for_turn[name] = content

    def cleanup(self):
        """Stop proxy and remove temp files."""
        import shutil

        if self._proxy:
            try:
                import asyncio

                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._stop_proxy())
                except RuntimeError:
                    asyncio.run(self._stop_proxy())
            except Exception:
                pass

        try:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        except Exception:
            pass
