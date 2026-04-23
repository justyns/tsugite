"""Subprocess-based code executor with IPC for tool calls.

Runs LLM-generated Python in a child process. Parent-only tools
(ask_user, spawn_agent, etc.) and final_answer/send_message are
dispatched via IPC. Non-parent-only tools run directly in the child.

Each turn runs in a fresh namespace. Values assigned to the injected
`state` dict persist across turns via a per-session JSON file; all
other bindings are discarded at turn end. JSON (not pickle) is used
because unpickling attacker-controlled state would be a sandbox escape
via __reduce__.
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

from .executor import EXECUTOR_BUILTIN_TOOLS, ExecutionResult
from .state import load_state, save_state

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

_RETURN_VALUE_STUB = textwrap.dedent("""\
    _return_value = None

    def return_value(*args, **kwargs):
        global _return_value
        if args:
            _return_value = args[0]
        elif kwargs:
            _return_value = next(iter(kwargs.values()))
        _ipc_call("tool_call", name="return_value", kwargs={"value": _return_value})

    # Backward-compat alias for older agent markdown files.
    final_answer = return_value
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

_REACT_TO_MESSAGE_STUB = textwrap.dedent("""\
    def react_to_message(emoji="", message_id=None):
        _ipc_call("tool_call", name="react_to_message", kwargs={"emoji": str(emoji), "message_id": message_id})
        return f"Reacted with {emoji}"
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
        state_path: Optional[Path] = None,
        session_id: Optional[str] = None,
    ):
        self.workspace_dir = workspace_dir
        self.event_bus = event_bus
        self.path_context = path_context
        self.sandbox_config = sandbox_config

        self._state_path = state_path
        self._session_id = session_id
        self._state: Dict[str, Any] = load_state(state_path) if state_path else {}
        self._sticky_injections: Dict[str, Any] = {}
        self._tools: List[Any] = []
        self._tool_map: Dict[str, Any] = {}
        self._parent_only_tools: set = set()
        self._local_tools: Dict[str, str] = {}  # name -> module_path
        self._return_value = None
        self._tools_called: List[str] = []
        self._loaded_skills_for_turn: Dict[str, str] = {}
        self._unloaded_skills_for_turn: List[str] = []

        self._tmpdir = tempfile.mkdtemp(prefix="tsugite_sub_")
        self._proxy = None
        self._proxy_socket: Optional[Path] = None

        if path_context:
            self._sticky_injections["WORKSPACE_DIR"] = (
                str(path_context.workspace_dir) if path_context.workspace_dir else None
            )
            self._sticky_injections["INVOKED_FROM"] = (
                str(path_context.invoked_from) if path_context.invoked_from else None
            )
        else:
            self._sticky_injections["WORKSPACE_DIR"] = None
            self._sticky_injections["INVOKED_FROM"] = None

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

        from tsugite.tools import _ensure_tools_loaded
        from tsugite.tools import _tools as registry

        _ensure_tools_loaded()

        for t in tools:
            if t.name in EXECUTOR_BUILTIN_TOOLS:
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
        injections_path = os.path.join(self._tmpdir, "injections.json")
        result_path = os.path.join(self._tmpdir, "result.json")

        # Build tool stubs
        tool_stubs = []
        for t in self._tools:
            if t.name in EXECUTOR_BUILTIN_TOOLS:
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
{_RETURN_VALUE_STUB}
{_SEND_MESSAGE_STUB}
{_REACT_TO_MESSAGE_STUB}
{_SPLIT_CODE_FN}

PPRINT_WIDTH = 100
STATE_PATH = {json.dumps(state_path)}
INJECTIONS_PATH = {json.dumps(injections_path)}
RESULT_PATH = {json.dumps(result_path)}

# Tool stubs
{"".join(tool_stubs)}

namespace = {{}}
namespace["return_value"] = return_value
namespace["final_answer"] = return_value  # backward-compat alias
namespace["send_message"] = send_message
namespace["react_to_message"] = react_to_message
def _blocked_open(*args, **kwargs):
    raise RuntimeError(
        "open() is not available. Use the provided tools instead:\\n"
        "  - read_file(path) to read file contents\\n"
        "  - write_file(path, content) to write to files"
    )
namespace["open"] = _blocked_open
{self._inject_tool_names_code()}

if os.path.exists(INJECTIONS_PATH):
    with open(INJECTIONS_PATH, "r") as f:
        namespace.update(json.load(f))

# Load content blocks from files (consumed once per turn)
_cb_manifest = os.path.join(os.path.dirname(STATE_PATH), "content_blocks.json")
if os.path.exists(_cb_manifest):
    with open(_cb_manifest, "r") as f:
        for _cb_name, _cb_path in json.load(f).items():
            with open(_cb_path, "r") as _cb_f:
                namespace[_cb_name] = _cb_f.read()
    os.remove(_cb_manifest)

_state_data = {{}}
if os.path.exists(STATE_PATH):
    with open(STATE_PATH, "r") as f:
        _state_data = json.load(f)
namespace["state"] = _state_data

code = {code_escaped}

stdout_capture = io.StringIO()
stderr_capture = io.StringIO()
old_stdout = sys.stdout
old_stderr = sys.stderr
namespace_before = set(namespace.keys())

def _summarize(v):
    t = type(v).__name__
    if isinstance(v, dict):
        return t + "(" + str(len(v)) + " keys)"
    if isinstance(v, (list, tuple, set, frozenset)):
        return t + "(" + str(len(v)) + " items)"
    if isinstance(v, str):
        return t + "(" + str(len(v)) + " chars)"
    if isinstance(v, bytes):
        return t + "(" + str(len(v)) + " bytes)"
    return t

def _is_json_safe(v):
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError, OverflowError):
        return False

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
    exec_error = None

except Exception as e:
    output = stdout_capture.getvalue()
    stderr_output = stderr_capture.getvalue()
    exec_error = f"{{type(e).__name__}}: {{str(e)}}"
finally:
    sys.stdout = old_stdout
    sys.stderr = old_stderr

namespace_after = set(namespace.keys())
new_vars = namespace_after - namespace_before
variables_set = {{}}
for var_name in new_vars:
    if var_name.startswith("_"):
        continue
    try:
        variables_set[var_name] = _summarize(namespace[var_name])
    except Exception:
        try:
            variables_set[var_name] = type(namespace[var_name]).__name__
        except Exception:
            pass

state_final = namespace.get("state", {{}})
if not isinstance(state_final, dict):
    state_final = dict(state_final) if hasattr(state_final, "keys") else {{}}

state_keys = {{k: _summarize(v) for k, v in state_final.items()}}

save_error = None
try:
    serialized = json.dumps(state_final)
except (TypeError, ValueError, OverflowError) as err:
    bad_key = next((k for k, v in state_final.items() if not _is_json_safe(v)), "?")
    save_error = "StateSerializationError: state[" + repr(bad_key) + "] is not JSON-serializable: " + str(err)

if save_error is None:
    with open(STATE_PATH, "w") as f:
        f.write(serialized)

final_error = exec_error
if save_error is not None:
    final_error = (exec_error + "\\n" + save_error) if exec_error else save_error
    stderr_output = (stderr_output + "\\n" + save_error) if stderr_output else save_error

result = {{
    "output": output if exec_error is None else stdout_capture.getvalue(),
    "error": final_error,
    "stdout": output,
    "stderr": stderr_output if exec_error is None else (stderr_output + "\\n" + exec_error),
    "return_value": _return_value if exec_error is None else None,
    "tools_called": _tools_called[:],
    "variables_set": variables_set,
    "state_keys": state_keys,
}}

with open(RESULT_PATH, "w") as f:
    json.dump(result, f)
"""
        return harness

    def _inject_tool_names_code(self) -> str:
        """Generate code to inject tool function references into namespace."""
        lines = []
        for t in self._tools:
            if t.name in EXECUTOR_BUILTIN_TOOLS:
                continue
            lines.append(f'namespace["{t.name}"] = {t.name}')
        return "\n".join(lines)

    async def execute(self, code: str) -> ExecutionResult:
        """Execute code in a subprocess."""
        self._return_value = None
        self._tools_called = []
        self._loaded_skills_for_turn = {}
        self._unloaded_skills_for_turn = []

        harness_code = self._build_harness(code)
        harness_path = os.path.join(self._tmpdir, "harness.py")
        result_path = os.path.join(self._tmpdir, "result.json")
        state_path = os.path.join(self._tmpdir, "state.json")
        injections_path = os.path.join(self._tmpdir, "injections.json")

        with open(harness_path, "w") as f:
            f.write(harness_code)

        # Shuttle persistent state into the tmpdir where the (possibly sandboxed) child can read it.
        with open(state_path, "w") as f:
            json.dump(self._state, f)

        with open(injections_path, "w") as f:
            json.dump(self._sticky_injections, f)

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
                cwd=str(self.workspace_dir) if self.workspace_dir else None,
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

            # Pull updated state back from the child and persist to the real path.
            try:
                with open(state_path, "r") as f:
                    self._state = json.load(f)
                if self._state_path is not None:
                    save_state(self._state, self._state_path, session_id=self._session_id)
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning("Failed to reload state from subprocess: %s", e)

            return ExecutionResult(
                output=result_data.get("output", ""),
                error=result_data.get("error"),
                stdout=result_data.get("stdout", ""),
                stderr=result_data.get("stderr", ""),
                return_value=result_data.get("return_value"),
                tools_called=result_data.get("tools_called", []),
                variables_set=result_data.get("variables_set", {}),
                state_keys=result_data.get("state_keys", {}),
                loaded_skills=self._loaded_skills_for_turn.copy(),
                unloaded_skills=list(self._unloaded_skills_for_turn),
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
                return_value=None,
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
                        line = await asyncio.wait_for(loop.run_in_executor(None, req_file.readline), timeout=0.1)
                        if not line:
                            break
                        await self._handle_ipc_message(json.loads(line), resp_file)
                except (asyncio.TimeoutError, Exception):
                    pass
                break

            try:
                line = await asyncio.wait_for(loop.run_in_executor(None, req_file.readline), timeout=1.0)
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

            if name == "return_value":
                self._return_value = kwargs.get("value")
                resp = {"call_id": call_id, "result": None, "error": None}
            elif name == "send_message":
                message = kwargs.get("message", "")
                if self.event_bus:
                    from tsugite.events import InfoEvent

                    self.event_bus.emit(InfoEvent(message=str(message)))
                resp = {"call_id": call_id, "result": f"Message sent: {message}", "error": None}
            elif name == "react_to_message":
                emoji = kwargs.get("emoji", "")
                message_id = kwargs.get("message_id")
                if self.event_bus:
                    from tsugite.events import ReactionEvent

                    self.event_bus.emit(ReactionEvent(emoji=str(emoji), message_id=message_id))
                resp = {"call_id": call_id, "result": f"Reacted with {emoji}", "error": None}
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
        """Register harness-level variables that are re-injected at the start of every turn."""
        for k, v in variables.items():
            try:
                json.dumps(v)
                self._sticky_injections[k] = v
            except (TypeError, ValueError, OverflowError):
                logger.warning("SubprocessExecutor: skipping non-serializable variable %r", k)

    async def inject_content_blocks(self, blocks: Dict[str, str]):
        """Inject content block variables for the next turn.

        Writes blocks to content-addressed temp files and a manifest
        so the harness can load them into the subprocess namespace.
        """
        from .content_blocks import write_content_blocks_to_files

        block_files = write_content_blocks_to_files(blocks, self._tmpdir)
        manifest_path = os.path.join(self._tmpdir, "content_blocks.json")
        with open(manifest_path, "w") as f:
            json.dump(block_files, f)

    def register_loaded_skill(self, name: str, content: str):
        """Register a skill loaded during this execution turn."""
        self._loaded_skills_for_turn[name] = content

    def register_unloaded_skill(self, name: str):
        """Record that a skill was unloaded during this execution turn."""
        if name not in self._unloaded_skills_for_turn:
            self._unloaded_skills_for_turn.append(name)

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
