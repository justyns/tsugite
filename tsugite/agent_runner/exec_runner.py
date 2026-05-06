"""Synchronous one-shot Python executor for `<!-- tsu:exec -->` directives.

This is a deliberate parallel path to `tsugite.core.subprocess_executor.SubprocessExecutor`.
Both run user Python in a subprocess, but they differ:

  - `SubprocessExecutor` is async, turn-stateful, blocks `open()`, forces tool usage,
    and manages a network proxy when sandboxed. It's designed for the LLM agent loop
    where each turn is a long-running session.
  - `run_python_block` (this module) is synchronous, one-shot, allows plain Python
    (`open`, `os.listdir`, etc.), and is invoked at agent-prep time before the LLM runs.

Sandbox plumbing is deferred: passing `sandbox_config` raises `NotImplementedError`.
The async path's bubblewrap+proxy lifecycle does not translate cleanly to the
synchronous one-shot model; a future shared `build_python_subprocess_command` helper
would converge both paths. Any sandbox-related change must update both call sites.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ExecBlockResult:
    """Result of running a single tsu:exec block."""

    return_value: Any
    stdout: str
    stderr: str
    error: Optional[str]


_HARNESS = textwrap.dedent("""\
    import ast
    import json
    import sys
    import traceback
    from datetime import date, datetime
    from pathlib import Path

    with open(sys.argv[1]) as _f:
        _payload = json.load(_f)
    _locals = _payload["locals"]
    _user_code = _payload["code"]

    _return_value = None

    def return_value(*args, **kwargs):
        global _return_value
        if args:
            _return_value = args[0]
        elif kwargs:
            _return_value = next(iter(kwargs.values()))


    def _split_last_expr(code):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, None
        if not tree.body:
            return code, None
        last = tree.body[-1]
        if not isinstance(last, ast.Expr):
            return code, None
        if len(tree.body) == 1:
            return "", ast.unparse(last.value)
        setup_tree = ast.Module(body=tree.body[:-1], type_ignores=[])
        return ast.unparse(setup_tree), ast.unparse(last.value)


    def _coerce(v):
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float, str)):
            return v
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, (datetime, date)):
            return v.isoformat()
        if isinstance(v, (set, frozenset)):
            return sorted(_coerce(x) for x in v)
        if isinstance(v, dict):
            return {str(k): _coerce(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [_coerce(x) for x in v]
        raise RuntimeError(
            f"tsu:exec return_value type {type(v).__name__!r} is not JSON-serializable "
            "and has no built-in coercion - convert it explicitly (e.g. str(...)) before returning"
        )


    _ns = dict(_locals)
    _ns["return_value"] = return_value

    _err = None
    try:
        _setup, _last = _split_last_expr(_user_code)
        if _last is not None:
            if _setup.strip():
                exec(_setup, _ns)
            _v = eval(_last, _ns)
            if _return_value is None and _v is not None:
                _return_value = _v
        else:
            exec(_user_code, _ns)
        _rv = _coerce(_return_value)
    except Exception:
        _err = traceback.format_exc()
        _rv = None

    with open(sys.argv[2], "w") as _f:
        json.dump({"return_value": _rv, "error": _err}, _f)
""")


def run_python_block(
    code: str,
    locals_dict: Dict[str, Any],
    timeout: int = 30,
    continue_on_error: bool = False,
    sandbox_config: Optional[Any] = None,
    workspace_dir: Optional[str] = None,
) -> ExecBlockResult:
    """Run a Python block synchronously in a one-shot subprocess.

    Args:
        code: Python source code (the directive body, opaque to Jinja).
        locals_dict: Vars to inject as Python locals (must be JSON-serializable;
            non-serializable entries are filtered out with a warning).
        timeout: Wall-clock timeout in seconds.
        continue_on_error: If True, exceptions are captured to result.error and
            return_value is None instead of re-raising.
        sandbox_config: Optional SandboxConfig; when set, blocks run in bwrap.
        workspace_dir: Optional workspace path for sandbox bind mounts.

    Raises:
        RuntimeError: On timeout, on hard execution failure when not continuing,
            or on non-coercible return-value types.
    """
    safe_locals = _filter_json_safe(locals_dict)

    with tempfile.TemporaryDirectory(prefix="tsu_exec_") as tmpdir:
        harness_path = os.path.join(tmpdir, "harness.py")
        payload_path = os.path.join(tmpdir, "payload.json")
        result_path = os.path.join(tmpdir, "result.json")

        with open(harness_path, "w") as f:
            f.write(_HARNESS)
        with open(payload_path, "w") as f:
            json.dump({"locals": safe_locals, "code": code}, f)

        cmd = _build_command(harness_path, payload_path, result_path, sandbox_config, workspace_dir)

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"tsu:exec block timed out after {timeout}s") from e

        result_data: Dict[str, Any] = {}
        if os.path.exists(result_path):
            with open(result_path) as f:
                result_data = json.load(f)

        # Subprocess crashed before writing the result file (e.g. coercion error
        # raised inside the harness, or harness import error).
        if not result_data and proc.returncode != 0:
            err = proc.stderr.strip() or f"subprocess exited {proc.returncode}"
            if continue_on_error:
                return ExecBlockResult(return_value=None, stdout=proc.stdout, stderr=proc.stderr, error=err)
            raise RuntimeError(f"tsu:exec block failed: {err}")

        error = result_data.get("error")
        if error and not continue_on_error:
            raise RuntimeError(f"tsu:exec block raised:\n{error}")

        return ExecBlockResult(
            return_value=result_data.get("return_value"),
            stdout=proc.stdout,
            stderr=proc.stderr,
            error=error,
        )


def _filter_json_safe(d: Dict[str, Any]) -> Dict[str, Any]:
    """Drop entries that can't survive a JSON round-trip.

    Locals from prior pipeline stages may include non-serializable values (e.g. a
    Path returned by a tool). Silently skip those rather than blocking the whole
    block - the user's Python can re-derive them if needed.
    """
    safe = {}
    for k, v in d.items():
        try:
            json.dumps(v)
            safe[k] = v
        except (TypeError, ValueError, OverflowError):
            continue
    return safe


def _build_command(
    harness_path: str,
    payload_path: str,
    result_path: str,
    sandbox_config: Optional[Any],
    workspace_dir: Optional[str],
) -> list[str]:
    if sandbox_config is None:
        return [sys.executable, "-u", harness_path, payload_path, result_path]
    raise NotImplementedError(
        "sandbox_config support for tsu:exec is not implemented yet - the existing async "
        "SubprocessExecutor path manages a network proxy that doesn't translate cleanly to "
        "this synchronous one-shot path. Track in a follow-up."
    )
