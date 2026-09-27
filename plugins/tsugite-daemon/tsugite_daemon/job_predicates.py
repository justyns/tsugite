"""Predicate acceptance criteria: parse them out of an AC list and run them.

A predicate AC is a shell command graded by exit status, so unlike the prose ACs
it needs no verifier model. Runs under the job's sandbox when one is configured.
"""

import logging
import os
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from tsugite_daemon.job_store import Job

logger = logging.getLogger(__name__)

_PREDICATE_TIMEOUT_SECONDS = 30

_PREDICATE_STDERR_TRUNCATE = 100

_JOB_COMMAND_OUTPUT_TAIL = 2000


class JobCommandError(Exception):
    def __init__(self, message: str, output: str = ""):
        super().__init__(message)
        self.output = output


def _command_argv(cmd: str, cwd: str, sandbox_override: Optional[dict]) -> Optional[list[str]]:
    """`sh -c cmd`, wrapped in bwrap (no network, `cwd` plus the override's binds) when
    the job is sandboxed. None when it is sandboxed but no sandbox backend is installed."""
    if not sandbox_override:
        return ["sh", "-c", cmd]

    from tsugite.core.sandbox import SandboxConfig, get_sandbox_class

    sandbox_cls = get_sandbox_class()
    if sandbox_cls is None:
        return None
    bwrap = sandbox_cls(
        config=SandboxConfig(
            no_network=True,
            extra_ro_binds=[Path(p) for p in sandbox_override.get("extra_ro_binds", [])],
            extra_rw_binds=[Path(p) for p in sandbox_override.get("extra_rw_binds", [])],
            pass_env=list(sandbox_override.get("pass_env", [])),
        ),
        workspace_dir=Path(cwd),
        state_dir=None,
    )
    return bwrap.build_command(["sh", "-c", cmd])


def run_job_command(cmd: str, cwd: Optional[str], timeout: float, sandbox_override: Optional[dict]) -> None:
    """Run a job's setup/teardown command in `cwd`, raising JobCommandError on failure.

    Output goes to a temp file because a backgrounded child (`npm run dev &`) would
    hold a pipe open and block the call until the timeout.
    """
    if cwd is None:
        raise JobCommandError("job has no working directory")
    argv = _command_argv(cmd, cwd, sandbox_override)
    if argv is None:
        raise JobCommandError("the job is sandboxed but no sandbox backend is installed")
    with tempfile.TemporaryFile() as out:
        try:
            proc = subprocess.Popen(argv, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as e:
            raise JobCommandError(str(e)) from e
        try:
            code = proc.wait(timeout=timeout)
            error = None if code == 0 else f"exit code {code}"
        except subprocess.TimeoutExpired:
            # The command's own process group, so its backgrounded children die with it.
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            error = f"timed out after {timeout:.0f}s"
        if error is None:
            return
        out.seek(max(0, out.seek(0, os.SEEK_END) - 4 * _JOB_COMMAND_OUTPUT_TAIL))
        output = out.read().decode("utf-8", "replace")[-_JOB_COMMAND_OUTPUT_TAIL:]
    raise JobCommandError(error, output)


def _parse_ac_predicate(text: str) -> Optional[dict]:
    """Recognise a predicate AC prefix and return a structured dict.

    Returns None for free-text ACs (which then go to the LLM verifier as today).
    Recognised prefixes:
      `exit_code:<cmd>`       → {kind: "exit_code", cmd, expected: 0}
      `exit_code:<cmd>:<n>`   → {kind: "exit_code", cmd, expected: <n>}
      `cmd:<command>`         → {kind: "cmd", cmd} (sugar for exit_code:<cmd>:0)
      `file_exists:<path>`    → {kind: "file_exists", path}
    """
    if not text:
        return None
    s = text.strip()
    if not s:
        return None
    if s.startswith("exit_code:"):
        body = s[len("exit_code:") :].strip()
        if not body:
            return None
        # The exit code suffix is the trailing `:<int>`. Tolerate command strings
        # that themselves contain colons by splitting from the right and checking
        # if the last segment parses as an int.
        last_colon = body.rfind(":")
        if last_colon != -1:
            tail = body[last_colon + 1 :].strip()
            try:
                expected = int(tail)
                cmd = body[:last_colon].strip()
                if cmd:
                    return {"kind": "exit_code", "cmd": cmd, "expected": expected}
            except ValueError:
                pass
        return {"kind": "exit_code", "cmd": body, "expected": 0}
    if s.startswith("cmd:"):
        body = s[len("cmd:") :].strip()
        if not body:
            return None
        return {"kind": "cmd", "cmd": body}
    if s.startswith("file_exists:"):
        body = s[len("file_exists:") :].strip()
        if not body:
            return None
        return {"kind": "file_exists", "path": body}
    return None


def partition_acs(acs: list[str]) -> tuple[list[dict], list[dict]]:
    """Split an AC list into (predicates, prose), preserving original indices.

    Each predicate entry: {ac_index, ac_text, predicate}.
    Each prose entry:     {ac_index, ac_text}.
    Original ac_index is preserved so predicate and LLM results can be merged
    back into a single ac_results list spanning the full AC index space.
    """
    predicates: list[dict] = []
    prose: list[dict] = []
    for i, ac in enumerate(acs or []):
        parsed = _parse_ac_predicate(ac)
        if parsed is not None:
            predicates.append({"ac_index": i, "ac_text": ac, "predicate": parsed})
        else:
            prose.append({"ac_index": i, "ac_text": ac})
    return predicates, prose


def _resolve_predicate_cwd(job: Job) -> Optional[str]:
    """Pick cwd for predicate evaluation: worktree_path > repo > workspace anchor.

    The workspace fallback keeps non-repo jobs' `file_exists:`/`cmd:` predicates
    resolving against the directory the worker wrote into, not the daemon CWD.
    """
    return job.worktree_path or job.repo or job.workspace_path


def _evaluate_predicate(
    predicate: dict,
    *,
    cwd: Optional[str],
    ac_index: int,
    ac_text: str,
    attempt: int,
    sandbox_override: Optional[dict] = None,
) -> dict:
    """Run a predicate locally and return an ac_results entry.

    `exit_code:` / `cmd:` predicates shell out. When the job is sandboxed
    (sandbox_override set, resolved from the agent's config or an inheriting
    parent), the command runs inside bubblewrap - filesystem-isolated to the
    predicate cwd (the worktree) and with no network - so a sandboxed agent's
    predicate ACs can't execute outside the sandbox. Otherwise they run with
    `shell=True` against the worktree, same surface as the worker session.
    """

    def verdict(passed: bool, reason: str) -> dict:
        return {
            "ac_index": ac_index,
            "ac_text": ac_text,
            "pass": passed,
            "reason": reason,
            "attempt": attempt,
        }

    kind = predicate.get("kind")
    try:
        if kind == "file_exists":
            path = predicate.get("path", "")
            p = Path(path)
            if not p.is_absolute() and cwd:
                p = Path(cwd) / path
            if p.exists():
                return verdict(True, "path exists")
            return verdict(False, f"path does not exist: {path}")
        if kind in ("exit_code", "cmd"):
            cmd = predicate.get("cmd", "")
            expected = predicate.get("expected", 0) if kind == "exit_code" else 0
            if cwd is None:
                logger.warning(
                    "Predicate eval has no cwd (job has no worktree, repo, or workspace anchor); "
                    "refusing to run '%s' in the daemon's cwd - marking criterion unmet",
                    cmd,
                )
                return verdict(
                    False, "no working directory for command predicate (job has no worktree, repo, or workspace anchor)"
                )
            argv = _command_argv(cmd, cwd, sandbox_override)
            if argv is None:
                return verdict(False, "command predicate requires a sandbox but no backend is installed")
            completed = subprocess.run(argv, capture_output=True, timeout=_PREDICATE_TIMEOUT_SECONDS, cwd=cwd)
            if completed.returncode == expected:
                return verdict(True, f"exit code {completed.returncode}")
            stderr = (completed.stderr or b"").decode("utf-8", "replace").strip()
            reason = f"exited with code {completed.returncode} (expected {expected})"
            if stderr:
                reason += f"; stderr: {stderr[:_PREDICATE_STDERR_TRUNCATE]}"
            return verdict(False, reason)
        # Unknown predicate kind - defensive; partition_acs only emits the
        # three above. Treat as a fail rather than silently passing.
        return verdict(False, f"unknown predicate kind: {kind!r}")
    except subprocess.TimeoutExpired:
        return verdict(False, "timeout")
    except Exception as e:
        return verdict(False, f"evaluation error: {e}")
