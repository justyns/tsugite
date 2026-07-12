"""CCExecutor: the non-agent job executor that drives an interactive `claude`.

Implements the orchestrator's executor contract (async start/cancel) and reports
outcomes via complete_worker/fail_worker (called from the hook route in adapter.py
and, on an unexpected PTY exit, from here).

DriveState is in-memory per active job; it dies with the daemon (orphaned jobs are
recovered as ERRORED by the existing restart machinery - no new persistence).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import shutil
import tempfile
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tsugite_cc_driver.hooks import build_initial_prompt
from tsugite_cc_driver.settings import build_settings, cleanup, write_run_settings

logger = logging.getLogger(__name__)


@dataclass
class DriveState:
    """In-memory driving state for one active cc job."""

    job_id: str
    token: str
    terminal_id: Optional[str] = None
    cc_session_id: Optional[str] = None
    consecutive_continues: int = 0
    # True while an un-actioned permission prompt is (probably) on screen: set
    # on a permission_prompt Notification, cleared on the next Stop (claude
    # finishing a turn proves the prompt was answered). Drives the UI's
    # persistent needs-your-input marker via needs_attention/attention_cleared.
    attention_flagged: bool = False


class DriveStateStore:
    """Keyed by job_id, with a token index for the public hook route."""

    def __init__(self):
        self._by_job: dict[str, DriveState] = {}
        self._token_to_job: dict[str, str] = {}

    def create(self, job_id: str, token: str) -> DriveState:
        state = DriveState(job_id=job_id, token=token)
        self._by_job[job_id] = state
        self._token_to_job[token] = job_id
        return state

    def get(self, job_id: str) -> Optional[DriveState]:
        return self._by_job.get(job_id)

    def by_token(self, token: str) -> Optional[DriveState]:
        job_id = self._token_to_job.get(token)
        return self._by_job.get(job_id) if job_id else None

    def remove(self, job_id: str) -> None:
        state = self._by_job.pop(job_id, None)
        if state is not None:
            self._token_to_job.pop(state.token, None)


_EXIT_CODE_HINTS = {
    126: "claude binary is not executable",
    127: "claude binary not found (PATH, or missing from the sandbox binds)",
    130: "interrupted (SIGINT)",
}

# ANSI escape sequences (CSI colors/cursor, OSC title) to strip from a captured tail.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


def describe_exit(exit_code) -> str:
    hint = _EXIT_CODE_HINTS.get(exit_code)
    return f"code {exit_code} - {hint}" if hint else f"code {exit_code}"


def pty_tail(buffer: bytes, *, max_bytes: int = 4096, max_lines: int = 40) -> Optional[str]:
    """ANSI-stripped, blank-trimmed tail of a PTY buffer, or None if empty."""
    text = bytes(buffer)[-max_bytes:].decode("utf-8", errors="replace")
    lines = [ln.rstrip() for ln in _ANSI_RE.sub("", text).splitlines() if ln.strip()]
    return "\n".join(lines[-max_lines:]) or None


def build_claude_command(
    claude_binary: str,
    settings_path: str,
    permission_mode: str,
    prompt: str,
    *,
    model: Optional[str] = None,
    resume_session_id: Optional[str] = None,
    effort: Optional[str] = None,
    ax_screen_reader: bool = False,
) -> str:
    """Build the shell command for spawn_terminal (which wraps it in `sh -c`).

    Everything is shlex.quoted so a goal/followup containing quotes or newlines
    can't break out of the argument.
    """
    parts = [claude_binary, "--settings", str(settings_path), "--permission-mode", permission_mode]
    if model:
        parts += ["--model", model]
    if effort:
        parts += ["--effort", effort]
    if ax_screen_reader:
        parts.append("--ax-screen-reader")
    if resume_session_id:
        parts += ["--resume", resume_session_id]
    parts.append(prompt)
    return shlex.join(parts)


# Serializes the ~/.claude.json read-modify-write so two concurrent trust
# provisions can't clobber each other's merge.
_trust_config_lock = threading.Lock()


def _default_trust_config_path() -> Path:
    base = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~")
    return Path(base) / ".claude.json"


def is_workspace_trusted(cwd, config_path=None) -> bool:
    """Whether Claude Code already trusts `cwd`, so an unattended spawn won't hang
    on the fresh-cwd "Is this a project you trust?" dialog.

    A dir is trusted when itself OR any ancestor has
    projects["<abs path>"].hasTrustDialogAccepted == true in the config Claude
    reads. Claude Code treats a subdir of a trusted project as trusted (verified
    against claude 2.1.207) - a `--repo` Job's worker runs in a fresh worktree at
    <repo>/.tsugite-jobs/<id> whose exact path is never trusted, so the ancestor
    check is what lets those jobs launch.

    config_path defaults to <CLAUDE_CONFIG_DIR>/.claude.json (or ~/.claude.json).
    Tolerant: a missing / unreadable / malformed config means "not trusted".
    """
    if config_path is None:
        config_path = _default_trust_config_path()
    try:
        data = json.loads(Path(config_path).read_text())
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    projects = data.get("projects") or {}
    resolved = Path(cwd).resolve()
    for path in (resolved, *resolved.parents):
        project = projects.get(str(path))
        if isinstance(project, dict) and project.get("hasTrustDialogAccepted") is True:
            return True
    return False


def ensure_workspace_trusted(cwd, config_path=None) -> bool:
    """Provision Claude Code trust for `cwd` by setting
    projects["<abs cwd>"].hasTrustDialogAccepted = true in the config Claude
    reads, so an unattended spawn skips the trust dialog (a minimal entry with
    just this flag is enough - verified against claude 2.1.207). Returns True on
    success (including when already trusted), False if the config couldn't be
    provisioned.

    Read-modify-write, preserving every other key and project entry, then an
    atomic replace so a crash mid-write can't truncate the config. A malformed
    existing config is left untouched (we won't destroy recoverable user data);
    a missing config is created. Serialized across concurrent cc spawns via
    _trust_config_lock. claude itself also writes this file at runtime; the
    provision happens before we spawn claude for this job, so it can't race our
    own worker.
    """
    if config_path is None:
        config_path = _default_trust_config_path()
    config_path = Path(config_path)
    resolved = str(Path(cwd).resolve())
    with _trust_config_lock:
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text())
            except Exception:
                logger.warning("cc-driver: %s is not valid JSON; not provisioning trust for %s", config_path, cwd)
                return False
            if not isinstance(data, dict):
                logger.warning("cc-driver: %s is not a JSON object; not provisioning trust for %s", config_path, cwd)
                return False
        else:
            data = {}
        projects = data.setdefault("projects", {})
        if not isinstance(projects, dict):
            logger.warning("cc-driver: %s 'projects' is not an object; not provisioning trust", config_path)
            return False
        entry = projects.setdefault(resolved, {})
        if not isinstance(entry, dict):
            entry = projects[resolved] = {}
        entry["hasTrustDialogAccepted"] = True
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(config_path.parent), prefix=".claude.json.")
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            os.replace(tmp, config_path)
        except Exception:
            logger.exception("cc-driver: failed to write trust for %s into %s", cwd, config_path)
            return False
    logger.info("cc-driver: provisioned Claude Code trust for %s", resolved)
    return True


def trust_provision_target(cwd) -> Path:
    """The path to provision trust for when `cwd` needs it: the main repo root
    when cwd is a linked git worktree living under it (tsugite's --repo
    worktrees are <repo>/.tsugite-jobs/<id>), else cwd itself.

    Trusting the repo root instead of the ephemeral worktree keeps
    ~/.claude.json from accumulating one dead entry per --repo job - the
    ancestor rule then covers every future worktree under that root. An
    external worktree (outside the repo root) still gets its own entry, since
    the ancestor rule wouldn't reach it.

    A linked worktree's `.git` is a *file* containing
    `gitdir: <repo>/.git/worktrees/<name>`; anything else means cwd is not a
    worktree.
    """
    cwd = Path(cwd).resolve()
    gitfile = cwd / ".git"
    if not gitfile.is_file():
        return cwd
    try:
        m = re.match(r"gitdir:\s*(.+)", gitfile.read_text().strip())
    except OSError:
        return cwd
    if m is None:
        return cwd
    gitdir = Path(m.group(1))
    if not gitdir.is_absolute():
        gitdir = cwd / gitdir
    gitdir = gitdir.resolve()
    if gitdir.parent.name != "worktrees" or gitdir.parent.parent.name != ".git":
        return cwd
    root = gitdir.parent.parent.parent
    return root if cwd.is_relative_to(root) else cwd


def build_sandbox_ctx(sandbox: bool, cwd: Optional[str], *, claude_binary: str = "claude", settings_dir=None):
    """Sandbox policy for a cc job: None (unsandboxed) unless `sandbox` is set.

    When set: filesystem isolation to cwd, network ON (no_network=False - the
    driven claude needs the API), and ~/.claude bound rw so credentials refresh.

    bwrap does --clearenv and mounts only a fixed set of dirs (/usr /lib /bin
    /sbin, the venv, the workspace, and our extra binds), so we MUST also bind
    everything the driven claude touches by absolute path - each one, if missing
    from the jail, is a hard failure at launch:
    - the claude binary's dirs, or PATH points at a file that isn't there and
      claude dies with `claude: command not found` (exit 127);
    - the per-job settings dir holding settings.json, or `--settings <path>`
      errors with "Settings file not found";
    - ~/.claude.json, or claude can't see the workspace-trusted flag and hangs
      on the trust prompt inside the jail (the exact hang the pre-check prevents).
      Bound read-WRITE: claude persists per-run session metadata (numStartups,
      lastCost) into it on shutdown and errors on a read-only mount.
    """
    if not sandbox:
        return None
    from tsugite.agent_runner.helpers import SandboxContext

    ro: list[Path] = []
    resolved = shutil.which(claude_binary)
    if resolved:
        # The PATH entry is usually a symlink into a per-version install dir; bind
        # both so the symlink resolves AND the real files are present in the jail.
        for d in (Path(resolved).parent, Path(os.path.realpath(resolved)).parent):
            if d not in ro:
                ro.append(d)
    if settings_dir:
        ro.append(Path(settings_dir))

    rw: list[Path] = [Path.home() / ".claude"]
    config_json = _default_trust_config_path()
    if config_json.exists():
        rw.append(config_json)

    return SandboxContext(
        no_network=False,
        extra_ro_binds=ro,
        extra_rw_binds=rw,
        workspace_dir=Path(cwd) if cwd else None,
    )


class CCExecutor:
    """Drives interactive claude sessions toward job goals. One instance, shared
    across all cc jobs; per-job state lives in the DriveStateStore."""

    def __init__(self, config, drive_state: DriveStateStore):
        self.config = config
        self.drive_state = drive_state
        # Injected by the gateway at plugin-executor registration so the executor
        # can report an unexpected PTY exit via fail_worker.
        self.orchestrator = None

    async def start(self, job, followup: Optional[str] = None) -> None:
        """Executor contract entry point. followup=None on the initial attempt; on
        a retry, drive the SAME live session (write_stdin) when the PTY is alive,
        else respawn (with --resume when the cc session id is known)."""
        from tsugite_pty.tools import get_terminal_runtime

        state = self.drive_state.get(job.id)
        if followup is not None and state is not None and state.terminal_id:
            pty_manager, _store, _cb = get_terminal_runtime()
            proc = pty_manager.get(state.terminal_id) if pty_manager else None
            if proc is not None and proc.exit_code is None:
                # claude's Ink TUI submits on carriage return, not newline, and
                # debounces paste, so the text and the \r must be sent separately.
                pty_manager.write_stdin(state.terminal_id, followup.strip().encode())
                await asyncio.sleep(0.2)
                pty_manager.write_stdin(state.terminal_id, b"\r")
                state.consecutive_continues = 0
                return
        await self._spawn(job, followup)

    async def _spawn(self, job, followup: Optional[str]) -> None:
        import secrets

        from tsugite_pty.terminal_runtime import spawn_terminal
        from tsugite_pty.tools import get_terminal_runtime

        claude = self.config.claude_binary
        if shutil.which(claude) is None:
            await self._fail(job.id, f"claude binary not found on PATH: {claude!r}")
            return

        cwd = job.worktree_path or job.workspace_path
        if not cwd or not Path(cwd).is_dir():
            await self._fail(job.id, f"cc job workspace does not exist: {cwd!r}")
            return

        # An untrusted cwd hangs forever on the trust dialog (no CLI flag skips
        # it), so we provision the flag ourselves, or fail fast if that's disabled.
        if not await asyncio.to_thread(is_workspace_trusted, cwd):
            if self.config.provision_trust:
                target = await asyncio.to_thread(trust_provision_target, cwd)
                if not await asyncio.to_thread(ensure_workspace_trusted, target):
                    await self._fail(
                        job.id,
                        f"cc-driver: could not provision Claude Code trust for {target} "
                        f"(check the config at ~/.claude.json is writable and valid JSON).",
                    )
                    return
            else:
                await self._fail(
                    job.id,
                    f"cc-driver: workspace {cwd} is not trusted by Claude Code, so an unattended "
                    f"session would hang on the trust prompt. Trust it once (run `claude` in {cwd} and "
                    f"accept the prompt), or enable plugins.cc_driver.provision_trust to auto-trust it.",
                )
                return

        pty_manager, store, on_state_change = get_terminal_runtime()
        if pty_manager is None or store is None:
            await self._fail(job.id, "PTY runtime not available (daemon terminal viewer not wired)")
            return

        state = self.drive_state.get(job.id)
        if state is None:
            state = self.drive_state.create(job.id, secrets.token_urlsafe(32))

        hook_url = f"{self.config.base_url.rstrip('/')}/api/plugins/cc_driver/hook/{state.token}"
        settings_path = write_run_settings(self.config.state_dir, job.id, build_settings(hook_url))

        prompt = (
            followup
            if followup is not None
            else build_initial_prompt(
                job.prompt, self.config.completion_marker, needs_input_marker=self.config.needs_input_marker
            )
        )
        cmd = build_claude_command(
            claude,
            str(settings_path),
            self.config.permission_mode,
            prompt,
            model=(getattr(job, "model", None) or self.config.model),
            effort=getattr(job, "effort", None) or self.config.effort,
            ax_screen_reader=self.config.ax_screen_reader,
            resume_session_id=state.cc_session_id if followup is not None else None,
        )

        try:
            session = spawn_terminal(
                store=store,
                manager=pty_manager,
                cmd=cmd,
                cwd=cwd,
                parent_session_id=None,
                on_state_change=on_state_change,
                sandbox_ctx=build_sandbox_ctx(
                    self.config.sandbox, cwd, claude_binary=claude, settings_dir=settings_path.parent
                ),
            )
        except Exception as e:
            logger.exception("cc-driver: failed to spawn claude PTY for job '%s'", job.id)
            await self._fail(job.id, f"failed to spawn claude PTY: {e}", detail=traceback.format_exc())
            return

        state.terminal_id = session.id
        state.consecutive_continues = 0
        self._set_worker_terminal(job.id, session.id)

        proc = pty_manager.get(session.id)
        if proc is not None:
            loop = asyncio.get_running_loop()
            proc.on_exit(lambda p: self._on_pty_exit(job.id, p, loop))

    async def cancel(self, job) -> None:
        """Tear down the driven session on any terminal finalize (best-effort). A
        resolved job (done/cancelled) drops all state; a parked one (stuck/errored)
        still kills the PTY but keeps DriveState so a retry can `--resume`."""
        from tsugite_pty.tools import get_terminal_runtime

        state = self.drive_state.get(job.id)
        if state is not None and state.terminal_id:
            pty_manager, _store, _cb = get_terminal_runtime()
            if pty_manager is not None:
                try:
                    pty_manager.kill(state.terminal_id)
                except Exception:
                    logger.exception("cc-driver: failed to kill PTY for job '%s'", job.id)
            state.terminal_id = None
        if getattr(job, "state", None) in ("stuck", "errored"):
            return
        cleanup(self.config.state_dir, job.id)
        self.drive_state.remove(job.id)

    def _on_pty_exit(self, job_id: str, proc, loop) -> None:
        """PTY-exit callback. A deliberate kill sets proc.killed and is ignored;
        any other exit fails the worker. The buffer tail rides along as the detail
        because the exit code alone can't convey the real error (missing binary,
        auth failure, ...)."""
        if getattr(proc, "killed", False):
            return
        reason = f"claude session exited ({describe_exit(proc.exit_code)}) before completing the task"
        try:
            detail = pty_tail(proc.buffer)
        except Exception:
            detail = None
        if self.orchestrator is None:
            return
        loop.call_soon_threadsafe(
            lambda: loop.create_task(self.orchestrator.fail_worker(job_id, reason, detail=detail))
        )

    def _set_worker_terminal(self, job_id: str, terminal_id: str) -> None:
        """Stamp job.worker_terminal_id so the existing job tile embeds the live
        terminal (the orchestrator prefers this field over a terminal_store lookup)."""
        if self.orchestrator is None:
            return
        try:
            self.orchestrator.attach_worker_terminal(job_id, terminal_id)
        except Exception:
            logger.debug("cc-driver: could not stamp worker_terminal_id for job '%s'", job_id)

    async def _fail(self, job_id: str, reason: str, detail: Optional[str] = None) -> None:
        if self.orchestrator is None:
            logger.warning("cc-driver: no orchestrator to report failure for job '%s': %s", job_id, reason)
            return
        await self.orchestrator.fail_worker(job_id, reason, detail=detail)
