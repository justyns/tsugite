"""CCExecutor: command building, sandbox policy, live-PTY followup, spawn, cancel.

Uses a real PtyManager + TerminalSessionStore (tmp) and a fake `claude` shell
script - no real claude, no network.
"""

import asyncio
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import tsugite_cc_driver.executor as executor_mod
from tsugite_cc_driver.adapter import CCDriverConfig
from tsugite_cc_driver.executor import (
    CCExecutor,
    DriveStateStore,
    build_claude_command,
    build_sandbox_ctx,
    ensure_workspace_trusted,
    is_workspace_trusted,
    reap_job_processes,
    trust_provision_target,
)
from tsugite_cc_driver.settings import build_settings, write_run_settings
from tsugite_pty.pty_manager import PtyManager
from tsugite_pty.terminal_runtime import spawn_terminal
from tsugite_pty.terminal_store import TerminalSessionStore

from tsugite.core.record_store import now_iso

from .conftest import _wait_until


class FakeJob:
    def __init__(self, job_id, *, prompt="do the thing", workspace_path=None, worktree_path=None, model=None):
        self.id = job_id
        self.state = "running"
        self.prompt = prompt
        self.workspace_path = workspace_path
        self.worktree_path = worktree_path
        self.worker_terminal_id = None
        self.last_activity_at = None
        self.model = model


class FakeJobStore:
    def __init__(self):
        self._jobs = {}

    def add(self, job):
        self._jobs[job.id] = job

    def get(self, job_id):
        return self._jobs.get(job_id)

    def update(self, job_id, **fields):
        job = self._jobs[job_id]
        for k, v in fields.items():
            setattr(job, k, v)
        return job


class FakeOrchestrator:
    def __init__(self):
        self._jobs = FakeJobStore()
        self.failed = []
        self.completed = []
        self.activity = []

    def get_job(self, job_id):
        return self._jobs.get(job_id)

    def attach_worker_terminal(self, job_id, terminal_id):
        self._jobs.update(job_id, worker_terminal_id=terminal_id)

    def record_worker_activity(self, job_id):
        self.activity.append(job_id)
        self._jobs.update(job_id, last_activity_at=now_iso())

    async def fail_worker(self, job_id, error, detail=None):
        self.failed.append((job_id, error, detail))

    async def complete_worker(self, job_id, summary):
        self.completed.append((job_id, summary))


@pytest.fixture
def runtime(tmp_path):
    """Wire a real PtyManager + TerminalSessionStore into tsugite_pty.tools."""
    from tsugite_pty.tools import set_terminal_runtime

    manager = PtyManager()
    store = TerminalSessionStore(tmp_path / "terminals.json")
    set_terminal_runtime(manager, store, None)
    try:
        yield manager, store
    finally:
        manager.shutdown()
        set_terminal_runtime(None, None, None)


def _executor(tmp_path, orch, **cfg):
    cfg.setdefault("state_dir", str(tmp_path / "state"))
    config = CCDriverConfig(**cfg)
    ex = CCExecutor(config, DriveStateStore())
    ex.orchestrator = orch
    return ex


def _write_trust_config(config_dir: Path, cwd, *, accepted=True) -> Path:
    """Write a fake Claude Code ~/.claude.json trusting `cwd` (by resolved abs path)."""
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / ".claude.json"
    path.write_text(json.dumps({"projects": {str(Path(cwd).resolve()): {"hasTrustDialogAccepted": accepted}}}))
    return path


def _fake_linked_worktree(repo: Path, name: str) -> Path:
    """Lay out a linked-worktree structure by hand, matching what
    `git worktree add` writes: <repo>/.git/worktrees/<name> plus a worktree
    under <repo>/.tsugite-jobs/ whose `.git` *file* points at it."""
    gitdir = repo / ".git" / "worktrees" / name
    gitdir.mkdir(parents=True)
    wt = repo / ".tsugite-jobs" / name
    wt.mkdir(parents=True)
    (wt / ".git").write_text(f"gitdir: {gitdir}\n")
    return wt


# ── is_workspace_trusted ──


def test_is_workspace_trusted_true_when_accepted(tmp_path):
    cwd = tmp_path / "ws"
    cwd.mkdir()
    cfg = _write_trust_config(tmp_path / "cfg", cwd)
    assert is_workspace_trusted(cwd, config_path=cfg) is True


def test_is_workspace_trusted_true_for_subdir_of_trusted_ancestor(tmp_path):
    # A --repo Job's worker runs in a fresh worktree at <repo>/.tsugite-jobs/<id>;
    # that exact path is never in CC's trust list, but CC treats a subdir of a
    # trusted project as trusted (verified against claude 2.1.207), so the
    # ancestor's trust must count or --repo cc jobs can never launch.
    repo = tmp_path / "ws"
    worktree = repo / ".tsugite-jobs" / "job-abc"
    worktree.mkdir(parents=True)
    cfg = _write_trust_config(tmp_path / "cfg", repo)  # trusts the repo root only
    assert is_workspace_trusted(worktree, config_path=cfg) is True


def test_is_workspace_trusted_false_for_nested_repo_under_trusted_ancestor(tmp_path):
    # Trust does not cross a nested git root.
    workspace = tmp_path / "ws"
    worktree = _fake_linked_worktree(workspace / "repos" / "widget", "job-1")
    cfg = _write_trust_config(tmp_path / "cfg", workspace)  # trusts the workspace root only
    assert is_workspace_trusted(worktree, config_path=cfg) is False


def test_is_workspace_trusted_true_for_worktree_of_trusted_repo_root(tmp_path):
    # A linked worktree's `.git` is a file, not a repo root.
    repo = tmp_path / "repo"
    worktree = _fake_linked_worktree(repo, "job-2")
    cfg = _write_trust_config(tmp_path / "cfg", repo)
    assert is_workspace_trusted(worktree, config_path=cfg) is True


def test_is_workspace_trusted_false_when_only_a_sibling_is_trusted(tmp_path):
    # A trusted path that is NOT an ancestor must not leak trust.
    trusted = tmp_path / "other"
    trusted.mkdir()
    cwd = tmp_path / "ws" / "sub"
    cwd.mkdir(parents=True)
    cfg = _write_trust_config(tmp_path / "cfg", trusted)
    assert is_workspace_trusted(cwd, config_path=cfg) is False


def test_is_workspace_trusted_false_when_key_absent(tmp_path):
    cwd = tmp_path / "ws"
    cwd.mkdir()
    cfg = tmp_path / "cfg" / ".claude.json"
    cfg.parent.mkdir()
    cfg.write_text(json.dumps({"projects": {"/some/other/path": {"hasTrustDialogAccepted": True}}}))
    assert is_workspace_trusted(cwd, config_path=cfg) is False


def test_is_workspace_trusted_false_when_flag_false(tmp_path):
    cwd = tmp_path / "ws"
    cwd.mkdir()
    cfg = _write_trust_config(tmp_path / "cfg", cwd, accepted=False)
    assert is_workspace_trusted(cwd, config_path=cfg) is False


def test_is_workspace_trusted_false_when_missing_or_malformed(tmp_path):
    cwd = tmp_path / "ws"
    cwd.mkdir()
    assert is_workspace_trusted(cwd, config_path=tmp_path / "nope.json") is False
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert is_workspace_trusted(cwd, config_path=bad) is False


def test_is_workspace_trusted_reads_claude_config_dir(tmp_path, monkeypatch):
    cwd = tmp_path / "ws"
    cwd.mkdir()
    config_dir = tmp_path / "cfgdir"
    _write_trust_config(config_dir, cwd)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    assert is_workspace_trusted(cwd) is True, "must read <CLAUDE_CONFIG_DIR>/.claude.json by default"


# ── ensure_workspace_trusted (provisioning) ──


def test_ensure_workspace_trusted_writes_the_flag(tmp_path):
    cwd = tmp_path / "ws"
    cwd.mkdir()
    cfg = tmp_path / "cfg" / ".claude.json"
    cfg.parent.mkdir()
    cfg.write_text(json.dumps({"projects": {}, "someOtherKey": 42}))

    assert ensure_workspace_trusted(cwd, config_path=cfg) is True
    data = json.loads(cfg.read_text())
    assert data["projects"][str(cwd.resolve())]["hasTrustDialogAccepted"] is True
    assert data["someOtherKey"] == 42, "unrelated top-level keys must be preserved"
    # And the read side now agrees.
    assert is_workspace_trusted(cwd, config_path=cfg) is True


def test_ensure_workspace_trusted_preserves_existing_projects(tmp_path):
    cwd = tmp_path / "ws"
    cwd.mkdir()
    cfg = tmp_path / "cfg" / ".claude.json"
    cfg.parent.mkdir()
    cfg.write_text(
        json.dumps({"projects": {"/other/proj": {"hasTrustDialogAccepted": True, "allowedTools": ["Read"]}}})
    )
    ensure_workspace_trusted(cwd, config_path=cfg)
    data = json.loads(cfg.read_text())
    assert data["projects"]["/other/proj"]["allowedTools"] == ["Read"], "existing project entries must survive"
    assert data["projects"][str(cwd.resolve())]["hasTrustDialogAccepted"] is True


def test_ensure_workspace_trusted_creates_config_when_absent(tmp_path):
    cwd = tmp_path / "ws"
    cwd.mkdir()
    cfg = tmp_path / "cfg" / ".claude.json"  # parent dir + file both missing
    assert ensure_workspace_trusted(cwd, config_path=cfg) is True
    assert is_workspace_trusted(cwd, config_path=cfg) is True


def test_ensure_workspace_trusted_idempotent(tmp_path):
    cwd = tmp_path / "ws"
    cwd.mkdir()
    cfg = _write_trust_config(tmp_path / "cfg", cwd)
    before = cfg.read_text()
    assert ensure_workspace_trusted(cwd, config_path=cfg) is True
    assert is_workspace_trusted(cwd, config_path=cfg) is True
    assert json.loads(cfg.read_text()) == json.loads(before), "already-trusted -> no meaningful change"


def test_ensure_workspace_trusted_does_not_clobber_malformed_config(tmp_path):
    cwd = tmp_path / "ws"
    cwd.mkdir()
    cfg = tmp_path / "cfg" / ".claude.json"
    cfg.parent.mkdir()
    cfg.write_text("{ this is not valid json")
    assert ensure_workspace_trusted(cwd, config_path=cfg) is False, "must not overwrite an unparseable config"
    assert cfg.read_text() == "{ this is not valid json", "the malformed file must be left untouched"


# ── trust_provision_target ──


def test_trust_provision_target_maps_worktree_to_repo_root(tmp_path):
    repo = tmp_path / "repo"
    wt = _fake_linked_worktree(repo, "job-1")
    assert trust_provision_target(wt) == repo.resolve()


def test_trust_provision_target_plain_repo_is_itself(tmp_path):
    ws = tmp_path / "ws"
    (ws / ".git").mkdir(parents=True)  # main checkout: .git is a directory
    assert trust_provision_target(ws) == ws.resolve()
    bare = tmp_path / "bare"
    bare.mkdir()  # no .git at all
    assert trust_provision_target(bare) == bare.resolve()


def test_trust_provision_target_external_worktree_is_itself(tmp_path):
    # A worktree OUTSIDE the repo root: trusting the repo root wouldn't cover
    # it (the ancestor rule only reaches subdirs), so it needs its own entry.
    repo = tmp_path / "repo"
    (repo / ".git" / "worktrees" / "wt").mkdir(parents=True)
    wt = tmp_path / "elsewhere" / "wt"
    wt.mkdir(parents=True)
    (wt / ".git").write_text(f"gitdir: {repo / '.git' / 'worktrees' / 'wt'}\n")
    assert trust_provision_target(wt) == wt.resolve()


# ── pure helpers ──


def test_build_claude_command_quotes_goal_with_quotes_and_newlines():
    prompt = "fix the 'bug'\nand add a test"
    cmd = build_claude_command("claude", "/s/settings.json", "bypassPermissions", prompt)
    # The whole prompt is a single shell token; re-splitting must recover it verbatim.
    argv = shlex.split(cmd)
    assert argv[0] == "claude"
    assert "--settings" in argv and "/s/settings.json" in argv
    assert "--permission-mode" in argv and "bypassPermissions" in argv
    assert argv[-1] == prompt, "goal must survive shell quoting intact"


def test_build_claude_command_adds_resume_when_given():
    cmd = build_claude_command("claude", "/s.json", "bypassPermissions", "go", resume_session_id="cc-42")
    argv = shlex.split(cmd)
    assert "--resume" in argv and "cc-42" in argv


def test_build_claude_command_adds_model_when_given():
    argv = shlex.split(build_claude_command("claude", "/s.json", "bypassPermissions", "go", model="opus"))
    assert "--model" in argv and "opus" in argv
    # None -> no --model flag (claude uses its own default).
    assert "--model" not in shlex.split(build_claude_command("claude", "/s.json", "bypassPermissions", "go"))


def test_build_claude_command_adds_effort_when_given():
    argv = shlex.split(build_claude_command("claude", "/s.json", "bypassPermissions", "go", effort="max"))
    assert "--effort" in argv and "max" in argv
    # None -> no --effort flag.
    assert "--effort" not in shlex.split(build_claude_command("claude", "/s.json", "bypassPermissions", "go"))


def test_build_claude_command_adds_ax_screen_reader_when_enabled():
    argv = shlex.split(build_claude_command("claude", "/s.json", "bypassPermissions", "go", ax_screen_reader=True))
    assert "--ax-screen-reader" in argv, "flat screen-reader output makes PTY capture cleaner"
    # Off -> no flag.
    assert "--ax-screen-reader" not in shlex.split(build_claude_command("claude", "/s.json", "bypassPermissions", "go"))


def test_cc_config_defaults_ax_screen_reader_off():
    assert CCDriverConfig().ax_screen_reader is False, "screen-reader output is opt-in; the TUI is the default"


@pytest.mark.asyncio
async def test_initial_spawn_threads_effort_and_ax_screen_reader(tmp_path, runtime, monkeypatch):
    manager, store = runtime
    orch = FakeOrchestrator()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    job = FakeJob("job-1", workspace_path=str(workspace))
    job.effort = "high"
    orch._jobs.add(job)

    config_dir = tmp_path / "cfgdir"
    _write_trust_config(config_dir, workspace)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

    captured = {}
    import tsugite_pty.terminal_runtime as tr_mod

    real_spawn = tr_mod.spawn_terminal

    def spy_spawn(**kwargs):
        captured["cmd"] = kwargs.get("cmd")
        return real_spawn(**kwargs)

    monkeypatch.setattr(tr_mod, "spawn_terminal", spy_spawn)

    fake = tmp_path / "claude"
    fake.write_text("#!/bin/sh\nsleep 30\n")
    fake.chmod(0o755)
    # ax_screen_reader is off by default; enable it here to prove it threads through.
    ex = _executor(tmp_path, orch, claude_binary=str(fake), sandbox=False, ax_screen_reader=True)

    await ex.start(job, followup=None)

    argv = shlex.split(captured["cmd"])
    assert "--effort" in argv and "high" in argv, "per-job effort must reach the claude command"
    assert "--ax-screen-reader" in argv, "the enabled config flag must reach the claude command"

    st = ex.drive_state.get("job-1")
    if st and st.terminal_id:
        manager.kill(st.terminal_id)


def test_build_sandbox_ctx_off_is_none():
    assert build_sandbox_ctx(False, "/work") is None


def test_build_sandbox_ctx_on_keeps_network_and_binds_claude_dir():
    ctx = build_sandbox_ctx(True, "/work")
    assert ctx is not None
    assert ctx.no_network is False, "the driven claude needs network for the API"
    assert Path.home() / ".claude" in ctx.extra_rw_binds
    assert ctx.workspace_dir == Path("/work")


def _fake_claude_install(tmp_path):
    """Create bin/claude -> share/claude/<ver>/claude, mirroring the native install
    layout (PATH symlink into a per-version dir). Returns (symlink, bindir, install)."""
    install = tmp_path / "share" / "claude" / "2.1.0"
    install.mkdir(parents=True)
    real = install / "claude"
    real.write_text("#!/bin/sh\n")
    real.chmod(0o755)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    link = bindir / "claude"
    link.symlink_to(real)
    return link, bindir, install


def test_build_sandbox_ctx_binds_the_claude_binary_dirs(tmp_path):
    # Without these, the jailed claude dies with exit 127 (command not found).
    link, bindir, install = _fake_claude_install(tmp_path)
    ctx = build_sandbox_ctx(True, str(tmp_path / "work"), claude_binary=str(link))
    assert bindir in ctx.extra_ro_binds, "PATH dir (symlink) must be bound"
    assert install in ctx.extra_ro_binds, "real per-version install dir must be bound"


def test_build_sandbox_ctx_binds_the_trust_config_rw(tmp_path, monkeypatch):
    # ~/.claude.json must be in the jail (trust flag) and bound read-WRITE - claude
    # persists session metadata into it on shutdown and errors on a RO mount.
    cfg = tmp_path / ".claude.json"
    cfg.write_text("{}")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    ctx = build_sandbox_ctx(True, str(tmp_path / "work"), claude_binary="claude-absent")
    assert cfg in ctx.extra_rw_binds
    assert cfg not in ctx.extra_ro_binds


def test_build_sandbox_ctx_binds_the_settings_dir(tmp_path):
    # Without it, `claude --settings <path>` errors "Settings file not found".
    settings_dir = tmp_path / "state" / "job-1"
    ctx = build_sandbox_ctx(True, str(tmp_path / "work"), claude_binary="claude-absent", settings_dir=settings_dir)
    assert settings_dir in ctx.extra_ro_binds


def test_sandboxed_claude_and_trust_present_in_real_bwrap_argv(tmp_path, monkeypatch):
    # Repro guard against the exit-127 bug: the claude dirs + trust file must land
    # as --ro-bind in the ACTUAL bwrap command, not just the SandboxContext.
    pytest.importorskip("tsugite_sandbox")
    import shutil

    if shutil.which("bwrap") is None:
        pytest.skip("bwrap not installed")
    from tsugite_pty.terminal_runtime import maybe_sandbox_argv

    link, bindir, install = _fake_claude_install(tmp_path)
    cfg = tmp_path / ".claude.json"
    cfg.write_text("{}")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    work = tmp_path / "work"
    work.mkdir()
    settings_dir = tmp_path / "state" / "job-1"
    settings_dir.mkdir(parents=True)

    ctx = build_sandbox_ctx(True, str(work), claude_binary=str(link), settings_dir=settings_dir)
    argv = maybe_sandbox_argv(["claude"], str(work), ctx, force_no_network=False)

    assert str(bindir) in argv, "claude PATH dir missing from bwrap argv -> exit 127"
    assert str(install) in argv, "claude install dir missing from bwrap argv"
    assert str(settings_dir) in argv, "settings dir missing from bwrap argv -> settings file not found"
    assert str(cfg) in argv, "trust config missing from bwrap argv -> trust-prompt hang"
    assert "--unshare-net" not in argv, "cc-driver sandbox keeps network for the API"


def test_real_claude_launches_inside_the_jail(tmp_path):
    # End-to-end guard against binds that are structurally present but runtime-
    # insufficient: claude's binary + bundled node runtime must fully load inside
    # the exact jail cc-driver builds. `--version` needs no login/trust/network.
    pytest.importorskip("tsugite_sandbox")
    import shutil
    import subprocess

    if shutil.which("claude") is None or shutil.which("bwrap") is None:
        pytest.skip("needs a real claude + bwrap install")
    from tsugite_pty.terminal_runtime import maybe_sandbox_argv

    work = tmp_path / "work"
    work.mkdir()
    sp = write_run_settings(tmp_path / "state", "job-1", build_settings("http://127.0.0.1:8899/hook/tok"))
    ctx = build_sandbox_ctx(True, str(work), claude_binary="claude", settings_dir=sp.parent)
    argv = maybe_sandbox_argv(["claude", "--version"], str(work), ctx, force_no_network=False)
    r = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"claude failed to launch in the jail: {r.stdout}{r.stderr}"
    assert r.stdout.strip(), "claude --version produced no output in the jail"


def test_settings_registers_exactly_the_three_hooks_with_token_url(tmp_path):
    url = "http://127.0.0.1:8374/api/plugins/cc_driver/hook/tok-abc"
    path = write_run_settings(tmp_path, "job-1", build_settings(url))
    data = json.loads(path.read_text())
    assert set(data["hooks"]) == {"Stop", "StopFailure", "Notification"}
    for event, entries in data["hooks"].items():
        assert entries[0]["hooks"][0] == {"type": "http", "url": url}, event


# ── executor behaviour ──


@pytest.mark.asyncio
async def test_missing_binary_fails_worker(tmp_path, runtime):
    orch = FakeOrchestrator()
    orch._jobs.add(FakeJob("job-1", workspace_path=str(tmp_path)))
    ex = _executor(tmp_path, orch, claude_binary="tsugite-no-such-claude-xyz")
    await ex.start(orch._jobs.get("job-1"), followup=None)
    assert orch.failed and orch.failed[0][0] == "job-1"
    assert "not found" in orch.failed[0][1]


@pytest.mark.asyncio
async def test_followup_on_live_pty_types_via_write_stdin(tmp_path, runtime):
    manager, store = runtime
    orch = FakeOrchestrator()
    job = FakeJob("job-1", workspace_path=str(tmp_path))
    orch._jobs.add(job)
    ex = _executor(tmp_path, orch)

    # A live PTY running `cat` echoes whatever we type into it.
    session = spawn_terminal(store=store, manager=manager, cmd="cat", cwd=str(tmp_path), sandbox_ctx=None)
    st = ex.drive_state.create("job-1", "tok-1")
    st.terminal_id = session.id
    st.consecutive_continues = 3

    # Spy on the raw bytes written - the echo buffer is unreliable for \r vs \n
    # because the PTY line discipline translates them.
    writes = []
    orig = manager.write_stdin
    manager.write_stdin = lambda tid, data: (writes.append(data), orig(tid, data))[1]

    await ex.start(job, followup="please keep going")

    proc = manager.get(session.id)
    proc.wait_drain(timeout=1.0)
    sent = b"".join(writes)
    assert b"please keep going" in sent, "followup text must be typed into the live PTY"
    assert b"\r" in sent, "the TUI submits on carriage return, not newline"
    assert b"\n" not in sent, "a bare newline would land as literal text, never submitting"
    assert b"please keep going" in proc.buffer, "the live PTY must actually receive it"
    assert st.consecutive_continues == 0, "a supervisor followup resets the continue budget"
    # No respawn: still the same terminal.
    assert st.terminal_id == session.id


@pytest.mark.asyncio
async def test_wrap_up_types_the_nudge_without_refreshing_the_continue_budget(tmp_path, runtime):
    """The wrap-up nudge reaches the live TUI the same way a followup does, but must
    leave consecutive_continues alone: topping the budget up at the deadline hands
    the worker a fresh run of Stop nudges exactly when it should be finishing."""
    manager, store = runtime
    orch = FakeOrchestrator()
    job = FakeJob("job-1", workspace_path=str(tmp_path))
    orch._jobs.add(job)
    ex = _executor(tmp_path, orch)

    session = spawn_terminal(store=store, manager=manager, cmd="cat", cwd=str(tmp_path), sandbox_ctx=None)
    st = ex.drive_state.create("job-1", "tok-1")
    st.terminal_id = session.id
    st.consecutive_continues = 4

    writes = []
    orig = manager.write_stdin
    manager.write_stdin = lambda tid, data: (writes.append(data), orig(tid, data))[1]

    await ex.wrap_up(job, minutes_left=5)

    proc = manager.get(session.id)
    proc.wait_drain(timeout=1.0)
    sent = b"".join(writes)
    assert b"5 minutes" in sent, "the nudge must reach the live PTY"
    assert b"CCDRIVER_WRAPPED_UP" in sent, "the worker needs a marker it can emit over unfinished work"
    assert b"\r" in sent, "the TUI submits on carriage return, not newline"
    assert st.consecutive_continues == 4, "the wrap-up nudge must not refresh the continue budget"
    assert st.terminal_id == session.id, "wrapping up must never respawn the worker"


@pytest.mark.asyncio
async def test_wrap_up_is_a_noop_when_the_pty_is_gone(tmp_path, runtime):
    """A worker whose PTY already exited has nothing to nudge; wrap_up must not
    respawn claude to deliver it."""
    manager, store = runtime
    orch = FakeOrchestrator()
    job = FakeJob("job-1", workspace_path=str(tmp_path))
    orch._jobs.add(job)
    ex = _executor(tmp_path, orch)
    st = ex.drive_state.create("job-1", "tok-1")
    st.terminal_id = "gone"

    await ex.wrap_up(job, minutes_left=3)

    assert store.list_all() == [], "wrap_up must not spawn a terminal"
    assert orch.failed == []


@pytest.mark.asyncio
async def test_wrap_up_skips_a_worker_waiting_on_a_permission_prompt(tmp_path, runtime):
    """A job sitting on a permission prompt stays RUNNING, so the nudge timer still
    fires. The prompt is the TUI's foreground element, so the nudge would be typed
    into the dialog rather than the worker."""
    manager, store = runtime
    orch = FakeOrchestrator()
    job = FakeJob("job-1", workspace_path=str(tmp_path))
    orch._jobs.add(job)
    ex = _executor(tmp_path, orch)

    session = spawn_terminal(store=store, manager=manager, cmd="cat", cwd=str(tmp_path), sandbox_ctx=None)
    st = ex.drive_state.create("job-1", "tok-1")
    st.terminal_id = session.id
    st.attention_flagged = True

    writes = []
    orig = manager.write_stdin
    manager.write_stdin = lambda tid, data: (writes.append(data), orig(tid, data))[1]

    await ex.wrap_up(job, minutes_left=5)

    assert writes == [], "nothing may be typed while a prompt is on screen"


@pytest.mark.asyncio
async def test_start_survives_drive_state_removed_mid_write(tmp_path, runtime):
    """Typing a followup awaits between the text and the carriage return, and
    cancel() drops the job's DriveState from the same loop."""
    manager, store = runtime
    orch = FakeOrchestrator()
    job = FakeJob("job-1", workspace_path=str(tmp_path))
    orch._jobs.add(job)
    ex = _executor(tmp_path, orch)

    session = spawn_terminal(store=store, manager=manager, cmd="cat", cwd=str(tmp_path), sandbox_ctx=None)
    st = ex.drive_state.create("job-1", "tok-1")
    st.terminal_id = session.id

    orig = manager.write_stdin

    def evict_then_write(terminal_id, data):
        ex.drive_state.remove("job-1")
        return orig(terminal_id, data)

    manager.write_stdin = evict_then_write

    await ex.start(job, followup="please keep going")

    assert ex.drive_state.get("job-1") is None
    assert len(store.list_all()) == 1, "the followup must not respawn claude"


def test_wrap_up_lead_defaults_to_a_tenth_of_the_timeout(tmp_path):
    ex = _executor(tmp_path, FakeOrchestrator())
    assert ex.wrap_up_lead_minutes(90) == 9
    assert ex.wrap_up_lead_minutes(20) == 3, "short jobs still get a 3 minute floor"


def test_wrap_up_lead_is_configurable_and_zero_disables(tmp_path):
    assert _executor(tmp_path, FakeOrchestrator(), wrap_up_lead_minutes=15).wrap_up_lead_minutes(90) == 15
    assert _executor(tmp_path, FakeOrchestrator(), wrap_up_lead_minutes=0).wrap_up_lead_minutes(90) == 0


@pytest.mark.asyncio
async def test_untrusted_workspace_fails_worker_when_provisioning_disabled(tmp_path, runtime, monkeypatch):
    manager, store = runtime
    orch = FakeOrchestrator()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    job = FakeJob("job-1", workspace_path=str(workspace))
    orch._jobs.add(job)
    fake = tmp_path / "claude"
    fake.write_text("#!/bin/sh\nsleep 30\n")
    fake.chmod(0o755)
    # Point trust lookup at an empty config dir -> cwd is NOT trusted.
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "empty-cfg"))
    ex = _executor(tmp_path, orch, claude_binary=str(fake), provision_trust=False)

    await ex.start(job, followup=None)

    assert orch.failed and "not trusted" in orch.failed[0][1]
    assert store.list_all() == [], "an untrusted workspace must not spawn a PTY (it would hang on the trust dialog)"
    assert ex.drive_state.get("job-1") is None


@pytest.mark.asyncio
async def test_untrusted_workspace_is_provisioned_then_spawns(tmp_path, runtime, monkeypatch):
    manager, store = runtime
    orch = FakeOrchestrator()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    job = FakeJob("job-1", workspace_path=str(workspace))
    orch._jobs.add(job)
    fake = tmp_path / "claude"
    fake.write_text("#!/bin/sh\nsleep 30\n")
    fake.chmod(0o755)
    config_dir = tmp_path / "cfgdir"  # starts empty -> cwd NOT trusted
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    # provision_trust defaults True; sandbox off keeps it hermetic.
    ex = _executor(tmp_path, orch, claude_binary=str(fake), sandbox=False)

    await ex.start(job, followup=None)

    assert not orch.failed, f"provisioning should let the spawn proceed, got {orch.failed}"
    # The trust flag was written for the workspace, so the read side now agrees.
    assert is_workspace_trusted(workspace, config_path=config_dir / ".claude.json") is True
    st = ex.drive_state.get("job-1")
    assert st is not None and st.terminal_id, "the worker PTY must spawn after trust is provisioned"
    manager.kill(st.terminal_id)


@pytest.mark.asyncio
async def test_repo_job_provisions_trust_for_repo_root_not_worktree(tmp_path, runtime, monkeypatch):
    manager, store = runtime
    orch = FakeOrchestrator()
    repo = tmp_path / "repo"
    wt = _fake_linked_worktree(repo, "job-1")
    job = FakeJob("job-1", worktree_path=str(wt))
    orch._jobs.add(job)
    fake = tmp_path / "claude"
    fake.write_text("#!/bin/sh\nsleep 30\n")
    fake.chmod(0o755)
    config_dir = tmp_path / "cfgdir"  # starts empty -> nothing trusted
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    ex = _executor(tmp_path, orch, claude_binary=str(fake), sandbox=False)

    await ex.start(job, followup=None)

    assert not orch.failed, f"provisioning should let the spawn proceed, got {orch.failed}"
    projects = json.loads((config_dir / ".claude.json").read_text())["projects"]
    assert str(repo.resolve()) in projects, "the repo ROOT must be trusted, not the ephemeral worktree"
    assert str(wt.resolve()) not in projects, "no per-worktree dead entry may accumulate"
    # The ancestor rule makes the worktree itself read as trusted.
    assert is_workspace_trusted(wt, config_path=config_dir / ".claude.json") is True
    st = ex.drive_state.get("job-1")
    assert st is not None and st.terminal_id, "the worker PTY must spawn after trust is provisioned"
    manager.kill(st.terminal_id)


@pytest.mark.asyncio
async def test_initial_spawn_stamps_worker_terminal_and_writes_settings(tmp_path, runtime, monkeypatch):
    manager, store = runtime
    orch = FakeOrchestrator()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    job = FakeJob("job-1", workspace_path=str(workspace))
    orch._jobs.add(job)

    # Pre-trust the workspace so the spawn isn't blocked by the trust check.
    config_dir = tmp_path / "cfgdir"
    _write_trust_config(config_dir, workspace)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

    # Fake claude: stay alive so on_exit doesn't fire mid-test.
    fake = tmp_path / "claude"
    fake.write_text("#!/bin/sh\nsleep 30\n")
    fake.chmod(0o755)
    # sandbox=False keeps this hermetic (no bwrap dependency); the sandbox path is
    # covered by the build_sandbox_ctx tests.
    ex = _executor(tmp_path, orch, claude_binary=str(fake), base_url="http://127.0.0.1:8374", sandbox=False)

    await ex.start(job, followup=None)

    st = ex.drive_state.get("job-1")
    assert st is not None and st.terminal_id, "DriveState must record the spawned terminal"
    assert job.worker_terminal_id == st.terminal_id, "job.worker_terminal_id must be stamped for the tile"
    settings = json.loads((tmp_path / "state" / "job-1" / "settings.json").read_text())
    assert set(settings["hooks"]) == {"Stop", "StopFailure", "Notification"}
    assert st.token in settings["hooks"]["Stop"][0]["hooks"][0]["url"]

    manager.kill(st.terminal_id)


@pytest.mark.asyncio
async def test_cancel_kills_pty(tmp_path, runtime):
    manager, store = runtime
    orch = FakeOrchestrator()
    job = FakeJob("job-1", workspace_path=str(tmp_path))
    ex = _executor(tmp_path, orch)

    session = spawn_terminal(store=store, manager=manager, cmd="sleep 30", cwd=str(tmp_path), sandbox_ctx=None)
    st = ex.drive_state.create("job-1", "tok-1")
    st.terminal_id = session.id

    await ex.cancel(job)

    proc = manager.get(session.id)
    if proc is not None:
        proc.wait_drain(timeout=2.0)
        assert proc.killed or proc.exit_code is not None, "cancel must kill the PTY"
    assert ex.drive_state.get("job-1") is None, "cancel must drop DriveState"


@pytest.mark.asyncio
async def test_pty_exit_interprets_exit_code_and_captures_buffer_tail(tmp_path, runtime):
    orch = FakeOrchestrator()
    ex = _executor(tmp_path, orch)

    class DeadProc:
        killed = False
        exit_code = 127
        buffer = b"$ claude --settings s.json\r\n\x1b[31msh: 1: claude: not found\x1b[0m\r\n"

    ex._on_pty_exit("job-1", DeadProc(), asyncio.get_running_loop())
    await asyncio.sleep(0.05)

    assert orch.failed, "a non-kill PTY exit must fail the worker"
    job_id, reason, detail = orch.failed[0]
    assert job_id == "job-1"
    assert "127" in reason
    assert "not found" in reason, "exit 127 must be interpreted, not left as a bare code"
    assert "claude: not found" in detail, "the PTY buffer tail must ride along as the failure detail"
    assert "\x1b" not in detail, "ANSI escapes must be stripped from the detail"


@pytest.mark.asyncio
async def test_pty_exit_without_buffer_still_fails_with_reason(tmp_path, runtime):
    orch = FakeOrchestrator()
    ex = _executor(tmp_path, orch)

    class DeadProc:
        killed = False
        exit_code = 1
        buffer = b""

    ex._on_pty_exit("job-1", DeadProc(), asyncio.get_running_loop())
    await asyncio.sleep(0.05)

    assert orch.failed
    _job_id, reason, detail = orch.failed[0]
    assert "code 1" in reason
    assert detail is None, "no output tail -> no detail field"


@pytest.mark.asyncio
async def test_cancel_on_parked_job_kills_pty_but_keeps_resume_state(tmp_path, runtime):
    manager, store = runtime
    orch = FakeOrchestrator()
    job = FakeJob("job-1", workspace_path=str(tmp_path))
    job.state = "stuck"
    ex = _executor(tmp_path, orch)

    session = spawn_terminal(store=store, manager=manager, cmd="sleep 30", cwd=str(tmp_path), sandbox_ctx=None)
    st = ex.drive_state.create("job-1", "tok-1")
    st.terminal_id = session.id
    st.cc_session_id = "cc-abc"

    await ex.cancel(job)

    proc = manager.get(session.id)
    if proc is not None:
        proc.wait_drain(timeout=2.0)
        assert proc.killed or proc.exit_code is not None, "parked teardown must still kill the PTY"
    st2 = ex.drive_state.get("job-1")
    assert st2 is not None, "parked teardown must keep DriveState so a retry can --resume"
    assert st2.cc_session_id == "cc-abc"
    assert st2.terminal_id is None, "the dead terminal must not look live to a later retry"


@pytest.mark.asyncio
async def test_worker_output_advances_the_liveness_stamp_and_silence_freezes_it(tmp_path, runtime, monkeypatch):
    """PTY output stamps the job record, and a silent worker stops advancing it."""
    manager, store = runtime
    orch = FakeOrchestrator()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    job = FakeJob("job-1", workspace_path=str(workspace))
    orch._jobs.add(job)

    config_dir = tmp_path / "cfgdir"
    _write_trust_config(config_dir, workspace)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

    # Talks for about half a second, then goes quiet without exiting, like a claude
    # session that is alive but between tool calls. The sentinel is the last thing
    # it prints, so a test can wait for silence instead of guessing at it.
    fake = tmp_path / "claude"
    fake.write_text(
        "#!/bin/sh\nfor i in 1 2 3 4 5 6 7 8 9 10; do echo tick $i; sleep 0.05; done\necho DONE_TALKING\nsleep 30\n"
    )
    fake.chmod(0o755)
    monkeypatch.setattr(executor_mod, "_ACTIVITY_THROTTLE_SECONDS", 0.0)
    ex = _executor(tmp_path, orch, claude_binary=str(fake), sandbox=False)

    await ex.start(job, followup=None)

    assert await _wait_until(lambda: job.last_activity_at is not None), "the spawn must stamp the job record"
    first = job.last_activity_at
    assert await _wait_until(lambda: job.last_activity_at != first), "the stamp must advance while the worker talks"

    terminal_id = ex.drive_state.get("job-1").terminal_id
    assert await _wait_until(lambda: b"DONE_TALKING" in manager.get(terminal_id).buffer), (
        "the fake must reach its sentinel"
    )
    await asyncio.sleep(0.1)  # let the sentinel chunk's callback land
    quiet = job.last_activity_at
    await asyncio.sleep(0.3)
    assert job.last_activity_at == quiet, "a silent worker must not advance the stamp"


@pytest.mark.asyncio
async def test_liveness_stamps_are_throttled(tmp_path):
    """A claude TUI emits chunks faster than the record should be written."""
    orch = FakeOrchestrator()
    job = FakeJob("job-1")
    orch._jobs.add(job)
    ex = _executor(tmp_path, orch)
    state = ex.drive_state.create("job-1", "tok-1")
    loop = asyncio.get_running_loop()

    for _ in range(20):
        ex._on_output(state, loop)
    await asyncio.sleep(0)

    assert job.last_activity_at is not None, "the first chunk must stamp"
    assert orch.activity == ["job-1"], f"a burst must stamp once, stamped {len(orch.activity)}x"


@pytest.mark.asyncio
async def test_spawn_stamps_liveness_before_any_output(tmp_path, runtime, monkeypatch):
    """A worker that hangs before its first byte is one of the hangs the stamp exists
    to catch, so null has to mean "this executor does not report liveness"."""
    manager, store = runtime
    orch = FakeOrchestrator()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    job = FakeJob("job-1", workspace_path=str(workspace))
    orch._jobs.add(job)

    config_dir = tmp_path / "cfgdir"
    _write_trust_config(config_dir, workspace)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

    # Never prints anything, like a claude wedged on a dialog before its first turn.
    fake = tmp_path / "claude"
    fake.write_text("#!/bin/sh\nsleep 30\n")
    fake.chmod(0o755)
    ex = _executor(tmp_path, orch, claude_binary=str(fake), sandbox=False)

    await ex.start(job, followup=None)

    assert job.last_activity_at is not None, "spawning the worker must stamp the job record"
    assert orch.activity == ["job-1"]

    manager.kill(ex.drive_state.get("job-1").terminal_id)


# ── detached-descendant reaping ──


def _alive(pid: int) -> bool:
    """A zombie is dead for our purposes: it only awaits a wait() from a parent we do not control."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _detached_sleeper(token: str) -> subprocess.Popen:
    """A sleeper in its own session (what `setsid` buys a worker's helper) carrying a job token."""
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
        env={**os.environ, "TSUGITE_JOB_TOKEN": token},
    )


def test_reap_job_processes_kills_the_marked_session_and_spares_another_job(monkeypatch):
    monkeypatch.setattr(executor_mod, "_REAP_GRACE_SECONDS", 0.2)
    marked = _detached_sleeper("tok-reap-a")
    other = _detached_sleeper("tok-reap-b")
    try:
        assert os.getpgid(marked.pid) != os.getpgid(os.getpid()), "the sleeper must be outside our process group"

        reaped = reap_job_processes("tok-reap-a")

        assert marked.pid in reaped, f"the marked process must be reaped, reaped {reaped}"
        marked.wait(timeout=5)
        assert other.poll() is None, "a process carrying another job's token must be left alone"
    finally:
        for p in (marked, other):
            p.kill()
            p.wait(timeout=5)


@pytest.mark.asyncio
async def test_cancel_reaps_a_descendant_the_worker_detached_with_setsid(tmp_path, runtime, monkeypatch):
    """The reported leak: a helper that left the PTY's process group survives the group kill."""
    manager, store = runtime
    monkeypatch.setattr(executor_mod, "_REAP_GRACE_SECONDS", 0.2)
    orch = FakeOrchestrator()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    job = FakeJob("job-1", workspace_path=str(workspace))
    orch._jobs.add(job)

    config_dir = tmp_path / "cfgdir"
    _write_trust_config(config_dir, workspace)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

    pid_file = tmp_path / "detached.pid"
    helper = tmp_path / "detach.py"
    helper.write_text(
        f"import os, time\nos.setsid()\nopen({str(pid_file)!r}, 'w').write(str(os.getpid()))\ntime.sleep(60)\n"
    )
    fake = tmp_path / "claude"
    fake.write_text(f"#!/bin/sh\n{sys.executable} {helper} &\nsleep 60\n")
    fake.chmod(0o755)
    ex = _executor(tmp_path, orch, claude_binary=str(fake), sandbox=False)

    await ex.start(job, followup=None)
    assert await _wait_until(lambda: pid_file.read_text().strip() if pid_file.exists() else ""), (
        "the fake worker must detach its helper"
    )
    detached = int(pid_file.read_text().strip())
    terminal_pid = manager.get(ex.drive_state.get("job-1").terminal_id).pid
    try:
        assert os.getpgid(detached) != os.getpgid(terminal_pid), "the helper must have escaped the PTY's group"

        await ex.cancel(job)

        assert await _wait_until(lambda: not _alive(detached), timeout=5.0), (
            "the detached helper must not outlive the job"
        )
    finally:
        try:
            os.kill(detached, signal.SIGKILL)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_cancel_spares_a_process_that_only_exports_the_job_id(tmp_path, runtime, monkeypatch):
    """A shell that exported the job's id for debugging is not the job's process tree."""
    monkeypatch.setattr(executor_mod, "_REAP_GRACE_SECONDS", 0.2)
    orch = FakeOrchestrator()
    job = FakeJob("job-1", workspace_path=str(tmp_path))
    ex = _executor(tmp_path, orch)
    ex.drive_state.create("job-1", "tok-1")
    bystander = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
        env={**os.environ, "TSUGITE_JOB_ID": "job-1"},
    )
    try:
        await ex.cancel(job)
        await asyncio.sleep(0.5)
        assert _alive(bystander.pid), "a process carrying only the job id must be left alone"
    finally:
        bystander.kill()
        bystander.wait(timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["running", "stuck"])
async def test_cancel_reaps_by_token_on_every_finalize(tmp_path, runtime, monkeypatch, state):
    manager, store = runtime
    orch = FakeOrchestrator()
    job = FakeJob("job-1", workspace_path=str(tmp_path))
    job.state = state
    ex = _executor(tmp_path, orch)

    session = spawn_terminal(store=store, manager=manager, cmd="sleep 30", cwd=str(tmp_path), sandbox_ctx=None)
    st = ex.drive_state.create("job-1", "tok-1")
    st.terminal_id = session.id

    reaped = []
    monkeypatch.setattr(executor_mod, "reap_job_processes", lambda token: reaped.append(token) or [])

    await ex.cancel(job)

    assert reaped == ["tok-1"], f"a {state} finalize must reap the job's detached processes"


@pytest.mark.asyncio
async def test_spawn_marks_the_worker_environment_with_the_job_token(tmp_path, runtime, monkeypatch):
    """The reaper matches on the token every descendant inherits, setsid or not."""
    manager, store = runtime
    orch = FakeOrchestrator()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    job = FakeJob("job-1", workspace_path=str(workspace))
    orch._jobs.add(job)

    config_dir = tmp_path / "cfgdir"
    _write_trust_config(config_dir, workspace)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

    captured = {}
    import tsugite_pty.terminal_runtime as tr_mod

    real_spawn = tr_mod.spawn_terminal

    def spy_spawn(**kwargs):
        captured["env"] = kwargs.get("env")
        return real_spawn(**kwargs)

    monkeypatch.setattr(tr_mod, "spawn_terminal", spy_spawn)

    fake = tmp_path / "claude"
    fake.write_text("#!/bin/sh\nsleep 30\n")
    fake.chmod(0o755)
    ex = _executor(tmp_path, orch, claude_binary=str(fake), sandbox=False)

    await ex.start(job, followup=None)

    assert captured["env"]["TSUGITE_JOB_ID"] == "job-1"
    assert captured["env"]["TSUGITE_JOB_TOKEN"] == ex.drive_state.get("job-1").token

    manager.kill(ex.drive_state.get("job-1").terminal_id)
