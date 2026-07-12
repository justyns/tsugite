"""CCExecutor: command building, sandbox policy, live-PTY followup, spawn, cancel.

Uses a real PtyManager + TerminalSessionStore (tmp) and a fake `claude` shell
script - no real claude, no network.
"""

import asyncio
import json
from pathlib import Path

import pytest
from tsugite_cc_driver.adapter import CCDriverConfig
from tsugite_cc_driver.executor import (
    CCExecutor,
    DriveStateStore,
    build_claude_command,
    build_sandbox_ctx,
    is_workspace_trusted,
)
from tsugite_cc_driver.settings import build_settings, write_run_settings
from tsugite_pty.pty_manager import PtyManager
from tsugite_pty.terminal_runtime import spawn_terminal
from tsugite_pty.terminal_store import TerminalSessionStore


class FakeJob:
    def __init__(self, job_id, *, prompt="do the thing", workspace_path=None, worktree_path=None, model=None):
        self.id = job_id
        self.state = "running"
        self.prompt = prompt
        self.workspace_path = workspace_path
        self.worktree_path = worktree_path
        self.worker_terminal_id = None
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

    def get_job(self, job_id):
        return self._jobs.get(job_id)

    def attach_worker_terminal(self, job_id, terminal_id):
        self._jobs.update(job_id, worker_terminal_id=terminal_id)

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


# ── is_workspace_trusted ──


def test_is_workspace_trusted_true_when_accepted(tmp_path):
    cwd = tmp_path / "ws"
    cwd.mkdir()
    cfg = _write_trust_config(tmp_path / "cfg", cwd)
    assert is_workspace_trusted(cwd, config_path=cfg) is True


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


# ── pure helpers ──


def test_build_claude_command_quotes_goal_with_quotes_and_newlines():
    prompt = "fix the 'bug'\nand add a test"
    cmd = build_claude_command("claude", "/s/settings.json", "bypassPermissions", prompt)
    # The whole prompt is a single shell token; re-splitting must recover it verbatim.
    import shlex

    argv = shlex.split(cmd)
    assert argv[0] == "claude"
    assert "--settings" in argv and "/s/settings.json" in argv
    assert "--permission-mode" in argv and "bypassPermissions" in argv
    assert argv[-1] == prompt, "goal must survive shell quoting intact"


def test_build_claude_command_adds_resume_when_given():
    cmd = build_claude_command("claude", "/s.json", "bypassPermissions", "go", resume_session_id="cc-42")
    import shlex

    argv = shlex.split(cmd)
    assert "--resume" in argv and "cc-42" in argv


def test_build_claude_command_adds_model_when_given():
    import shlex

    argv = shlex.split(build_claude_command("claude", "/s.json", "bypassPermissions", "go", model="opus"))
    assert "--model" in argv and "opus" in argv
    # None -> no --model flag (claude uses its own default).
    assert "--model" not in shlex.split(build_claude_command("claude", "/s.json", "bypassPermissions", "go"))


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
async def test_untrusted_workspace_fails_worker_without_spawning(tmp_path, runtime, monkeypatch):
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
    ex = _executor(tmp_path, orch, claude_binary=str(fake))

    await ex.start(job, followup=None)

    assert orch.failed and "not trusted" in orch.failed[0][1]
    assert store.list_all() == [], "an untrusted workspace must not spawn a PTY (it would hang on the trust dialog)"
    assert ex.drive_state.get("job-1") is None


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
    settings = json.loads(Path(st.settings_path).read_text())
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
