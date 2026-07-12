"""CCExecutor: command building, sandbox policy, live-PTY followup, spawn, cancel.

Uses a real PtyManager + TerminalSessionStore (tmp) and a fake `claude` shell
script - no real claude, no network.
"""

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
    def __init__(self, job_id, *, prompt="do the thing", workspace_path=None, worktree_path=None):
        self.id = job_id
        self.state = "running"
        self.prompt = prompt
        self.workspace_path = workspace_path
        self.worktree_path = worktree_path
        self.worker_terminal_id = None


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

    async def fail_worker(self, job_id, error):
        self.failed.append((job_id, error))

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


def test_build_sandbox_ctx_off_is_none():
    assert build_sandbox_ctx(False, "/work") is None


def test_build_sandbox_ctx_on_keeps_network_and_binds_claude_dir():
    ctx = build_sandbox_ctx(True, "/work")
    assert ctx is not None
    assert ctx.no_network is False, "the driven claude needs network for the API"
    assert Path.home() / ".claude" in ctx.extra_rw_binds
    assert ctx.workspace_dir == Path("/work")


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

    await ex.start(job, followup="please keep going")

    proc = manager.get(session.id)
    proc.wait_drain(timeout=1.0)
    assert b"please keep going" in proc.buffer, "followup must be typed into the live PTY"
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
