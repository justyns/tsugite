"""Sandboxing of daemon job shell predicates (routes through the sandbox seam)."""

import pytest

from tsugite.core.sandbox import sandbox_available


def test_unsandboxed_job_predicate_runs_via_shell(tmp_path):
    from tsugite_daemon.job_predicates import _evaluate_predicate

    (tmp_path / "inside.txt").write_text("x")
    ok = _evaluate_predicate(
        {"kind": "cmd", "cmd": "test -f inside.txt"},
        cwd=str(tmp_path),
        ac_index=0,
        ac_text="file present",
        attempt=1,
    )
    assert ok["pass"] is True


@pytest.mark.skipif(not sandbox_available(), reason="sandbox backend not available")
def test_sandboxed_job_predicate_runs_in_bwrap(tmp_path):
    """A sandboxed job's shell predicate runs inside bwrap: it sees the
    worktree (cwd) but cannot read outside it."""
    from tsugite_daemon.job_predicates import _evaluate_predicate

    (tmp_path / "inside.txt").write_text("x")
    override = {
        "enabled": True,
        "no_network": True,
        "allow_domains": [],
        "extra_ro_binds": [],
        "extra_rw_binds": [],
    }

    ok = _evaluate_predicate(
        {"kind": "cmd", "cmd": "test -f inside.txt"},
        cwd=str(tmp_path),
        ac_index=0,
        ac_text="file present",
        attempt=1,
        sandbox_override=override,
    )
    assert ok["pass"] is True

    # /etc/shadow is not bound into the sandbox -> cat fails -> non-zero exit.
    escaped = _evaluate_predicate(
        {"kind": "cmd", "cmd": "cat /etc/shadow"},
        cwd=str(tmp_path),
        ac_index=1,
        ac_text="cannot read host",
        attempt=1,
        sandbox_override=override,
    )
    assert escaped["pass"] is False


@pytest.mark.skipif(not sandbox_available(), reason="sandbox backend not available")
def test_sandboxed_job_setup_command_runs_in_bwrap(tmp_path):
    from tsugite_daemon.job_predicates import run_job_command

    override = {"enabled": True, "no_network": True, "extra_ro_binds": [], "extra_rw_binds": []}

    run_job_command("touch made_inside", str(tmp_path), 30, override)
    assert (tmp_path / "made_inside").exists()

    # Only a fresh pid namespace has bwrap as pid 1.
    run_job_command("grep -qx bwrap /proc/1/comm", str(tmp_path), 30, override)


def test_job_command_in_a_missing_directory_raises_job_command_error(tmp_path):
    from tsugite_daemon.job_predicates import JobCommandError, run_job_command

    with pytest.raises(JobCommandError):
        run_job_command("true", str(tmp_path / "gone"), 30, None)


def test_job_command_timeout_keeps_output_and_kills_the_process_group(tmp_path):
    import os
    import time

    from tsugite_daemon.job_predicates import JobCommandError, run_job_command

    with pytest.raises(JobCommandError, match="timed out") as err:
        run_job_command("echo started; sleep 30 & echo $! > child.pid; wait", str(tmp_path), 1, None)

    assert "started" in err.value.output
    child = int((tmp_path / "child.pid").read_text())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)
