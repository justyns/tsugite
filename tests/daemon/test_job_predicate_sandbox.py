"""Sandboxing of daemon job shell predicates (routes through the sandbox seam)."""

import pytest

from tsugite.core.sandbox import sandbox_available


def test_unsandboxed_job_predicate_runs_via_shell(tmp_path):
    from tsugite_daemon.jobs_orchestrator import _evaluate_predicate

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
    from tsugite_daemon.jobs_orchestrator import _evaluate_predicate

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
