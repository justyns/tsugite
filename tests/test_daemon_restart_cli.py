"""`tsu daemon` re-execs itself when the daemon asks to be restarted.

os.execv is always patched here: unpatched, it would replace the pytest process.
"""

import logging
import os
import sys

import pytest

from tsugite.cli import app


@pytest.fixture
def spy(monkeypatch):
    """Record the re-exec (and the log flush before it) instead of performing it."""
    calls = []
    monkeypatch.setattr(os, "execv", lambda path, argv: calls.append(("execv", path, argv)))
    monkeypatch.setattr(logging, "shutdown", lambda *a: calls.append(("logging.shutdown",)))
    return calls


def _patch_run_daemon(monkeypatch, result=None, exc=None):
    async def fake_run_daemon(config, config_overrides=None):
        if exc:
            raise exc
        return result

    monkeypatch.setattr("tsugite_daemon.gateway.run_daemon", fake_run_daemon)


def test_restart_flag_re_execs_the_process(cli_runner, monkeypatch, spy):
    _patch_run_daemon(monkeypatch, result=True)

    result = cli_runner.invoke(app, ["daemon"])

    assert result.exit_code == 0, result.output
    assert [c[0] for c in spy] == ["logging.shutdown", "execv"], "the log must be flushed before exec discards it"
    _, path, argv = spy[-1]
    assert path == sys.executable
    assert argv[0] == sys.executable


def test_no_restart_flag_does_not_re_exec(cli_runner, monkeypatch, spy):
    _patch_run_daemon(monkeypatch, result=False)

    result = cli_runner.invoke(app, ["daemon"])

    assert result.exit_code == 0, result.output
    assert spy == []


@pytest.mark.parametrize(
    "exc",
    [RuntimeError("Event loop stopped before Future completed."), KeyboardInterrupt()],
    ids=["runtime_error", "keyboard_interrupt"],
)
def test_a_swallowed_exit_does_not_re_exec(cli_runner, monkeypatch, spy, exc):
    """A forced shutdown stops the loop (RuntimeError); Ctrl-C raises KeyboardInterrupt.
    Neither assigns the restart flag, so the exit stays clean and nothing re-execs.
    """
    _patch_run_daemon(monkeypatch, exc=exc)

    result = cli_runner.invoke(app, ["daemon"])

    assert result.exit_code == 0, result.exception
    assert spy == []
