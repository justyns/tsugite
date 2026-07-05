"""Daemon logging must be persistent by default and capture crash tracebacks.

A daemon crash previously left no retrievable traceback: with no log_file
configured nothing was written anywhere durable, and unhandled main-thread /
worker-thread exceptions only ever reached the ephemeral stderr."""

import logging
import logging.handlers
import sys
import threading

import pytest
from tsugite_daemon.config import DaemonConfig
from tsugite_daemon.gateway import _configure_logging, _install_crash_hooks


@pytest.fixture(autouse=True)
def _restore_logging_and_hooks():
    root = logging.getLogger()
    prev_handlers = root.handlers[:]
    prev_level = root.level
    prev_excepthook = sys.excepthook
    prev_threading_hook = threading.excepthook
    yield
    for h in root.handlers[:]:
        root.removeHandler(h)
    for h in prev_handlers:
        root.addHandler(h)
    root.setLevel(prev_level)
    sys.excepthook = prev_excepthook
    threading.excepthook = prev_threading_hook


def _file_handlers():
    return [h for h in logging.getLogger().handlers if isinstance(h, logging.handlers.RotatingFileHandler)]


def test_log_file_defaults_to_state_dir(tmp_path):
    """No log_file configured must still produce a persistent rotating log
    under state_dir - otherwise a crash traceback is unrecoverable."""
    config = DaemonConfig(state_dir=tmp_path / "state", agents={})
    _configure_logging(config)
    handlers = _file_handlers()
    assert handlers, "a persistent file handler must be installed by default"
    assert handlers[0].baseFilename == str(tmp_path / "state" / "daemon.log")


def test_explicit_log_file_still_honored(tmp_path):
    config = DaemonConfig(state_dir=tmp_path / "state", agents={}, log_file=tmp_path / "custom" / "d.log")
    _configure_logging(config)
    handlers = _file_handlers()
    assert handlers
    assert handlers[0].baseFilename == str(tmp_path / "custom" / "d.log")


def test_crash_hooks_write_tracebacks_to_log(tmp_path):
    """Unhandled main-thread and worker-thread exceptions must land in the log
    file with a traceback (the agent loop runs in worker threads via to_thread)."""
    config = DaemonConfig(state_dir=tmp_path / "state", agents={})
    _configure_logging(config)
    _install_crash_hooks()

    try:
        raise ValueError("main thread boom")
    except ValueError:
        sys.excepthook(*sys.exc_info())

    def _worker():
        raise RuntimeError("worker thread boom")

    t = threading.Thread(target=_worker, name="agent-worker")
    t.start()
    t.join()

    for h in logging.getLogger().handlers:
        h.flush()
    text = (tmp_path / "state" / "daemon.log").read_text()
    assert "main thread boom" in text
    assert "ValueError" in text
    assert "worker thread boom" in text
    assert "agent-worker" in text
    assert "Traceback" in text
