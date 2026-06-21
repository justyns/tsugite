"""The executor backend is swappable via the tsugite.executors seam."""

import pytest

from tsugite.core.executor import LocalExecutor
from tsugite.core.executor_registry import get_executor_class
from tsugite.core.subprocess_executor import SubprocessExecutor


def test_default_is_subprocess(monkeypatch):
    """With no env/config override, production defaults to the subprocess executor."""
    monkeypatch.delenv("TSUGITE_EXECUTOR_BACKEND", raising=False)
    import tsugite.core.executor_registry as reg

    monkeypatch.setattr(reg, "_configured_backend", lambda: "subprocess")
    assert get_executor_class() is SubprocessExecutor


def test_env_selects_local(monkeypatch):
    monkeypatch.setenv("TSUGITE_EXECUTOR_BACKEND", "local")
    assert get_executor_class() is LocalExecutor


def test_explicit_backend_arg_wins(monkeypatch):
    monkeypatch.setenv("TSUGITE_EXECUTOR_BACKEND", "local")
    assert get_executor_class("subprocess") is SubprocessExecutor


def test_plugin_backend_resolves(monkeypatch):
    """An unknown name resolves via the tsugite.executors entry-point group."""
    import tsugite.plugins as plugins

    class _RemoteExecutor:
        pass

    class _EP:
        name = "k8s"

        def load(self):
            return _RemoteExecutor

    real = plugins.importlib.metadata.entry_points
    monkeypatch.setattr(
        plugins.importlib.metadata,
        "entry_points",
        lambda **kw: [_EP()] if kw.get("group") == "tsugite.executors" else real(**kw),
    )
    assert get_executor_class("k8s") is _RemoteExecutor


def test_unknown_backend_raises(monkeypatch):
    with pytest.raises(ValueError, match="Unknown executor backend"):
        get_executor_class("does-not-exist")
