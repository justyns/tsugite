"""Shared fixtures for the forge plugin's tests."""

import pytest


@pytest.fixture(autouse=True)
def isolate_home(tmp_path, monkeypatch):
    """Keep config and data lookups out of the real home directory."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))


@pytest.fixture
def clock():
    from forge_helpers import Clock

    return Clock()


@pytest.fixture
def secrets_backend():
    from forge_helpers import StubSecretBackend

    from tsugite.secrets import set_backend
    from tsugite.secrets.registry import get_registry

    set_backend(StubSecretBackend())
    yield
    set_backend(None)
    get_registry().clear()


@pytest.fixture(autouse=True)
def reset_history_backend():
    """Drop the cached history backend and connections; each test's XDG_DATA_HOME holds a fresh db."""
    from tsugite.history import reset_history_backend
    from tsugite.history.sqlite_conn import close_all

    reset_history_backend()
    yield
    close_all()
    reset_history_backend()


@pytest.fixture(autouse=True)
def reset_link_resolvers():
    """Clear the link resolver create_adapter registers."""
    from tsugite.link_previews import reset_link_resolvers

    reset_link_resolvers()
    yield
    reset_link_resolvers()
