"""The daemon client CLI must dial the port the daemon actually serves.

Flag > env > default precedence is Typer's, declared via `envvar=` on the
shared options, so it isn't retested here.
"""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from tsugite.cli import app
from tsugite.config import DEFAULT_DAEMON_HOST, DEFAULT_DAEMON_PORT


@pytest.fixture(autouse=True)
def no_daemon_env(monkeypatch):
    for var in ("TSUGITE_DAEMON_HOST", "TSUGITE_DAEMON_PORT", "TSUGITE_DAEMON_TOKEN"):
        monkeypatch.delenv(var, raising=False)


def request_url(argv):
    """Run a daemon command against a stubbed transport and return the URL it dialled."""
    with patch("httpx.request") as mock_request:
        mock_request.return_value.status_code = 200
        mock_request.return_value.json.return_value = {"schedules": [], "sessions": []}
        CliRunner().invoke(app, argv)
    assert mock_request.called, f"{argv} never issued an HTTP request"
    return mock_request.call_args[0][1]


def test_default_matches_the_address_the_daemon_serves():
    """The bug: the CLI defaulted to 8321 while HTTPConfig served 8374. One
    constant now feeds both, so this fails if either side re-hardcodes."""
    from tsugite_daemon.config import HTTPConfig

    served = HTTPConfig()
    assert (DEFAULT_DAEMON_HOST, DEFAULT_DAEMON_PORT) == (served.host, served.port)


@pytest.mark.parametrize(
    "argv",
    [
        ["daemon", "schedule", "list"],
        ["daemon", "session", "list"],
        ["daemon", "sessions", "myagent"],
    ],
)
def test_zero_flag_command_targets_the_served_port(argv):
    assert request_url(argv).startswith(f"http://{DEFAULT_DAEMON_HOST}:{DEFAULT_DAEMON_PORT}")


def test_env_var_overrides_the_default(monkeypatch):
    """One case to prove the options are actually wired to `envvar=`."""
    monkeypatch.setenv("TSUGITE_DAEMON_HOST", "10.0.0.5")
    monkeypatch.setenv("TSUGITE_DAEMON_PORT", "9100")

    assert request_url(["daemon", "schedule", "list"]).startswith("http://10.0.0.5:9100")


def test_flag_overrides_env(monkeypatch):
    monkeypatch.setenv("TSUGITE_DAEMON_PORT", "9100")

    assert request_url(["daemon", "schedule", "list", "--port", "9200"]).startswith("http://127.0.0.1:9200")
