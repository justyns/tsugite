"""Tests for the `tsu exec` command - run Python in tsugite's tool namespace.

`tsu exec` lets an external orchestrator (e.g. Claude Code) reuse a tsugite
skill's Python verbatim: tsugite tools like read_file/get_secret are available as
functions, secrets stay allowlisted + masked, and the snippet can be sandboxed.
"""

import pytest
from typer.testing import CliRunner

from tsugite.cli import app
from tsugite.core.sandbox import sandbox_available

runner = CliRunner()


@pytest.fixture(autouse=True)
def _register_exec_tools(reset_tool_registry):
    """Re-register the tools these tests touch.

    The parent conftest clears the tool registry per-test, and cached module
    imports stop `_ensure_tools_loaded()` from re-running @tool decorators. The
    secret tools must keep parent_only=True so they run in the parent process,
    where the secret allowlist + masking registry live.
    """
    from tsugite_web import web_search

    from tsugite.tools import tool
    from tsugite.tools.fs import (
        create_directory,
        edit_file,
        file_exists,
        get_file_info,
        list_files,
        read_file,
        write_file,
    )
    from tsugite.tools.http import check_url, download_file, fetch_json, fetch_text, http_request
    from tsugite.tools.secrets import get_secret, list_secrets

    for fn in [
        read_file,
        write_file,
        list_files,
        file_exists,
        create_directory,
        get_file_info,
        edit_file,
        check_url,
        download_file,
        fetch_json,
        fetch_text,
        http_request,
        web_search,
    ]:
        tool(fn)
    tool(parent_only=True)(get_secret)
    tool(parent_only=True)(list_secrets)


def test_exec_plain_value_from_stdin():
    """A trailing expression is evaluated and its value printed (like python -c)."""
    result = runner.invoke(app, ["exec", "-"], input="result = 1 + 1\nresult\n")
    assert result.exit_code == 0, result.output
    assert "2" in result.output


def test_exec_tsugite_tool_in_scope(tmp_path):
    """tsugite tools (here read_file) are callable inside the snippet."""
    target = tmp_path / "hello.txt"
    target.write_text("hi there")
    result = runner.invoke(app, ["exec", "-", "--tools", "@fs"], input=f"read_file(path={str(target)!r})\n")
    assert result.exit_code == 0, result.output
    assert "hi there" in result.output


@pytest.fixture
def env_secrets(monkeypatch):
    """Force the env secrets backend and seed two secrets.

    The backend is a process-global lazy singleton, so under `-n auto` a backend cached
    by an earlier test in the same worker would shadow these env vars (get_secret would
    miss). Install the env backend explicitly, and drop it on teardown so it doesn't leak.
    """
    monkeypatch.setenv("TSUGITE_SECRETS_BACKEND", "env")
    monkeypatch.setenv("MY_TOKEN", "supersecret123")
    monkeypatch.setenv("OTHER_TOKEN", "othervalue456")

    from tsugite.secrets import set_backend
    from tsugite.secrets.env import EnvSecretBackend

    set_backend(EnvSecretBackend(prefix=""))
    yield
    set_backend(None)  # force a fresh re-create for the next test in this worker


def test_exec_secret_success_and_masked(env_secrets):
    """get_secret resolves an allowlisted secret; its value is masked in output."""
    result = runner.invoke(
        app,
        ["exec", "-", "--allow-secret", "my-token"],
        input='token = get_secret(name="my-token")\nprint("token is", token)\n',
    )
    assert result.exit_code == 0, result.output
    assert "supersecret123" not in result.output
    assert "***" in result.output


def test_exec_secret_denied_by_allowlist(env_secrets):
    """A secret outside --allow-secret raises and the value never resolves."""
    result = runner.invoke(
        app,
        ["exec", "-", "--allow-secret", "my-token"],
        input='get_secret(name="other-token")\n',
    )
    assert result.exit_code == 1
    assert "other-token" in result.output
    assert "othervalue456" not in result.output


def test_exec_agent_inherits_allowed_secrets(tmp_path, env_secrets):
    """--agent inherits that agent's allowed_secrets (here scoped to my-token only)."""
    agent = tmp_path / "scoped.md"
    agent.write_text("---\nname: scoped\nextends: none\ntools: [get_secret]\nallowed_secrets: [my-token]\n---\nBody.\n")
    # other-token is outside the agent's allowlist -> denied. (The bare-exec default
    # allowlist is empty == all allowed, so a denial proves inheritance is applied.)
    result = runner.invoke(
        app,
        ["exec", "-", "--agent", str(agent)],
        input='get_secret(name="other-token")\n',
    )
    assert result.exit_code == 1
    assert "other-token" in result.output


def test_exec_tools_comma_or_space_separated():
    """--tools takes comma- or space-separated specs in one flag, not only repeats."""
    for spec in ("@fs,@http", "@fs @http"):
        result = runner.invoke(app, ["exec", "-", "--tools", spec], input="read_file\nhttp_request\nprint('both')\n")
        assert result.exit_code == 0, f"{spec!r}: {result.output}"
        assert "both" in result.output


@pytest.mark.skipif(not sandbox_available(), reason="sandbox backend not available")
def test_exec_sandbox_no_network_blocks_egress():
    """--no-network implies --sandbox and blocks outbound connections."""
    result = runner.invoke(
        app,
        ["exec", "-", "--no-network"],
        input='import socket\nsocket.create_connection(("1.1.1.1", 53), timeout=5)\n',
    )
    assert result.exit_code == 1, result.output
