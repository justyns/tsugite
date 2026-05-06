"""Tests for the now() helper tool."""

from datetime import datetime

import pytest

from tsugite.tools import call_tool, tool
from tsugite.tools.time import now


@pytest.fixture
def time_tools(reset_tool_registry):
    """Re-register time tools after the autouse registry reset."""
    tool(now)


def test_now_returns_iso_string_with_timezone(time_tools):
    result = call_tool("now")
    assert isinstance(result, str)
    parsed = datetime.fromisoformat(result)
    assert parsed.tzinfo is not None, "now() must return a timezone-aware ISO string"


def test_now_is_close_to_current_time(time_tools):
    before = datetime.now().astimezone()
    result = call_tool("now")
    after = datetime.now().astimezone()
    parsed = datetime.fromisoformat(result)
    assert before <= parsed <= after


def test_now_is_discoverable_via_default_loader():
    """The standard tool loader (`_ensure_tools_loaded`) must wire up time.py
    so agents can call `now` without a custom fixture path. Without this the
    tool registers in tests but goes missing in production. Run in a fresh
    subprocess to bypass the autouse registry reset and Python import cache.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "from tsugite.tools import list_tools; print('now' in list_tools())"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "True"
