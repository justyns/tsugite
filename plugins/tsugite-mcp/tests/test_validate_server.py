"""Server name and URL validation."""

import pytest
from tsugite_mcp.adapter import create_adapter, validate_server


@pytest.mark.parametrize(
    "name, url, valid",
    [
        ("notion", "https://mcp.notion.com/mcp", True),
        ("local", "http://localhost:8000/mcp", True),
        ("loop", "http://127.0.0.1:8000/mcp", True),
        ("Bad", "https://mcp.example.com", False),
        ("two_words", "https://mcp.example.com", False),
        ("", "https://mcp.example.com", False),
        ("remote", "http://mcp.example.com/mcp", False),
        ("nohost", "https://", False),
        ("file", "file:///etc/passwd", False),
    ],
)
def test_validate_server(name, url, valid):
    if valid:
        validate_server(name, url)
    else:
        with pytest.raises(ValueError):
            validate_server(name, url)


@pytest.mark.parametrize(("config", "active"), [(None, False), ({"enabled": False}, False), ({"enabled": True}, True)])
def test_adapter_loads_only_when_enabled(config, active):
    adapter = create_adapter(config=config, runtime=None, session_store=None, identity_map=None)
    assert (adapter is not None) == active
