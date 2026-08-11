"""tsugite-onlyoffice packaging + adapter factory contract."""

import importlib.metadata as md
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

VALID_CONFIG = {
    "enabled": True,
    "server_url": "https://onlyoffice.example.net",
    "jwt_secret_name": "onlyoffice-jwt-secret",
    "public_base_url": "https://tsugite.example.net",
    "documents_dir": "/srv/documents",
}


def _create(config):
    from tsugite_onlyoffice.adapter import create_adapter

    return create_adapter(config=config, agents_config={}, session_store=MagicMock(), identity_map={})


def test_adapter_entry_point_registered():
    assert "onlyoffice" in {ep.name for ep in md.entry_points(group="tsugite.adapters")}


def test_tool_plugin_entry_point_registered():
    assert "onlyoffice" in {ep.name for ep in md.entry_points(group="tsugite.plugins")}


def test_absent_config_returns_none():
    assert _create({}) is None
    assert _create({"enabled": False, **{k: v for k, v in VALID_CONFIG.items() if k != "enabled"}}) is None


def test_enabled_without_required_keys_raises():
    """Half a block is a mistake in daemon.yaml, and raising is what gets the daemon to say
    so. Returning None files it under nobody turned the plugin on."""
    for incomplete in (
        {"enabled": True},
        {"enabled": True, "server_url": "https://onlyoffice.example.net"},
        {k: v for k, v in VALID_CONFIG.items() if k != "documents_dir"},
        {k: v for k, v in VALID_CONFIG.items() if k != "public_base_url"},
    ):
        with pytest.raises(ValidationError):
            _create(incomplete)


def test_valid_config_returns_adapter():
    adapter = _create(VALID_CONFIG)
    assert adapter is not None
    assert adapter.config.documents_dir == Path("/srv/documents")


def test_unknown_key_raises():
    """extra="forbid" is there to catch a typo in daemon.yaml, and the key it names has to
    reach whoever typed it."""
    with pytest.raises(ValidationError) as raised:
        _create({**VALID_CONFIG, "documnets_dir": "/srv/documents"})
    assert "documnets_dir" in str(raised.value)


def test_a_config_nobody_turned_on_is_not_read_at_all():
    """The switch is above the parse, so a block left half-written next to a false stays
    the opt-out it looks like."""
    assert _create({**VALID_CONFIG, "enabled": False, "documnets_dir": "/srv/documents"}) is None


def test_tools_module_imports_without_daemon_or_starlette():
    """Every process that lists tools imports this, and none of them needs the daemon half."""
    probe = (
        "import sys, tsugite_onlyoffice.tools as t;"
        "assert 'tsugite_daemon' not in sys.modules, 'tools.py pulled in tsugite_daemon';"
        "assert 'starlette' not in sys.modules, 'tools.py pulled in starlette';"
        "t.set_onlyoffice_runtime(None)"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)
