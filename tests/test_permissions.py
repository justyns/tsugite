"""Tests for the two-source permissions store."""

import pytest
import yaml

from tsugite.permissions import Permissions, get_permissions, set_permissions


@pytest.fixture(autouse=True)
def _clear_permissions():
    """Reset the permissions contextvar before/after each test."""
    set_permissions(None)
    yield
    set_permissions(None)


def _write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


class TestContextVar:
    def test_default_none(self):
        assert get_permissions() is None

    def test_set_get(self, tmp_path):
        perms = Permissions(tmp_path / "permissions.yaml")
        set_permissions(perms)
        assert get_permissions() is perms


class TestMissingFiles:
    def test_missing_runtime_not_allowed(self, tmp_path):
        perms = Permissions(tmp_path / "permissions.yaml")
        assert perms.is_allowed("web.fetch_allowlist", "example.com") is False
        assert perms.web_fetch_allowed("example.com") is False

    def test_empty_runtime_not_allowed(self, tmp_path):
        rt = tmp_path / "permissions.yaml"
        rt.write_text("", encoding="utf-8")
        perms = Permissions(rt)
        assert perms.web_fetch_allowed("example.com") is False


class TestUnion:
    def test_runtime_only(self, tmp_path):
        rt = tmp_path / "permissions.yaml"
        _write_yaml(rt, {"web": {"fetch_allowlist": ["daemon.test"]}})
        perms = Permissions(rt, workspace_dir=tmp_path / "ws")
        assert perms.web_fetch_allowed("daemon.test")
        assert not perms.web_fetch_allowed("other.test")

    def test_workspace_only(self, tmp_path):
        ws = tmp_path / "ws"
        _write_yaml(ws / ".tsugite" / "permissions.yaml", {"web": {"fetch_allowlist": ["ws.test"]}})
        perms = Permissions(tmp_path / "permissions.yaml", workspace_dir=ws)
        assert perms.web_fetch_allowed("ws.test")

    def test_either_source(self, tmp_path):
        rt = tmp_path / "permissions.yaml"
        ws = tmp_path / "ws"
        _write_yaml(rt, {"web": {"fetch_allowlist": ["daemon.test"]}})
        _write_yaml(ws / ".tsugite" / "permissions.yaml", {"web": {"fetch_allowlist": ["ws.test"]}})
        perms = Permissions(rt, workspace_dir=ws)
        assert perms.web_fetch_allowed("daemon.test")
        assert perms.web_fetch_allowed("ws.test")
        assert not perms.web_fetch_allowed("nope.test")


class TestAllow:
    def test_allow_then_is_allowed(self, tmp_path):
        perms = Permissions(tmp_path / "permissions.yaml")
        assert not perms.web_fetch_allowed("new.test")
        perms.web_fetch_allow("new.test")
        assert perms.web_fetch_allowed("new.test")

    def test_allow_persists_to_disk(self, tmp_path):
        rt = tmp_path / "permissions.yaml"
        Permissions(rt).web_fetch_allow("persist.test")
        # A fresh instance sees the write, proving it hit disk.
        assert Permissions(rt).web_fetch_allowed("persist.test")
        assert yaml.safe_load(rt.read_text()) == {"web": {"fetch_allowlist": ["persist.test"]}}

    def test_allow_writes_runtime_only(self, tmp_path):
        rt = tmp_path / "permissions.yaml"
        ws = tmp_path / "ws"
        ws_file = ws / ".tsugite" / "permissions.yaml"
        _write_yaml(ws_file, {"web": {"fetch_allowlist": ["ws.test"]}})
        ws_before = ws_file.read_text()

        perms = Permissions(rt, workspace_dir=ws)
        perms.web_fetch_allow("runtime.test")

        assert ws_file.read_text() == ws_before  # workspace file never mutated
        assert perms.web_fetch_allowed("runtime.test")
        assert perms.web_fetch_allowed("ws.test")  # union still holds

    def test_allow_preserves_unrelated_sections(self, tmp_path):
        rt = tmp_path / "permissions.yaml"
        _write_yaml(rt, {"web": {"fetch_allowlist": ["a.test"]}, "other": {"keep": [1, 2]}})
        perms = Permissions(rt)
        perms.web_fetch_allow("b.test")
        data = yaml.safe_load(rt.read_text())
        assert data["other"] == {"keep": [1, 2]}
        assert set(data["web"]["fetch_allowlist"]) == {"a.test", "b.test"}

    def test_allow_idempotent(self, tmp_path):
        rt = tmp_path / "permissions.yaml"
        perms = Permissions(rt)
        perms.web_fetch_allow("dupe.test")
        perms.web_fetch_allow("dupe.test")
        assert yaml.safe_load(rt.read_text()) == {"web": {"fetch_allowlist": ["dupe.test"]}}

    def test_domain_case_insensitive(self, tmp_path):
        rt = tmp_path / "permissions.yaml"
        perms = Permissions(rt)
        perms.web_fetch_allow("Example.COM")
        assert perms.web_fetch_allowed("example.com")
        assert yaml.safe_load(rt.read_text()) == {"web": {"fetch_allowlist": ["example.com"]}}

    def test_generic_is_allowed_and_allow(self, tmp_path):
        perms = Permissions(tmp_path / "permissions.yaml")
        assert not perms.is_allowed("web.fetch_allowlist", "g.test")
        perms.allow("web.fetch_allowlist", "g.test")
        assert perms.is_allowed("web.fetch_allowlist", "g.test")


class TestAtomicWrite:
    def test_no_temp_litter(self, tmp_path):
        rt = tmp_path / "permissions.yaml"
        Permissions(rt).web_fetch_allow("x.test")
        assert [f for f in rt.parent.iterdir() if f.suffix == ".tmp"] == []

    def test_failed_write_preserves_original(self, tmp_path, monkeypatch):
        rt = tmp_path / "permissions.yaml"
        _write_yaml(rt, {"web": {"fetch_allowlist": ["orig.test"]}})
        original = rt.read_text()

        def boom(*args, **kwargs):
            raise OSError("simulated replace failure")

        monkeypatch.setattr("tsugite.utils.os.replace", boom)

        perms = Permissions(rt)
        with pytest.raises(OSError):
            perms.web_fetch_allow("new.test")

        assert rt.read_text() == original  # untouched, not truncated
        assert [f for f in rt.parent.iterdir() if f.suffix == ".tmp"] == []
