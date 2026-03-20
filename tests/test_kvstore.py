"""Tests for the KV store."""

import time

import pytest

from tsugite.kvstore import _create_backend, get_backend, set_backend
from tsugite.kvstore.sqlite import SqliteKVBackend
from tsugite.tools.kv import _resolve_namespace


@pytest.fixture
def backend(tmp_path):
    return SqliteKVBackend(db_path=tmp_path / "test.db")


class TestSqliteKVBackend:
    def test_get_set(self, backend):
        backend.set("ns", "key1", "value1")
        assert backend.get("ns", "key1") == "value1"

    def test_get_missing(self, backend):
        assert backend.get("ns", "missing") is None

    def test_overwrite(self, backend):
        backend.set("ns", "key1", "v1")
        backend.set("ns", "key1", "v2")
        assert backend.get("ns", "key1") == "v2"

    def test_delete(self, backend):
        backend.set("ns", "key1", "value1")
        assert backend.delete("ns", "key1") is True
        assert backend.get("ns", "key1") is None

    def test_delete_missing(self, backend):
        assert backend.delete("ns", "missing") is False

    def test_list_keys(self, backend):
        backend.set("ns", "a1", "v")
        backend.set("ns", "a2", "v")
        backend.set("ns", "b1", "v")
        assert backend.list_keys("ns") == ["a1", "a2", "b1"]

    def test_list_keys_prefix(self, backend):
        backend.set("ns", "user:1", "v")
        backend.set("ns", "user:2", "v")
        backend.set("ns", "task:1", "v")
        assert backend.list_keys("ns", "user:") == ["user:1", "user:2"]

    def test_namespace_isolation(self, backend):
        backend.set("ns1", "key", "val1")
        backend.set("ns2", "key", "val2")
        assert backend.get("ns1", "key") == "val1"
        assert backend.get("ns2", "key") == "val2"
        assert backend.list_keys("ns1") == ["key"]

    def test_ttl_expiry(self, backend):
        backend.set("ns", "key1", "value1", ttl_seconds=1)
        assert backend.get("ns", "key1") == "value1"
        time.sleep(1.1)
        assert backend.get("ns", "key1") is None

    def test_ttl_filters_list_keys(self, backend):
        backend.set("ns", "alive", "v")
        backend.set("ns", "expired", "v", ttl_seconds=1)
        time.sleep(1.1)
        assert backend.list_keys("ns") == ["alive"]

    def test_list_namespaces(self, backend):
        backend.set("alpha", "k", "v")
        backend.set("beta", "k", "v")
        assert backend.list_namespaces() == ["alpha", "beta"]

    def test_get_with_metadata(self, backend):
        backend.set("ns", "key1", "value1", ttl_seconds=60)
        result = backend.get_with_metadata("ns", "key1")
        assert result["value"] == "value1"
        assert result["expires_at"] is not None
        assert result["expires_at"] > int(time.time())

    def test_get_with_metadata_missing(self, backend):
        assert backend.get_with_metadata("ns", "missing") is None


class TestResolveNamespace:
    def test_explicit(self):
        assert _resolve_namespace("my-ns") == "my-ns"

    def test_default_fallback(self):
        assert _resolve_namespace(None) == "default"

    def test_agent_fallback(self):
        from tsugite.agent_runner.helpers import clear_current_agent, set_current_agent
        set_current_agent("test-agent")
        try:
            assert _resolve_namespace(None) == "test-agent"
        finally:
            clear_current_agent()


class TestBackendFactory:
    def test_default_sqlite(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TSUGITE_KV_BACKEND", raising=False)
        backend = _create_backend()
        assert isinstance(backend, SqliteKVBackend)

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("TSUGITE_KV_BACKEND", "nonexistent")
        with pytest.raises(ValueError, match="Unknown KV backend"):
            _create_backend()

    def test_config_override(self, monkeypatch):
        monkeypatch.delenv("TSUGITE_KV_BACKEND", raising=False)
        with pytest.raises(ValueError, match="Unknown KV backend"):
            _create_backend({"backend": "nonexistent"})

    def test_get_set_backend(self, backend):
        set_backend(backend)
        assert get_backend() is backend
        set_backend(None)
