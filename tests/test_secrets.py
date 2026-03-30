"""Tests for the secrets system: backends, registry, masking, tools, CLI."""

import logging
import os
import sqlite3
import stat

import pytest
import typer.testing

from tsugite.secrets.env import EnvSecretBackend
from tsugite.secrets.exec import ExecSecretBackend
from tsugite.secrets.file import FileSecretBackend
from tsugite.secrets.registry import SecretRegistry


# -- Fixtures --


@pytest.fixture
def env_backend():
    return EnvSecretBackend()


@pytest.fixture
def env_backend_prefixed():
    return EnvSecretBackend(prefix="TSU_")


@pytest.fixture
def file_backend(tmp_path):
    return FileSecretBackend({"path": str(tmp_path / "secrets")})


@pytest.fixture
def sqlite_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("TSUGITE_SECRETS_KEY", "test-passphrase")
    from tsugite.secrets.sqlite import SqliteSecretBackend

    return SqliteSecretBackend({"path": str(tmp_path / "secrets.db")})


@pytest.fixture
def registry():
    r = SecretRegistry()
    yield r
    r.clear()


# -- EnvSecretBackend --


class TestEnvSecretBackend:
    def test_get_exact_name(self, env_backend, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "abc123")
        assert env_backend.get("MY_TOKEN") == "abc123"

    def test_get_normalized_name(self, env_backend, monkeypatch):
        monkeypatch.setenv("MY_API_KEY", "secret")
        assert env_backend.get("my-api-key") == "secret"

    def test_get_with_prefix(self, env_backend_prefixed, monkeypatch):
        monkeypatch.setenv("TSU_TOKEN", "prefixed")
        assert env_backend_prefixed.get("token") == "prefixed"

    def test_get_missing(self, env_backend):
        assert env_backend.get("nonexistent_secret_xyz") is None

    def test_list_names_empty(self, env_backend):
        assert env_backend.list_names() == []

    def test_set_raises(self, env_backend):
        with pytest.raises(NotImplementedError):
            env_backend.set("key", "val")

    def test_delete_raises(self, env_backend):
        with pytest.raises(NotImplementedError):
            env_backend.delete("key")


# -- ExecSecretBackend --


class TestExecSecretBackend:
    def test_get_argv(self):
        b = ExecSecretBackend({"command": ["echo", "{{ name }}"]})
        assert b.get("mytoken") == "mytoken"

    def test_get_string(self):
        b = ExecSecretBackend({"command": "echo {{ name }}"})
        assert b.get("mytoken") == "mytoken"

    def test_get_invalid_name(self):
        b = ExecSecretBackend({"command": ["echo", "{{ name }}"]})
        assert b.get("../etc/passwd") is None
        assert b.get("; rm -rf /") is None

    def test_get_failing_command(self):
        b = ExecSecretBackend({"command": ["false"]})
        assert b.get("anything") is None

    def test_list_names_with_command(self):
        b = ExecSecretBackend({"command": "echo x", "list_command": ["echo", "a\nb\nc"]})
        names = b.list_names()
        assert "a" in names

    def test_list_names_without_command(self):
        b = ExecSecretBackend({"command": "echo x"})
        assert b.list_names() == []

    def test_set_raises(self):
        b = ExecSecretBackend({"command": "echo x"})
        with pytest.raises(NotImplementedError):
            b.set("key", "val")

    def test_delete_raises(self):
        b = ExecSecretBackend({"command": "echo x"})
        with pytest.raises(NotImplementedError):
            b.delete("key")

    def test_missing_command_raises(self):
        with pytest.raises(ValueError, match="requires 'command'"):
            ExecSecretBackend({})


# -- FileSecretBackend --


class TestFileSecretBackend:
    def test_set_get_roundtrip(self, file_backend):
        file_backend.set("token", "abc123")
        assert file_backend.get("token") == "abc123"

    def test_list_names_sorted(self, file_backend):
        file_backend.set("b-key", "v")
        file_backend.set("a-key", "v")
        assert file_backend.list_names() == ["a-key", "b-key"]

    def test_delete_existing(self, file_backend):
        file_backend.set("token", "val")
        assert file_backend.delete("token") is True
        assert file_backend.get("token") is None

    def test_delete_missing(self, file_backend):
        assert file_backend.delete("nonexistent") is False

    def test_get_missing(self, file_backend):
        assert file_backend.get("nonexistent") is None

    def test_path_traversal_blocked(self, file_backend):
        with pytest.raises(ValueError, match="Invalid secret name"):
            file_backend.get("../../etc/passwd")

    def test_path_traversal_blocked_set(self, file_backend):
        with pytest.raises(ValueError, match="Invalid secret name"):
            file_backend.set("../evil", "data")

    def test_file_permissions(self, file_backend):
        file_backend.set("token", "secret")
        path = file_backend._dir / "token"
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600


# -- SqliteSecretBackend --


class TestSqliteSecretBackend:
    def test_set_get_roundtrip(self, sqlite_backend):
        sqlite_backend.set("token", "abc123")
        assert sqlite_backend.get("token") == "abc123"

    def test_list_names(self, sqlite_backend):
        sqlite_backend.set("b", "v")
        sqlite_backend.set("a", "v")
        assert sqlite_backend.list_names() == ["a", "b"]

    def test_delete(self, sqlite_backend):
        sqlite_backend.set("token", "val")
        assert sqlite_backend.delete("token") is True
        assert sqlite_backend.get("token") is None
        assert sqlite_backend.delete("token") is False

    def test_wrong_passphrase(self, sqlite_backend, tmp_path, monkeypatch):
        sqlite_backend.set("token", "secret")
        db_path = str(tmp_path / "secrets.db")

        monkeypatch.setenv("TSUGITE_SECRETS_KEY", "wrong-passphrase")
        from tsugite.secrets.sqlite import SqliteSecretBackend

        b2 = SqliteSecretBackend({"path": db_path})
        with pytest.raises(ValueError, match="Wrong passphrase"):
            b2.get("token")

    def test_no_passphrase_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TSUGITE_SECRETS_KEY", raising=False)
        from tsugite.secrets.sqlite import SqliteSecretBackend

        with pytest.raises(ValueError, match="requires a passphrase"):
            SqliteSecretBackend({"path": str(tmp_path / "test.db")})

    def test_values_encrypted_in_db(self, sqlite_backend):
        sqlite_backend.set("token", "plaintext-secret")
        conn = sqlite3.connect(str(sqlite_backend._db_path))
        row = conn.execute("SELECT value FROM secrets WHERE name='token'").fetchone()
        assert row[0] != "plaintext-secret"

    def test_db_permissions(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TSUGITE_SECRETS_KEY", "test")
        from tsugite.secrets.sqlite import SqliteSecretBackend

        db_path = tmp_path / "new.db"
        SqliteSecretBackend({"path": str(db_path)}).set("k", "v")
        mode = stat.S_IMODE(db_path.stat().st_mode)
        assert mode == 0o600


# -- SecretRegistry --


class TestSecretRegistry:
    def test_register_returns_raw_value(self, registry):
        assert registry.register("key", "secret123") == "secret123"

    def test_mask_replaces_values(self, registry):
        registry.register("key", "secret123")
        assert registry.mask("token: secret123") == "token: ***"

    def test_mask_no_secrets(self, registry):
        assert registry.mask("no secrets here") == "no secrets here"

    def test_mask_empty_string(self, registry):
        registry.register("k", "v")
        assert registry.mask("") == ""

    def test_mask_none(self, registry):
        assert registry.mask(None) is None

    def test_empty_value_not_registered(self, registry):
        registry.register("key", "")
        assert registry.mask("anything") == "anything"
        assert len(registry._sorted) == 0

    def test_longest_first_replacement(self, registry):
        registry.register("short", "abc")
        registry.register("long", "abcdef")
        result = registry.mask("value: abcdef")
        assert result == "value: ***"
        assert "abc" not in result or result == "value: ***"

    def test_clear(self, registry):
        registry.register("key", "val")
        registry.clear()
        assert registry.mask("val") == "val"


# -- MaskingFilter --


class TestMaskingFilter:
    def test_masks_string_msg(self, registry):
        from tsugite.secrets.masking import SecretMaskingFilter

        registry.register("k", "secret_val")
        f = SecretMaskingFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "got secret_val here", (), None)
        # Patch the global registry
        import tsugite.secrets.masking as masking_mod

        orig = masking_mod.get_registry
        masking_mod.get_registry = lambda: registry
        try:
            f.filter(record)
            assert "secret_val" not in record.msg
            assert "***" in record.msg
        finally:
            masking_mod.get_registry = orig

    def test_masks_tuple_args(self, registry):
        from tsugite.secrets.masking import SecretMaskingFilter

        registry.register("k", "mysecret")
        f = SecretMaskingFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "%s %s", ("mysecret", 42), None)
        import tsugite.secrets.masking as masking_mod

        orig = masking_mod.get_registry
        masking_mod.get_registry = lambda: registry
        try:
            f.filter(record)
            assert record.args[0] == "***"
            assert record.args[1] == 42
        finally:
            masking_mod.get_registry = orig

    def test_masks_dict_args(self, registry):
        from tsugite.secrets.masking import SecretMaskingFilter

        registry.register("k", "mysecret")
        f = SecretMaskingFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "%(val)s", {"val": "mysecret", "num": 5}, None)
        import tsugite.secrets.masking as masking_mod

        orig = masking_mod.get_registry
        masking_mod.get_registry = lambda: registry
        try:
            f.filter(record)
            assert record.args["val"] == "***"
            assert record.args["num"] == 5
        finally:
            masking_mod.get_registry = orig

    def test_install_idempotent(self):
        import tsugite.secrets.masking as masking_mod

        masking_mod._installed = False
        root = logging.getLogger()
        before = len(root.filters)
        masking_mod.install_masking_filter()
        masking_mod.install_masking_filter()
        after = len(root.filters)
        assert after - before == 1
        masking_mod._installed = False


# -- get_secret tool --


class TestGetSecretTool:
    def test_get_secret_returns_value(self, monkeypatch, tmp_path):
        from tsugite.secrets import set_backend
        from tsugite.secrets.file import FileSecretBackend
        from tsugite.secrets.registry import get_registry

        b = FileSecretBackend({"path": str(tmp_path / "secrets")})
        b.set("test-token", "my-value")
        set_backend(b)

        from tsugite.agent_runner.helpers import set_allowed_secrets, set_current_agent

        set_current_agent("test-agent")
        set_allowed_secrets(None)

        from tsugite.tools.secrets import get_secret

        result = get_secret("test-token")
        assert result == "my-value"

        registry = get_registry()
        assert registry.mask("my-value") == "***"
        registry.clear()

    def test_get_secret_missing_raises(self, monkeypatch, tmp_path):
        from tsugite.secrets import set_backend
        from tsugite.secrets.file import FileSecretBackend

        set_backend(FileSecretBackend({"path": str(tmp_path / "empty")}))

        from tsugite.agent_runner.helpers import set_allowed_secrets, set_current_agent

        set_current_agent("test-agent")
        set_allowed_secrets(None)

        from tsugite.tools.secrets import get_secret

        with pytest.raises(RuntimeError, match="not found"):
            get_secret("nonexistent")

    def test_get_secret_allowlist_blocks(self, monkeypatch, tmp_path):
        from tsugite.secrets import set_backend
        from tsugite.secrets.file import FileSecretBackend

        b = FileSecretBackend({"path": str(tmp_path / "secrets")})
        b.set("allowed-key", "val")
        b.set("blocked-key", "val")
        set_backend(b)

        from tsugite.agent_runner.helpers import set_allowed_secrets, set_current_agent

        set_current_agent("test-agent")
        set_allowed_secrets(["allowed-key"])

        from tsugite.tools.secrets import get_secret

        assert get_secret("allowed-key") == "val"
        with pytest.raises(PermissionError, match="not allowed"):
            get_secret("blocked-key")


# -- SecretsConfig --


class TestSecretsConfig:
    def test_extra_fields_preserved(self):
        from tsugite.config import SecretsConfig

        sc = SecretsConfig(provider="exec", command=["pass", "show", "{{ name }}"])
        d = sc.model_dump()
        assert d["provider"] == "exec"
        assert d["command"] == ["pass", "show", "{{ name }}"]

    def test_create_backend_defaults_to_env(self, monkeypatch):
        monkeypatch.delenv("TSUGITE_SECRETS_BACKEND", raising=False)
        from tsugite.secrets import _create_backend

        b = _create_backend({"provider": "env"})
        assert isinstance(b, EnvSecretBackend)

    def test_create_backend_env_var_override(self, monkeypatch):
        monkeypatch.setenv("TSUGITE_SECRETS_BACKEND", "env")
        from tsugite.secrets import _create_backend

        b = _create_backend({"provider": "file", "path": "/tmp/x"})
        assert isinstance(b, EnvSecretBackend)


# -- CLI --


class TestSecretsCLI:
    def test_list(self, tmp_path):
        from tsugite.secrets import set_backend

        b = FileSecretBackend({"path": str(tmp_path / "secrets")})
        b.set("alpha", "v")
        b.set("beta", "v")
        set_backend(b)

        from tsugite.cli import app

        runner = typer.testing.CliRunner()
        result = runner.invoke(app, ["secrets", "list"])
        assert result.exit_code == 0
        assert "alpha" in result.output
        assert "beta" in result.output

    def test_set_and_delete(self, tmp_path):
        from tsugite.secrets import set_backend

        b = FileSecretBackend({"path": str(tmp_path / "secrets")})
        set_backend(b)

        from tsugite.cli import app

        runner = typer.testing.CliRunner()
        result = runner.invoke(app, ["secrets", "set", "mykey", "--value", "myval"])
        assert result.exit_code == 0
        assert b.get("mykey") == "myval"

        result = runner.invoke(app, ["secrets", "delete", "mykey"])
        assert result.exit_code == 0
        assert b.get("mykey") is None
