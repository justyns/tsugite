"""Tests for daemon auth token management."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from tsugite_daemon.auth import TokenStore


class TestAdminTokens:
    def test_create_admin_token(self, tmp_path):
        store = TokenStore(tmp_path / "tokens.json")
        t, raw = store.create_admin_token(name="test-token")

        assert raw.startswith("tsu_")
        assert t.identity == "admin:test-token"
        assert t.prefix == raw[:8]
        assert t.persistent is True

    def test_create_admin_token_persists(self, tmp_path):
        path = tmp_path / "tokens.json"
        store = TokenStore(path)
        t, raw = store.create_admin_token(name="persist-test")

        store2 = TokenStore(path)
        tokens = store2.list_admin_tokens()
        assert len(tokens) == 1
        assert tokens[0].identity == "admin:persist-test"
        assert tokens[0].hash == t.hash

    def test_validate_admin_token(self, tmp_path):
        store = TokenStore(tmp_path / "tokens.json")
        _t, raw = store.create_admin_token(name="valid-test")

        valid, identity = store.validate(raw)
        assert valid is True
        assert identity == "admin:valid-test"

    def test_validate_invalid_token(self, tmp_path):
        store = TokenStore(tmp_path / "tokens.json")
        store.create_admin_token(name="test")

        valid, identity = store.validate("tsu_bogus_token_value")
        assert valid is False
        assert identity == ""

    def test_list_admin_tokens(self, tmp_path):
        store = TokenStore(tmp_path / "tokens.json")
        store.create_admin_token(name="first")
        store.create_admin_token(name="second")

        tokens = store.list_admin_tokens()
        assert len(tokens) == 2
        identities = {t.identity for t in tokens}
        assert identities == {"admin:first", "admin:second"}

    def test_revoke_admin_token_by_name(self, tmp_path):
        store = TokenStore(tmp_path / "tokens.json")
        _t, raw = store.create_admin_token(name="to-revoke")

        assert store.revoke_admin_token("to-revoke") is True
        valid, _ = store.validate(raw)
        assert valid is False
        assert len(store.list_admin_tokens()) == 0

    def test_revoke_admin_token_by_prefix(self, tmp_path):
        store = TokenStore(tmp_path / "tokens.json")
        t, raw = store.create_admin_token(name="prefix-test")

        assert store.revoke_admin_token(t.prefix) is True
        valid, _ = store.validate(raw)
        assert valid is False

    def test_revoke_nonexistent(self, tmp_path):
        store = TokenStore(tmp_path / "tokens.json")
        assert store.revoke_admin_token("nonexistent") is False

    def test_has_admin_tokens(self, tmp_path):
        store = TokenStore(tmp_path / "tokens.json")
        assert store.has_admin_tokens() is False

        store.create_admin_token(name="test")
        assert store.has_admin_tokens() is True

    def test_daemon_db_permissions(self, tmp_path):
        """daemon.db now holds token hashes (and session content) - it must be
        owner-only like tokens.json was."""
        path = tmp_path / "tokens.json"
        store = TokenStore(path)
        store.create_admin_token(name="perm-test")

        mode = (tmp_path / "daemon.db").stat().st_mode & 0o777
        assert mode == 0o600

    def test_corrupt_file_handled(self, tmp_path):
        path = tmp_path / "tokens.json"
        path.write_text("not valid json")

        store = TokenStore(path)
        assert store.has_admin_tokens() is False

    def test_unnamed_token(self, tmp_path):
        store = TokenStore(tmp_path / "tokens.json")
        t, raw = store.create_admin_token()

        assert t.identity == "admin:"
        valid, identity = store.validate(raw)
        assert valid is True
        assert identity == "admin:"


class TestTempTokens:
    def test_issue_and_validate(self, tmp_path):
        store = TokenStore(tmp_path / "tokens.json")
        raw = store.issue(agent="my-agent", schedule_id="daily-report")

        assert raw.startswith("tsu_")
        valid, identity = store.validate(raw)
        assert valid is True
        assert identity == "my-agent:daily-report"

    def test_revoke_temp_token(self, tmp_path):
        store = TokenStore(tmp_path / "tokens.json")
        raw = store.issue(agent="agent", schedule_id="sched")

        store.revoke(raw)
        valid, _ = store.validate(raw)
        assert valid is False

    def test_temp_token_expiry(self, tmp_path):
        store = TokenStore(tmp_path / "tokens.json")
        raw = store.issue(agent="agent", schedule_id="sched", ttl=1)

        valid, _ = store.validate(raw)
        assert valid is True

        expired_time = datetime.now(timezone.utc) + timedelta(seconds=2)
        with patch("tsugite_daemon.auth.datetime") as mock_dt:
            mock_dt.now.return_value = expired_time
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            valid, _ = store.validate(raw)
            assert valid is False

    def test_cleanup_expired(self, tmp_path):
        store = TokenStore(tmp_path / "tokens.json")
        store.issue(agent="a", schedule_id="s", ttl=1)
        store.issue(agent="b", schedule_id="s", ttl=1)
        store.issue(agent="c", schedule_id="s", ttl=3600)

        expired_time = datetime.now(timezone.utc) + timedelta(seconds=2)
        with patch("tsugite_daemon.auth.datetime") as mock_dt:
            mock_dt.now.return_value = expired_time
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            removed = store.cleanup_expired()
            assert removed == 2

    def test_temp_tokens_not_persisted(self, tmp_path):
        path = tmp_path / "tokens.json"
        store = TokenStore(path)
        store.issue(agent="agent", schedule_id="sched")

        store2 = TokenStore(path)
        assert not any(not t.persistent for t in store2._tokens.values())

    def test_issue_default_ttl(self, tmp_path):
        store = TokenStore(tmp_path / "tokens.json", default_ttl_seconds=600)
        raw = store.issue(agent="a", schedule_id="s")

        h = store._hash(raw)
        t = store._tokens[h]
        created = datetime.fromisoformat(t.created_at)
        expires = datetime.fromisoformat(t.expires_at)
        assert (expires - created).total_seconds() == 600


class TestCrossProcessVisibility:
    """Tokens created/revoked by one process (the CLI) must take effect in
    another (the daemon) without restart - persistent tokens are queried from
    daemon.db per validation, no reload machinery."""

    def test_new_token_visible_without_restart(self, tmp_path):
        path = tmp_path / "tokens.json"
        daemon = TokenStore(path)
        _t, raw = daemon.create_admin_token(name="existing")
        assert daemon.validate(raw)[0] is True

        # Simulate `tsugite daemon token create` in another process
        cli = TokenStore(path)
        _t2, new_raw = cli.create_admin_token(name="fresh")

        # Daemon's running TokenStore picks up the new token on next validate
        valid, identity = daemon.validate(new_raw)
        assert valid is True
        assert identity == "admin:fresh"

    def test_revoked_token_rejected_without_restart(self, tmp_path):
        path = tmp_path / "tokens.json"
        daemon = TokenStore(path)
        _t, raw = daemon.create_admin_token(name="doomed")
        assert daemon.validate(raw)[0] is True

        cli = TokenStore(path)
        cli.revoke_admin_token("doomed")

        valid, _ = daemon.validate(raw)
        assert valid is False

    def test_reload_preserves_in_memory_temp_tokens(self, tmp_path):
        path = tmp_path / "tokens.json"
        daemon = TokenStore(path)
        temp_raw = daemon.issue(agent="agent", schedule_id="sched")

        cli = TokenStore(path)
        cli.create_admin_token(name="new-admin")

        daemon.validate("unknown-token")
        valid, identity = daemon.validate(temp_raw)
        assert valid is True
        assert identity == "agent:sched"


class TestMixedValidation:
    def test_admin_and_temp_coexist(self, tmp_path):
        store = TokenStore(tmp_path / "tokens.json")
        _t, admin_raw = store.create_admin_token(name="admin")
        temp_raw = store.issue(agent="agent", schedule_id="sched")

        valid, identity = store.validate(admin_raw)
        assert valid is True
        assert identity == "admin:admin"

        valid, identity = store.validate(temp_raw)
        assert valid is True
        assert identity == "agent:sched"

    def test_revoked_admin_doesnt_affect_temp(self, tmp_path):
        store = TokenStore(tmp_path / "tokens.json")
        _t, admin_raw = store.create_admin_token(name="admin")
        temp_raw = store.issue(agent="agent", schedule_id="sched")

        store.revoke_admin_token("admin")

        valid, _ = store.validate(admin_raw)
        assert valid is False

        valid, identity = store.validate(temp_raw)
        assert valid is True
        assert identity == "agent:sched"


class TestSqliteTokenStorage:
    def test_no_tokens_json_writes(self, tmp_path):
        path = tmp_path / "tokens.json"
        store = TokenStore(path)
        store.create_admin_token(name="x")
        assert not path.exists(), "legacy tokens.json must not be written"
        assert (tmp_path / "daemon.db").exists()

    def test_legacy_tokens_json_migrated_and_left_untouched(self, tmp_path):
        import hashlib
        import json

        path = tmp_path / "tokens.json"
        h = hashlib.sha256(b"tsu_legacyraw").hexdigest()
        path.write_text(
            json.dumps(
                [
                    {
                        "hash": h,
                        "identity": "admin:old",
                        "created_at": "2026-01-01",
                        "prefix": "tsu_lega",
                        "persistent": True,
                    }
                ]
            )
        )
        original = path.read_text()

        store = TokenStore(path)
        valid, identity = store.validate("tsu_legacyraw")
        assert valid is True and identity == "admin:old"

        store.revoke_admin_token("old")
        assert path.read_text() == original, "legacy file stays as an untouched backup"
        assert TokenStore(path).validate("tsu_legacyraw")[0] is False, "the db, not the stale JSON, is the authority"


class TestCliTokensPathResolution:
    def test_resolves_state_dir_from_daemon_config(self, tmp_path):
        """`tsu daemon token create` must write where the DAEMON reads: a custom
        state_dir in daemon.yaml previously left the CLI writing to the XDG
        default, and the daemon could never see created tokens."""
        cfg = tmp_path / "daemon.yaml"
        cfg.write_text(f"state_dir: {tmp_path / 'custom-state'}\nagents: {{}}\n")

        from tsugite.cli.daemon import _get_tokens_path

        assert _get_tokens_path(cfg) == tmp_path / "custom-state" / "tokens.json"

    def test_falls_back_to_default_when_config_missing(self, tmp_path, monkeypatch):
        """A fresh box (no daemon.yaml yet) must still be able to create tokens
        in the default state dir instead of crashing."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

        from tsugite.cli.daemon import _get_tokens_path

        assert (
            _get_tokens_path(tmp_path / "nonexistent.yaml") == tmp_path / "data" / "tsugite" / "daemon" / "tokens.json"
        )
