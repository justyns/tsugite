"""SQLite secret backend with Fernet encryption."""

import base64
import getpass
import os
import sqlite3
import sys
from pathlib import Path


class SqliteSecretBackend:
    """Stores secrets in SQLite, encrypted with Fernet. A passphrase is always required."""

    def __init__(self, config: dict):
        from tsugite.config import get_xdg_data_path

        path = config.get("path") or str(get_xdg_data_path("secrets") / "secrets.db")
        self._db_path = Path(path)
        self._conn: sqlite3.Connection | None = None
        self._key_cache: dict[bytes, bytes] = {}
        self._passphrase = self._resolve_passphrase(
            config_passphrase=config.get("passphrase"),
            key_file=config.get("key_file"),
        )
        if not self._passphrase:
            raise ValueError(
                "sqlite secrets backend requires a passphrase. Set TSUGITE_SECRETS_KEY env var, "
                "configure key_file in secrets config, or use an interactive terminal."
            )

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            new_db = not self._db_path.exists()
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            if new_db:
                self._db_path.chmod(0o600)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("CREATE TABLE IF NOT EXISTS secrets (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self._conn.commit()
        return self._conn

    @staticmethod
    def _resolve_passphrase(
        config_passphrase: str | None = None,
        key_file: str | None = None,
    ) -> str | None:
        env_key = os.environ.get("TSUGITE_SECRETS_KEY")
        if env_key:
            return env_key
        if key_file:
            kf = Path(key_file).expanduser()
            if kf.is_file():
                return kf.read_text().strip()
        if config_passphrase:
            return config_passphrase
        if sys.stdin.isatty():
            passphrase = getpass.getpass("Secrets passphrase: ")
            return passphrase or None
        return None

    def _derive_key(self, salt: bytes) -> bytes:
        cached = self._key_cache.get(salt)
        if cached:
            return cached
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=480_000)
        key = base64.urlsafe_b64encode(kdf.derive(self._passphrase.encode()))
        self._key_cache[salt] = key
        return key

    def _encrypt(self, value: str) -> str:
        from cryptography.fernet import Fernet

        salt = os.urandom(16)
        key = self._derive_key(salt)
        token = Fernet(key).encrypt(value.encode())
        return base64.b64encode(salt + token).decode()

    def _decrypt(self, blob: str) -> str:
        from cryptography.fernet import Fernet, InvalidToken

        try:
            raw = base64.b64decode(blob)
        except Exception as e:
            raise ValueError(f"Corrupted secret value: {e}") from e
        if len(raw) < 17:
            raise ValueError("Corrupted secret value: too short")
        salt, token = raw[:16], raw[16:]
        key = self._derive_key(salt)
        try:
            return Fernet(key).decrypt(token).decode()
        except InvalidToken:
            raise ValueError("Wrong passphrase or corrupted secret")

    def get(self, name: str) -> str | None:
        row = self._get_conn().execute("SELECT value FROM secrets WHERE name=?", (name,)).fetchone()
        if row is None:
            return None
        return self._decrypt(row[0])

    def set(self, name: str, value: str) -> None:
        encrypted = self._encrypt(value)
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO secrets (name, value) VALUES (?, ?)",
            (name, encrypted),
        )
        conn.commit()

    def delete(self, name: str) -> bool:
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM secrets WHERE name=?", (name,))
        conn.commit()
        return cursor.rowcount > 0

    def list_names(self) -> list[str]:
        rows = self._get_conn().execute("SELECT name FROM secrets ORDER BY name").fetchall()
        return [r[0] for r in rows]
