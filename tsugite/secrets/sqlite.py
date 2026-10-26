"""SQLite secret backend with Fernet encryption."""

import base64
import getpass
import os
import secrets
import shutil
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

KEYCHAIN_SERVICE = "tsugite"
# `security find-generic-password` exit status for errSecItemNotFound.
MACOS_ITEM_NOT_FOUND = 44


class SqliteSecretBackend:
    """Stores secrets in SQLite, encrypted with Fernet. A passphrase is always required."""

    def __init__(self, config: dict):
        from tsugite.config import get_xdg_data_path

        path = config.get("path") or str(get_xdg_data_path("secrets") / "secrets.db")
        self._db_path = Path(path)
        self._conn: sqlite3.Connection | None = None
        self._key_cache: dict[bytes, bytes] = {}
        self._keychain_account = (config.get("keychain_account") or "secrets") if config.get("keychain") else None
        if self._keychain_account and any(c in self._keychain_account for c in '"\\\r\n'):
            raise ValueError("keychain_account must not contain quotes, backslashes or line breaks")
        self._keychain_lock = threading.Lock()
        self._passphrase_value = self._resolve_passphrase(config)
        if not self._passphrase_value and not self._keychain_account:
            raise ValueError(
                "sqlite secrets backend requires a passphrase. Set TSUGITE_SECRETS_KEY env var, "
                "configure key_file or keychain in secrets config, or use an interactive terminal."
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

    def _resolve_passphrase(self, config: dict) -> str | None:
        env_key = os.environ.get("TSUGITE_SECRETS_KEY")
        if env_key:
            return env_key
        key_file = config.get("key_file")
        if key_file:
            kf = Path(key_file).expanduser()
            if kf.is_file():
                return kf.read_text().strip()
        if config.get("keychain"):
            return None
        if config.get("passphrase"):
            return config["passphrase"]
        if sys.stdin.isatty():
            passphrase = getpass.getpass("Secrets passphrase: ")
            return passphrase or None
        return None

    @property
    def _passphrase(self) -> str:
        with self._keychain_lock:
            if self._passphrase_value is None:
                self._passphrase_value = self._keychain_passphrase()
            return self._passphrase_value

    def _keychain_passphrase(self) -> str:
        account = self._keychain_account
        key = _keychain_lookup(account)
        if key:
            return key
        if self._get_conn().execute("SELECT 1 FROM secrets LIMIT 1").fetchone():
            raise ValueError(
                f"The OS keychain has no secrets key (service '{KEYCHAIN_SERVICE}', account '{account}'), "
                f"but {self._db_path} already holds encrypted secrets. Restore the key to the keychain "
                "or set TSUGITE_SECRETS_KEY."
            )
        key = secrets.token_urlsafe(32)
        _keychain_store(account, key)
        if _keychain_lookup(account) != key:
            raise ValueError(
                f"Stored the secrets key in the OS keychain (account '{account}') but could not read it back"
            )
        return key

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


def _keychain_tool() -> str:
    if sys.platform == "darwin":
        return "/usr/bin/security"
    tool = shutil.which("secret-tool")
    if not tool:
        raise ValueError(
            "secrets keychain is enabled but secret-tool was not found. Install libsecret-tools "
            "and run a Secret Service (gnome-keyring, KeePassXC or KWallet)."
        )
    return tool


def _keychain_lookup(account: str) -> str | None:
    tool = _keychain_tool()
    if sys.platform == "darwin":
        cmd = [tool, "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", account, "-w"]
    else:
        cmd = [tool, "lookup", "service", KEYCHAIN_SERVICE, "account", account]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    # secret-tool exits 1 with empty stderr for a missing item.
    missing = result.returncode == MACOS_ITEM_NOT_FOUND if sys.platform == "darwin" else not result.stderr.strip()
    if missing:
        return None
    raise ValueError(f"Could not read the secrets key from the OS keychain: {result.stderr.strip()}")


def _keychain_store(account: str, key: str) -> None:
    tool = _keychain_tool()
    if sys.platform == "darwin":
        # argv is visible to other local users through `ps`.
        cmd = [tool, "-i"]
        stdin = f'add-generic-password -s "{KEYCHAIN_SERVICE}" -a "{account}" -w "{key}"\n'
    else:
        cmd = [tool, "store", "--label=Tsugite secrets key", "service", KEYCHAIN_SERVICE, "account", account]
        stdin = key
    result = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"Could not store the secrets key in the OS keychain: {result.stderr.strip()}")
