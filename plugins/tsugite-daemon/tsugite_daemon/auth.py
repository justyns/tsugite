"""Token management for daemon HTTP API authentication.

All tokens are hashed (SHA-256). Persistent admin tokens live in daemon.db
(`auth_tokens` collection) and are queried per operation, so tokens created or
revoked by another process (the `tsu daemon token` CLI) take effect on the
running daemon's next validation - no reload machinery. Ephemeral agent tokens
(TTL-based, per scheduled task) are memory-only and die with the process.

The legacy ``tokens.json`` path remains the constructor argument as a one-time
migration source; the file is left untouched as a backup.
"""

import hashlib
import logging
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tsugite.core.record_store import SqliteCollectionStorage

logger = logging.getLogger(__name__)

TOKEN_PREFIX = "tsu_"
TOKENS_FILENAME = "tokens.json"


@dataclass
class Token:
    hash: str
    identity: str  # "admin:<name>" or "<agent>:<schedule_id>"
    created_at: str  # ISO format
    prefix: str  # first 8 chars for display
    persistent: bool = False
    expires_at: str | None = None  # ISO format, None = never


class TokenStore:
    """Manages all API tokens (admin and agent)."""

    def __init__(self, path: Path, default_ttl_seconds: int = 3600):
        self._path = path
        self._storage = SqliteCollectionStorage.for_state_file(path, "auth_tokens")
        self._default_ttl = default_ttl_seconds
        self._tokens: dict[str, Token] = {}  # ephemeral (agent) tokens only
        self._migrate_legacy()

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    # --- Persistence ---

    @staticmethod
    def _token_from_entry(entry: dict) -> Token | None:
        try:
            return Token(**entry)
        except TypeError as e:
            logger.warning("Skipping malformed token entry: %s", e)
            return None

    def _migrate_legacy(self) -> None:
        entries, migrating = self._storage.load_or_migrate(self._path, "auth_tokens")
        if not migrating:
            return
        imported = {t.hash: asdict(t) for e in entries if (t := self._token_from_entry(e)) and t.persistent}
        self._storage.replace_all(imported)

    def _persistent_token(self, token_hash: str) -> Token | None:
        entry = self._storage.get(token_hash)
        return self._token_from_entry(entry) if entry is not None else None

    # --- Admin tokens (persistent, no expiry) ---

    def create_admin_token(self, name: str = "") -> tuple[Token, str]:
        """Create a persistent admin token. Returns (token_meta, raw_token).

        The raw token is only available at creation time.
        """
        raw = TOKEN_PREFIX + secrets.token_urlsafe(32)
        t = Token(
            hash=self._hash(raw),
            identity=f"admin:{name}",
            created_at=datetime.now(timezone.utc).isoformat(),
            prefix=raw[:8],
            persistent=True,
        )
        self._storage.upsert(t.hash, asdict(t))
        return t, raw

    def list_admin_tokens(self) -> list[Token]:
        return [t for e in self._storage.load_all() if (t := self._token_from_entry(e))]

    def revoke_admin_token(self, name_or_prefix: str) -> bool:
        """Revoke an admin token by name or prefix. Returns True if found."""
        target = f"admin:{name_or_prefix}"
        for t in self.list_admin_tokens():
            if t.identity == target or t.prefix == name_or_prefix:
                self._storage.delete(t.hash)
                return True
        return False

    def has_admin_tokens(self) -> bool:
        return self._storage.exists_any()

    # --- Agent tokens (temporary, with TTL) ---

    def issue(self, agent: str, schedule_id: str = "", ttl: int | None = None) -> str:
        """Issue a temporary token for an agent/schedule. Returns the raw token."""
        if len(self._tokens) > 100:
            self.cleanup_expired()
        raw = TOKEN_PREFIX + secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        t = Token(
            hash=self._hash(raw),
            identity=f"{agent}:{schedule_id}",
            created_at=now.isoformat(),
            prefix=raw[:8],
            expires_at=(now + timedelta(seconds=ttl or self._default_ttl)).isoformat(),
        )
        self._tokens[t.hash] = t
        return raw

    def revoke(self, token: str) -> None:
        """Revoke a token by raw value (ephemeral or persistent)."""
        h = self._hash(token)
        self._tokens.pop(h, None)
        self._storage.delete(h)

    def cleanup_expired(self) -> int:
        """Remove expired ephemeral tokens. Returns count removed."""
        now = datetime.now(timezone.utc).isoformat()
        before = len(self._tokens)
        self._tokens = {h: t for h, t in self._tokens.items() if t.expires_at is None or t.expires_at > now}
        return before - len(self._tokens)

    # --- Validation ---

    def validate(self, token: str) -> tuple[bool, str]:
        """Validate a token. Returns (valid, identity)."""
        h = self._hash(token)
        t = self._tokens.get(h) or self._persistent_token(h)
        if not t:
            return False, ""
        if t.expires_at and t.expires_at < datetime.now(timezone.utc).isoformat():
            self._tokens.pop(h, None)
            return False, ""
        return True, t.identity
