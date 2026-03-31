"""Token management for daemon HTTP API authentication.

All tokens are hashed (SHA-256) and stored in a single dict. Tokens differ by:
- persistent: saved to tokens.json (admin tokens for humans/web UI)
- expires_at: TTL-based expiry (agent tokens per scheduled task)
"""

import hashlib
import json
import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
    """Manages all API tokens (admin and agent) in a single store."""

    def __init__(self, path: Path, default_ttl_seconds: int = 3600):
        self._path = path
        self._default_ttl = default_ttl_seconds
        self._tokens: dict[str, Token] = {}  # hash -> Token
        self._load()

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    # --- Persistence ---

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            for entry in json.loads(self._path.read_text(encoding="utf-8")):
                t = Token(**entry)
                self._tokens[t.hash] = t
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning("Failed to load tokens from %s: %s", self._path, e)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        persistent = [
            {"hash": t.hash, "identity": t.identity, "created_at": t.created_at, "prefix": t.prefix, "persistent": True}
            for t in self._tokens.values()
            if t.persistent
        ]
        tmp = self._path.with_suffix(".tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(persistent, f, indent=2)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        os.replace(str(tmp), str(self._path))

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
        self._tokens[t.hash] = t
        self._save()
        return t, raw

    def list_admin_tokens(self) -> list[Token]:
        return [t for t in self._tokens.values() if t.persistent]

    def revoke_admin_token(self, name_or_prefix: str) -> bool:
        """Revoke an admin token by name or prefix. Returns True if found."""
        target = f"admin:{name_or_prefix}"
        for h, t in self._tokens.items():
            if t.persistent and (t.identity == target or t.prefix == name_or_prefix):
                del self._tokens[h]
                self._save()
                return True
        return False

    def has_admin_tokens(self) -> bool:
        return any(t.persistent for t in self._tokens.values())

    # --- Agent tokens (temporary, with TTL) ---

    def issue(self, agent: str, schedule_id: str = "", ttl: int | None = None) -> str:
        """Issue a temporary token for an agent/schedule. Returns the raw token."""
        if sum(1 for t in self._tokens.values() if not t.persistent) > 100:
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
        """Revoke a token by raw value."""
        self._tokens.pop(self._hash(token), None)

    def cleanup_expired(self) -> int:
        """Remove expired tokens. Returns count removed."""
        now = datetime.now(timezone.utc).isoformat()
        before = len(self._tokens)
        self._tokens = {h: t for h, t in self._tokens.items() if t.expires_at is None or t.expires_at > now}
        return before - len(self._tokens)

    # --- Validation ---

    def validate(self, token: str) -> tuple[bool, str]:
        """Validate a token. Returns (valid, identity)."""
        h = self._hash(token)
        t = self._tokens.get(h)
        if not t:
            return False, ""
        if t.expires_at and t.expires_at < datetime.now(timezone.utc).isoformat():
            del self._tokens[h]
            return False, ""
        return True, t.identity
