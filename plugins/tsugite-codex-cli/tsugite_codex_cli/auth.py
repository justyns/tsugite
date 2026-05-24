"""Codex CLI OAuth token store: read, refresh, atomic writeback of ~/.codex/auth.json."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from pathlib import Path

import httpx
import portalocker

# Hardcoded OAuth client id used by the upstream Codex CLI.
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
TOKEN_ENDPOINT = "https://auth.openai.com/oauth/token"
REFRESH_SKEW_SECONDS = 30
REFRESH_TIMEOUT_SECONDS = 30

# Refresh-token failures that require the user to re-run `codex login`.
_RE_LOGIN_ERRORS = {
    "invalid_grant",
    "refresh_token_expired",
    "refresh_token_reused",
    "refresh_token_invalidated",
}


class CodexAuthError(Exception):
    """Raised when the Codex auth store is missing, in apikey mode, or refresh fails."""


def _resolve_home(home: Path | None) -> Path:
    if home is not None:
        return home
    env = os.environ.get("CODEX_HOME")
    if env:
        return Path(env)
    return Path.home() / ".codex"


def _decode_jwt_exp(token: str) -> int | None:
    """Return the `exp` claim from a JWT, or None if it can't be decoded."""
    try:
        payload_b64 = token.split(".")[1]
        padding = "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + padding))
        exp = payload.get("exp")
        return int(exp) if exp is not None else None
    except (IndexError, ValueError, json.JSONDecodeError):
        return None


class CodexAuthStore:
    """Token store backed by ~/.codex/auth.json (override via $CODEX_HOME)."""

    def __init__(self, home: Path | None = None):
        self._home = _resolve_home(home)
        self._auth_path = self._home / "auth.json"
        self._lock_path = self._home / "auth.json.lock"
        self._async_lock = asyncio.Lock()

    async def get_access_token(self) -> tuple[str, str]:
        """Return (access_token, account_id), refreshing if near expiry."""
        async with self._async_lock:
            return await asyncio.to_thread(self._sync_get_or_refresh)

    def _sync_get_or_refresh(self) -> tuple[str, str]:
        if not self._auth_path.exists():
            raise CodexAuthError(
                f"No Codex auth file at {self._auth_path}. Run `codex login` first."
            )

        # Lock the sidecar, not auth.json itself: os.replace unlinks the inode and
        # any waiter blocked on the original file descriptor would never wake.
        self._home.mkdir(parents=True, exist_ok=True)
        with open(self._lock_path, "a+") as lock_fh:
            portalocker.lock(lock_fh, portalocker.LOCK_EX)
            try:
                os.chmod(self._lock_path, 0o600)
            except OSError:
                pass
            try:
                return self._read_check_refresh()
            finally:
                portalocker.unlock(lock_fh)

    def _read_check_refresh(self) -> tuple[str, str]:
        data = self._load_json()
        auth_mode = data.get("auth_mode")
        if auth_mode != "chatgpt":
            raise CodexAuthError(
                f"Codex auth mode is {auth_mode!r}; this provider requires ChatGPT subscription auth. "
                "Run `codex login` and choose the ChatGPT sign-in."
            )

        tokens = data.get("tokens") or {}
        access_token = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token")
        account_id = tokens.get("account_id")
        if not access_token or not refresh_token or not account_id:
            raise CodexAuthError(
                f"Codex auth file at {self._auth_path} is missing tokens. Run `codex login`."
            )

        exp = _decode_jwt_exp(access_token)
        now = int(time.time())
        if exp is None or exp - now <= REFRESH_SKEW_SECONDS:
            new_tokens = self._refresh(refresh_token)
            # Merge so we keep account_id and any unknown sibling fields.
            data["tokens"] = {**tokens, **new_tokens}
            data["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
            self._write_atomic(data)
            access_token = new_tokens.get("access_token", access_token)

        return access_token, account_id

    def _load_json(self) -> dict:
        try:
            return json.loads(self._auth_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise CodexAuthError(f"Could not read {self._auth_path}: {exc}. Run `codex login`.") from exc

    def _refresh(self, refresh_token: str) -> dict:
        try:
            resp = httpx.post(
                TOKEN_ENDPOINT,
                data={
                    "client_id": CLIENT_ID,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
                timeout=REFRESH_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError as exc:
            raise CodexAuthError(f"OAuth refresh request failed: {exc}") from exc

        if resp.status_code >= 400:
            body: dict = {}
            try:
                body = resp.json() if resp.text else {}
            except ValueError:
                pass
            err = body.get("error", "")
            if err in _RE_LOGIN_ERRORS:
                raise CodexAuthError(
                    f"Codex refresh token rejected ({err}). Run `codex login` to sign in again."
                )
            raise CodexAuthError(
                f"OAuth refresh returned HTTP {resp.status_code}: {body or resp.text}"
            )

        try:
            body = resp.json()
        except ValueError as exc:
            raise CodexAuthError(f"OAuth refresh returned non-JSON body: {resp.text}") from exc

        new_tokens: dict = {}
        for key in ("access_token", "refresh_token", "id_token"):
            if body.get(key):
                new_tokens[key] = body[key]
        if "access_token" not in new_tokens:
            raise CodexAuthError(f"OAuth refresh response missing access_token: {body}")
        return new_tokens

    def _write_atomic(self, data: dict) -> None:
        tmp_path = self._auth_path.with_suffix(self._auth_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(data, indent=2))
        try:
            os.chmod(tmp_path, 0o600)
        except OSError:
            pass
        os.replace(tmp_path, self._auth_path)
