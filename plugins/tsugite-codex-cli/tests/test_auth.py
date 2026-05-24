"""Tests for CodexAuthStore (cases A1-A6 from the plan)."""

from __future__ import annotations

import asyncio
import base64
import json
import multiprocessing
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from tsugite_codex_cli.auth import CodexAuthError, CodexAuthStore


def _fake_jwt(exp_seconds_from_now: int) -> str:
    """Build a JWT-shaped string (header.payload.sig) with a controllable exp claim."""

    def b64u(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    header = b64u(b'{"alg":"none","typ":"JWT"}')
    payload = b64u(json.dumps({"exp": int(time.time()) + exp_seconds_from_now}).encode())
    sig = b64u(b"sig")
    return f"{header}.{payload}.{sig}"


def _write_auth(tmp_path: Path, *, auth_mode: str = "chatgpt", access_exp: int = 3600, extras: dict | None = None) -> Path:
    """Write a minimal auth.json into tmp_path and return its path."""
    payload = {
        "OPENAI_API_KEY": None,
        "auth_mode": auth_mode,
        "tokens": {
            "access_token": _fake_jwt(access_exp),
            "id_token": "id-tok",
            "refresh_token": "rt-1",
            "account_id": "acct-42",
        },
        "last_refresh": "2026-05-22T00:00:00+00:00",
    }
    if extras:
        payload.update(extras)
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps(payload))
    return auth_path


@pytest.fixture
def codex_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    return tmp_path


# ── A1: missing auth.json ──
@pytest.mark.asyncio
async def test_missing_auth_file_raises_actionable_error(codex_home):
    store = CodexAuthStore()
    with pytest.raises(CodexAuthError) as exc:
        await store.get_access_token()
    assert "codex login" in str(exc.value).lower()


# ── A2: wrong auth_mode ──
@pytest.mark.asyncio
async def test_apikey_mode_refused(codex_home):
    _write_auth(codex_home, auth_mode="apikey")
    store = CodexAuthStore()
    with pytest.raises(CodexAuthError) as exc:
        await store.get_access_token()
    msg = str(exc.value).lower()
    assert "chatgpt" in msg or "codex login" in msg


# ── A3: valid JWT, no refresh ──
@pytest.mark.asyncio
async def test_valid_token_no_refresh(codex_home):
    _write_auth(codex_home, access_exp=3600)
    store = CodexAuthStore()
    with patch("httpx.post") as mock_post:
        tok, account = await store.get_access_token()
    assert tok.count(".") == 2  # JWT shape
    assert account == "acct-42"
    mock_post.assert_not_called()


# ── A4: near-expiry refresh + atomic writeback + unknown fields preserved ──
@pytest.mark.asyncio
async def test_near_expiry_refreshes_atomically_and_preserves_unknown_fields(codex_home):
    _write_auth(
        codex_home,
        access_exp=10,  # within REFRESH_SKEW_SECONDS=30
        extras={"some_future_field": {"nested": [1, 2, 3]}},
    )

    new_access = _fake_jwt(3600)
    new_refresh = "rt-2"

    class FakeResp:
        status_code = 200

        def json(self):
            return {"access_token": new_access, "refresh_token": new_refresh, "id_token": "id-new"}

        @property
        def text(self):
            return ""

    real_replace = os.replace
    real_chmod = os.chmod
    seen_modes: list[int] = []
    seen_replaces: list[tuple[str, str]] = []

    def spy_replace(src, dst):
        seen_replaces.append((str(src), str(dst)))
        return real_replace(src, dst)

    def spy_chmod(path, mode):
        seen_modes.append(mode)
        return real_chmod(path, mode)

    with patch("httpx.post", return_value=FakeResp()) as mock_post, \
         patch("tsugite_codex_cli.auth.os.replace", side_effect=spy_replace), \
         patch("tsugite_codex_cli.auth.os.chmod", side_effect=spy_chmod):
        store = CodexAuthStore()
        tok, account = await store.get_access_token()

    assert tok == new_access
    assert account == "acct-42"
    mock_post.assert_called_once()

    # Atomic write: ".tmp" file replaced into final path
    assert seen_replaces, "os.replace was not called"
    src, dst = seen_replaces[-1]
    assert src.endswith(".tmp")
    assert dst.endswith("auth.json")

    # Mode tightened to 0o600 on the new file
    assert 0o600 in seen_modes

    # Unknown sibling preserved
    on_disk = json.loads((codex_home / "auth.json").read_text())
    assert on_disk["some_future_field"] == {"nested": [1, 2, 3]}
    assert on_disk["tokens"]["refresh_token"] == new_refresh


# ── A5: refresh failure ──
@pytest.mark.asyncio
@pytest.mark.parametrize("err", ["invalid_grant", "refresh_token_expired", "refresh_token_reused"])
async def test_refresh_failure_emits_re_login_error(codex_home, err):
    _write_auth(codex_home, access_exp=10)

    class FakeResp:
        status_code = 400

        def json(self):
            return {"error": err, "error_description": "go away"}

        @property
        def text(self):
            return json.dumps(self.json())

    with patch("httpx.post", return_value=FakeResp()):
        store = CodexAuthStore()
        with pytest.raises(CodexAuthError) as exc:
            await store.get_access_token()

    msg = str(exc.value).lower()
    assert "codex login" in msg


# ── A6: concurrent refresh, sidecar lock, cross-process serialisation ──
def _hold_lock(lock_path: str, hold_seconds: float, ready_path: str, done_path: str) -> None:
    """Child process: grab portalocker.LOCK_EX on the sidecar, then sit on it."""
    import time as t

    import portalocker

    Path(ready_path).touch()  # signal parent we're about to lock
    with open(lock_path, "a+") as fh:
        portalocker.lock(fh, portalocker.LOCK_EX)
        Path(done_path).touch()  # signal we've actually acquired the lock
        t.sleep(hold_seconds)
        portalocker.unlock(fh)


@pytest.mark.asyncio
async def test_concurrent_refresh_serialises(codex_home):
    """Two coroutines + a second-process lock holder must not double-refresh or tear the file."""
    _write_auth(codex_home, access_exp=10)

    new_access = _fake_jwt(3600)

    class FakeResp:
        status_code = 200

        def json(self):
            return {"access_token": new_access, "refresh_token": "rt-new", "id_token": "id-new"}

        @property
        def text(self):
            return ""

    ready = codex_home / "child_ready"
    holding = codex_home / "child_holding"
    lock_path = codex_home / "auth.json.lock"

    proc = multiprocessing.Process(
        target=_hold_lock,
        args=(str(lock_path), 0.5, str(ready), str(holding)),
    )
    proc.start()
    try:
        # Wait until the child has actually acquired the lock
        for _ in range(100):
            if holding.exists():
                break
            await asyncio.sleep(0.02)
        else:
            pytest.fail("child process never reported lock acquisition")

        with patch("httpx.post", return_value=FakeResp()) as mock_post:
            store = CodexAuthStore()
            results = await asyncio.gather(store.get_access_token(), store.get_access_token())

        proc.join(timeout=5)
        assert proc.exitcode == 0
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2)

    assert mock_post.call_count == 1, "refresh should fire exactly once across both coroutines"
    assert all(tok == new_access for tok, _ in results)

    # File survived without truncation/corruption
    payload = json.loads((codex_home / "auth.json").read_text())
    assert payload["tokens"]["access_token"] == new_access
