"""HS256 sign/verify contract for the ONLYOFFICE Docs handshake."""

import base64
import json

import pytest
from tsugite_onlyoffice import jwt

SECRET = "example-shared-secret"


def _seg(data: dict) -> str:
    """Encode a header/claims segment the way a hostile caller would."""
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip("=")


def test_round_trip():
    claims = jwt.verify(jwt.sign({"payload": {"key": "abc123"}}, SECRET), SECRET)
    assert claims.get("payload") == {"key": "abc123"}
    assert "exp" in claims


def test_tampered_payload_does_not_verify():
    """Re-base64 the body of a genuine token and keep its signature."""
    parts = jwt.sign({"role": "reader"}, SECRET, expires_in=0).split(".")
    assert len(parts) == 3
    header, _claims, signature = parts
    with pytest.raises(ValueError):
        jwt.verify(f"{header}.{_seg({'role': 'admin'})}.{signature}", SECRET)


def test_alg_none_rejected():
    header = _seg({"alg": "none", "typ": "JWT"})
    with pytest.raises(ValueError):
        jwt.verify(f"{header}.{_seg({'role': 'admin'})}.", SECRET)


def test_alg_rs256_rejected():
    header = _seg({"alg": "RS256", "typ": "JWT"})
    with pytest.raises(ValueError):
        jwt.verify(f"{header}.{_seg({'role': 'admin'})}.c2lnbmF0dXJl", SECRET)


def test_expired_token_rejected():
    with pytest.raises(ValueError):
        jwt.verify(jwt.sign({"role": "reader"}, SECRET, expires_in=-1), SECRET)


def test_wrong_secret_rejected():
    with pytest.raises(ValueError):
        jwt.verify(jwt.sign({"role": "reader"}, SECRET), "not-the-secret")


@pytest.mark.parametrize("token", ["", "a", "a.b", "a.b.c.d"])
def test_bad_segment_count_rejected(token):
    with pytest.raises(ValueError):
        jwt.verify(token, SECRET)


def test_non_base64_segment_rejected():
    with pytest.raises(ValueError):
        jwt.verify(f"!!!not-base64!!!.{_seg({'role': 'admin'})}.c2ln", SECRET)


@pytest.mark.parametrize("segment", [0, 1, 2])
def test_non_ascii_segment_rejected(segment):
    """Every rejection is a ValueError, or a caller catching one gets a 500 instead of a 401."""
    parts = jwt.sign({"role": "reader"}, SECRET).split(".")
    parts[segment] += "é"
    with pytest.raises(ValueError):
        jwt.verify(".".join(parts), SECRET)


@pytest.mark.parametrize("filler", ["", "a", "aa", "aaa", "aaaa"])
def test_base64_padding_variants_round_trip(filler):
    """Padding is stripped on the wire, so every claims length must decode back."""
    name = f"report{filler}.docx"
    assert jwt.verify(jwt.sign({"doc": name}, SECRET), SECRET).get("doc") == name


def test_exp_omitted_when_expiry_disabled():
    assert "exp" not in jwt.verify(jwt.sign({"role": "reader"}, SECRET, expires_in=0), SECRET)
