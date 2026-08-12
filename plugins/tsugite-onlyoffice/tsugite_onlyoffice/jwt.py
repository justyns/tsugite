"""HS256 JSON Web Tokens for the ONLYOFFICE Docs server handshake.

The document server speaks one algorithm with one shared secret, and this side
has to refuse every other one, so the whole thing fits in the stdlib.
"""

import base64
import hashlib
import hmac
import json
import time

_ALGORITHM = "HS256"


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _decode_segment(segment: str, label: str) -> dict:
    padding = "=" * (-len(segment) % 4)
    try:
        value = json.loads(base64.urlsafe_b64decode(segment + padding))
    except ValueError as exc:  # binascii.Error and JSONDecodeError are both ValueErrors
        raise ValueError(f"jwt: {label} segment is not valid base64url JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"jwt: {label} segment is not a JSON object")
    return value


def _signature(signing_input: str, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    return _b64url_encode(digest)


def sign(payload: dict, secret: str, expires_in: int = 300) -> str:
    """Sign a payload into a compact HS256 JWT.

    Args:
        payload: Claims to carry. Must be JSON-serializable.
        secret: The shared secret.
        expires_in: Seconds until the token expires, added as an `exp` claim.
            Falsy values leave `exp` off entirely.

    Returns:
        The `header.claims.signature` token.
    """
    claims = dict(payload)
    if expires_in:
        claims["exp"] = int(time.time()) + expires_in
    header = _b64url_encode(json.dumps({"alg": _ALGORITHM, "typ": "JWT"}, separators=(",", ":")).encode("utf-8"))
    body = _b64url_encode(json.dumps(claims, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header}.{body}"
    return f"{signing_input}.{_signature(signing_input, secret)}"


def verify(token: str, secret: str) -> dict:
    """Verify a compact HS256 JWT and return its claims.

    Args:
        token: The token to check.
        secret: The shared secret.

    Returns:
        The decoded claims.

    Raises:
        ValueError: The token is malformed, uses another algorithm, carries a bad
            signature, or has expired.
    """
    # A compact JWT is ASCII by definition, and both the segment encode and
    # hmac.compare_digest raise something other than ValueError without this.
    if not token.isascii():
        raise ValueError("jwt: token is not ascii")
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError(f"jwt: expected 3 segments, got {len(parts)}")
    header_seg, claims_seg, signature = parts

    # Pin the algorithm before touching the signature: a caller who picks it can
    # drop verification with "none" or make a public key act as the shared secret.
    header = _decode_segment(header_seg, "header")
    if header.get("alg") != _ALGORITHM:
        raise ValueError(f"jwt: unsupported alg {header.get('alg')!r}")

    if not hmac.compare_digest(_signature(f"{header_seg}.{claims_seg}", secret), signature):
        raise ValueError("jwt: signature mismatch")

    claims = _decode_segment(claims_seg, "claims")
    exp = claims.get("exp")
    if exp is not None and time.time() > exp:
        raise ValueError("jwt: token expired")
    return claims
