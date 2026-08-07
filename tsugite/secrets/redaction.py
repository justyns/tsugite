"""Key/path-based redaction for tool arguments.

Complements the value-based masking in `registry.py`: that one replaces values
it was told about (`get_secret()` results), this one replaces values it can
recognise by *where they sit* - an `Authorization` header, a declared
`body.password` - so a token derived at runtime and never registered still
doesn't reach an audit event, history, or replay.

Deliberately no regex/entropy detection: v1 is exact keys and declared paths.
"""

from typing import Any, Iterable, Sequence

REDACTED = "***"

# Redacted wherever they appear, at any depth. Lowercase; matching is
# case-insensitive.
SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "x-auth-token",
        "x-csrf-token",
    }
)

# Kept in the redacted value ("Bearer ***"): which scheme was used is useful
# when reading an audit trail, and it isn't the secret.
_AUTH_SCHEMES = ("bearer", "basic", "digest", "token", "negotiate")


def _redact_value(value: Any) -> Any:
    """Redact one value, keeping a recognised auth scheme prefix."""
    if not isinstance(value, str):
        return REDACTED
    scheme, _, rest = value.partition(" ")
    if rest and scheme.lower() in _AUTH_SCHEMES:
        return f"{scheme} {REDACTED}"
    return REDACTED


def _split_paths(sensitive_paths: Iterable[str] | None) -> list[Sequence[str]]:
    if not sensitive_paths:
        return []
    return [tuple(seg.lower() for seg in path.split(".") if seg) for path in sensitive_paths if path]


def redact_sensitive_obj(obj: Any, sensitive_paths: Iterable[str] | None = None) -> Any:
    """Return a copy of `obj` with sensitive values replaced by `***`.

    Redacts a dict value when its key is a built-in sensitive key (case-
    insensitive, at any depth) or when its position matches one of
    `sensitive_paths` (dotted, e.g. `headers.Authorization`, `body.password`).
    Lists are transparent to path matching, so `items.token` matches every
    element of `{"items": [{"token": ...}]}`.

    Args:
        obj: Arbitrary nested structure (dict/list/tuple/scalars).
        sensitive_paths: Extra dotted paths to redact, e.g. a tool's declared ones.

    Returns:
        A redacted copy; the input is never mutated.
    """
    return _walk(obj, (), _split_paths(sensitive_paths))


def _walk(obj: Any, path: Sequence[str], paths: list[Sequence[str]]) -> Any:
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            child = (*path, str(key).lower())
            if child[-1] in SENSITIVE_KEYS or child in paths:
                out[key] = _redact_value(value)
            else:
                out[key] = _walk(value, child, paths)
        return out
    if isinstance(obj, list):
        # Lists are transparent: an element sits at its container's path, so a
        # declared `items.token` covers every element without index bookkeeping.
        return [_walk(v, path, paths) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_walk(v, path, paths) for v in obj)
    return obj
