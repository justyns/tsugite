"""User-Agent header management for outbound HTTP requests."""

import logging

logger = logging.getLogger(__name__)

_user_agent_cached: str | None = ...  # type: ignore[assignment]  # ellipsis sentinel


def get_user_agent() -> str | None:
    """Return User-Agent string from config, or None if disabled.

    Config values: null/unset → default, non-empty string → that string, "" → None (disabled).
    Cached after first call since the value is constant for the process lifetime.
    """
    global _user_agent_cached
    if _user_agent_cached is not ...:
        return _user_agent_cached

    from tsugite import __version__
    from tsugite.config import load_config

    configured = load_config().user_agent
    if configured is not None:
        _user_agent_cached = configured if configured else None
        return _user_agent_cached

    try:
        from importlib.metadata import version

        v = version("tsugite-cli")
    except Exception:
        v = __version__
    _user_agent_cached = f"Tsugite/{v} (+https://github.com/justyns/tsugite)"
    return _user_agent_cached


def set_user_agent_header(headers: dict[str, str]) -> None:
    """Force the framework User-Agent on outbound headers.

    Any caller-supplied User-Agent (any case) is dropped with a warning.
    The framework value cannot be overridden from agent or skill code; this
    closes a PII-leak class where agent code copied identifiers from system
    context into a hand-rolled UA.

    No-op when UA management is disabled (config user_agent=""): caller-set
    UA is left untouched. Disable means hands-off.
    """
    ua = get_user_agent()
    if not ua:
        return
    existing = None
    for key in [k for k in headers if k.lower() == "user-agent"]:
        popped = headers.pop(key)
        if existing is None:
            existing = popped
    if existing and existing != ua:
        logger.warning(
            "Dropped caller-supplied User-Agent %r; framework UA %r is enforced",
            existing[:100],
            ua,
        )
    headers["User-Agent"] = ua
