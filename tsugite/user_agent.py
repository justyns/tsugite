"""User-Agent header management for outbound HTTP requests."""

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
    """Set User-Agent on headers dict if configured and not already present."""
    ua = get_user_agent()
    if ua:
        headers.setdefault("User-Agent", ua)
