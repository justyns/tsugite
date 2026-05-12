"""Jinja2 template rendering for agent content."""

import os
import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, Optional

from jinja2 import DictLoader, Environment, StrictUndefined


class _CallableEnv(dict):
    """dict of env vars that also supports env('KEY', 'default') call syntax."""

    def __call__(self, key: str, default: str = "") -> str:
        return self.get(key, default)


@lru_cache(maxsize=1)
def local_tz():
    """Return the system local tzinfo, falling back to UTC if tzlocal can't resolve it.

    Cached because tzlocal does syscalls and the process timezone doesn't change
    at runtime; called per-stat in file tools and per-turn in the message context.
    """
    try:
        from tzlocal import get_localzone

        return get_localzone()
    except Exception:
        return timezone.utc


def parse_iso_utc(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 string (with or without `Z` suffix) into a tz-aware datetime.

    Returns None for empty input or unparseable values so callers can render
    with a graceful fallback rather than guarding every site with try/except.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError):
        return None


def now() -> str:
    return datetime.now().isoformat()


def today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def yesterday() -> str:
    return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


def tomorrow() -> str:
    return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")


def days_ago(n: int) -> datetime:
    """Return a timezone-aware datetime for N days ago."""
    return datetime.now(tz=local_tz()) - timedelta(days=n)


def weeks_ago(n: int) -> datetime:
    """Return a timezone-aware datetime for Monday of the ISO week N weeks ago."""
    dt = datetime.now(tz=local_tz()) - timedelta(weeks=n)
    return dt - timedelta(days=dt.weekday())


def date_format(dt, fmt: str) -> str:
    """Format a datetime (or date string) with strftime."""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    return dt.strftime(fmt)


def humanize_relative(dt: datetime, ref: datetime) -> str:
    """Render `dt` as a coarse human delta relative to `ref`.

    Picks the largest meaningful unit (just now / minutes / hours / days /
    weeks / months / years). Future or equal timestamps render as "just now"
    so callers don't have to special-case negative deltas.
    """
    delta = ref - dt
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "just now"

    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"

    hours = minutes // 60
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"

    days = hours // 24
    if days < 7:
        return f"{days} day{'s' if days != 1 else ''} ago"

    if days < 30:
        weeks = days // 7
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"

    if days < 365:
        months = days // 30
        return f"{months} month{'s' if months != 1 else ''} ago"

    years = days // 365
    return f"{years} year{'s' if years != 1 else ''} ago"


def format_prompt_ts(dt: datetime, *, ref: Optional[datetime] = None, tz_label: Optional[str] = None) -> str:
    """Render a tz-aware datetime as `YYYY-MM-DD HH:MM TZ`, optionally with a humanized delta.

    Used wherever the agent prompt or replay surfaces a timestamp. Stays
    byte-stable when `ref` is None (no relative phrase) so prompt-cache hits
    on replayed history aren't invalidated.
    """
    label = tz_label if tz_label is not None else (dt.strftime("%Z") or "UTC")
    out = dt.strftime("%Y-%m-%d %H:%M ") + label
    if ref is not None:
        out += f" ({humanize_relative(dt, ref)})"
    return out


def render_iso_element(name: str, raw: str, tz, tz_label: str, now: datetime) -> str:
    """Render `<name>YYYY-MM-DD HH:MM TZ (N units ago)</name>` from an ISO string.

    Returns "" if `raw` is missing or unparseable so the element is silently
    omitted rather than rendered broken. Adapter callers compose multiple of
    these into the message-context block.
    """
    dt = parse_iso_utc(raw)
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local = dt.astimezone(tz)
    return f"\n  <{name}>{format_prompt_ts(local, ref=now, tz_label=tz_label)}</{name}>"


def humanize_mtime(epoch: Any) -> str:
    """Render an epoch float as a coarse human delta (e.g. "6 days ago").

    Returns "" if epoch is missing or zero, so callers can append unconditionally.
    """
    if not epoch:
        return ""
    try:
        tz = local_tz()
        return humanize_relative(datetime.fromtimestamp(float(epoch), tz=tz), datetime.now(tz=tz))
    except (ValueError, TypeError, OSError):
        return ""


def slugify(text: str) -> str:
    import re

    text = text.lower()
    # Replace special characters with dashes, then keep only ASCII letters, numbers, and dashes
    text = re.sub(r"[^\w\s-]", "-", text, flags=re.ASCII)
    text = re.sub(r"[-\s]+", "-", text)
    return text.strip("-")


def file_exists(path: str) -> bool:
    from pathlib import Path

    return Path(path).exists()


def is_file(path: str) -> bool:
    from pathlib import Path

    p = Path(path)
    return p.exists() and p.is_file()


def is_dir(path: str) -> bool:
    from pathlib import Path

    p = Path(path)
    return p.exists() and p.is_dir()


def read_text(path: str, default: str = "") -> str:
    from pathlib import Path

    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return default


def strip_ignored_sections(content: str) -> str:
    """Remove <!-- tsu:ignore --> blocks from content.

    Supports both block and inline forms:
    - Block: <!-- tsu:ignore -->\\ncontent\\n<!-- /tsu:ignore -->
    - Inline: <!-- tsu:ignore -->content<!-- /tsu:ignore -->

    Args:
        content: Raw markdown content

    Returns:
        Content with ignored sections removed

    Example:
        >>> content = '''
        ... Normal content
        ... <!-- tsu:ignore -->
        ... This is ignored
        ... <!-- /tsu:ignore -->
        ... More content
        ... '''
        >>> result = strip_ignored_sections(content)
        >>> 'This is ignored' not in result
        True
    """
    # Pattern matches opening tag, content (non-greedy), closing tag
    # Handles both <!-- tsu:ignore --> and <!--tsu:ignore--> (with/without spaces)
    # Uses non-greedy match (.*?) to handle multiple blocks correctly
    # DOTALL flag allows matching across newlines
    pattern = r"<!--\s*tsu:ignore\s*-->.*?<!--\s*/tsu:ignore\s*-->"

    # Remove all ignore blocks
    result = re.sub(pattern, "", content, flags=re.DOTALL)

    return result


def cwd() -> str:
    """Get current working directory."""
    from pathlib import Path

    return str(Path.cwd())


def _tmux_sessions() -> list:
    """Get active tsugite-managed tmux sessions for Jinja2 templates.

    Returns [] when the tsugite-tmux-plugin package is not installed.
    """
    try:
        from tsugite_tmux import get_tmux_sessions

        return get_tmux_sessions()
    except (ImportError, FileNotFoundError, OSError):
        return []


class AgentRenderer:
    """Jinja2 template renderer for agent content."""

    def __init__(self):
        self.env = Environment(
            loader=DictLoader({}),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Add helper functions
        self.env.globals.update(
            {
                "now": now,
                "today": today,
                "yesterday": yesterday,
                "tomorrow": tomorrow,
                "timedelta": timedelta,
                "datetime": datetime,
                "slugify": slugify,
                "days_ago": days_ago,
                "weeks_ago": weeks_ago,
                "date_format": date_format,
                "humanize_relative": humanize_relative,
                "humanize_mtime": humanize_mtime,
                "file_exists": file_exists,
                "is_file": is_file,
                "is_dir": is_dir,
                "read_text": read_text,
                "env": _CallableEnv(os.environ),
                "cwd": cwd,
                "tmux_sessions": _tmux_sessions,
            }
        )

        # Deferred to avoid an import cycle: tsugite.tools.secrets depends on
        # tsugite.tools, which is loaded before this module in some import orders.
        from tsugite.tools.secrets import register_jinja_globals

        register_jinja_globals(self.env)

        # Add filters
        self.env.filters["slugify"] = slugify

    def render_string(self, content: str, context: Dict[str, Any] | None = None) -> str:
        """Render a template string without preprocessing (no ignore-block stripping).

        Use this for short values like prefetch tool args, where <!-- tsu:ignore --> is not
        a meaningful marker. Globals (get_secret, env, today, etc.) remain available.
        """
        if context is None:
            context = {}
        return self.env.from_string(content).render(**context)

    def render(self, content: str, context: Dict[str, Any] = None) -> str:
        """Render agent content with Jinja2.

        Preprocessing steps:
        1. Strip <!-- tsu:ignore --> blocks
        2. Render Jinja2 template with context

        Args:
            content: Raw markdown content
            context: Template variables

        Returns:
            Rendered content

        Raises:
            ValueError: If template rendering fails
        """
        if context is None:
            context = {}

        try:
            # Step 1: Strip ignored sections BEFORE rendering
            preprocessed = strip_ignored_sections(content)

            # Step 2: Render Jinja2 template
            template = self.env.from_string(preprocessed)
            return template.render(**context)
        except Exception as e:
            raise ValueError(f"Template rendering failed: {e}") from e
