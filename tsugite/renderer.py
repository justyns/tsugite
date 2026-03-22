"""Jinja2 template rendering for agent content."""

import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict

from jinja2 import DictLoader, Environment, StrictUndefined


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
    try:
        from tzlocal import get_localzone

        tz = get_localzone()
    except Exception:
        tz = None
    return datetime.now(tz=tz) - timedelta(days=n)


def weeks_ago(n: int) -> datetime:
    """Return a timezone-aware datetime for Monday of the ISO week N weeks ago."""
    try:
        from tzlocal import get_localzone

        tz = get_localzone()
    except Exception:
        tz = None
    dt = datetime.now(tz=tz) - timedelta(weeks=n)
    # Roll back to Monday of that ISO week
    dt = dt - timedelta(days=dt.weekday())
    return dt


def date_format(dt, fmt: str) -> str:
    """Format a datetime (or date string) with strftime."""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    return dt.strftime(fmt)


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
    """Get active tsugite-managed tmux sessions for Jinja2 templates."""
    try:
        from tsugite.tools.tmux import get_tmux_sessions

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
                "file_exists": file_exists,
                "is_file": is_file,
                "is_dir": is_dir,
                "read_text": read_text,
                "env": dict(os.environ),
                "cwd": cwd,
                "tmux_sessions": _tmux_sessions,
            }
        )

        # Add filters
        self.env.filters["slugify"] = slugify

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
