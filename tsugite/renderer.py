"""Jinja2 template rendering for agent content."""

from jinja2 import Environment, DictLoader, StrictUndefined
from datetime import datetime
from typing import Dict, Any
import os


def now() -> str:
    """Return current timestamp."""
    return datetime.now().isoformat()


def today() -> str:
    """Return today's date."""
    return datetime.now().strftime("%Y-%m-%d")


def slugify(text: str) -> str:
    """Convert text to slug format."""
    import re

    text = text.lower()
    # Replace special characters with dashes, then keep only ASCII letters, numbers, and dashes
    text = re.sub(r"[^\w\s-]", "-", text, flags=re.ASCII)
    text = re.sub(r"[-\s]+", "-", text)
    return text.strip("-")


def file_exists(path: str) -> bool:
    """Check if a file or directory exists."""
    from pathlib import Path

    return Path(path).exists()


def is_file(path: str) -> bool:
    """Check if path exists and is a file."""
    from pathlib import Path

    p = Path(path)
    return p.exists() and p.is_file()


def is_dir(path: str) -> bool:
    """Check if path exists and is a directory."""
    from pathlib import Path

    p = Path(path)
    return p.exists() and p.is_dir()


def read_text(path: str, default: str = "") -> str:
    """Safely read file content, returning default on error."""
    from pathlib import Path

    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return default


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
                "slugify": slugify,
                "file_exists": file_exists,
                "is_file": is_file,
                "is_dir": is_dir,
                "read_text": read_text,
                "env": dict(os.environ),
            }
        )

        # Add filters
        self.env.filters["slugify"] = slugify

    def render(self, content: str, context: Dict[str, Any] = None) -> str:
        """Render agent content with Jinja2."""
        if context is None:
            context = {}

        try:
            template = self.env.from_string(content)
            return template.render(**context)
        except Exception as e:
            raise ValueError(f"Template rendering failed: {e}")

    def render_with_variables(self, content: str, user_prompt: str = "", variables: Dict[str, Any] = None) -> str:
        """Render content with standard agent variables."""
        if variables is None:
            variables = {}

        context = {"user_prompt": user_prompt, **variables}

        return self.render(content, context)
