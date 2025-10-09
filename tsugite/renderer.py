"""Jinja2 template rendering for agent content."""

import os
from datetime import datetime
from typing import Any, Dict

from jinja2 import DictLoader, Environment, StrictUndefined


def now() -> str:
    return datetime.now().isoformat()


def today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


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
