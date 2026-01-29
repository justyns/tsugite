"""Soul template system for workspace creation."""

from pathlib import Path
from typing import List, Optional

from jinja2 import Template


def _get_templates_dir() -> Path:
    """Get path to soul templates directory.

    Returns:
        Path to templates/souls directory
    """
    return Path(__file__).parent.parent / "templates" / "souls"


def list_soul_templates() -> List[str]:
    """List available soul templates.

    Returns:
        List of template names (without .md extension)
    """
    templates_dir = _get_templates_dir()
    if not templates_dir.exists():
        return []
    return sorted([p.stem for p in templates_dir.glob("*.md")])


def load_soul_template(name: str, user_name: Optional[str] = None) -> str:
    """Load and render soul template using Jinja2.

    Args:
        name: Template name (without .md extension)
        user_name: Optional user name for template rendering

    Returns:
        Rendered template content

    Raises:
        ValueError: If template not found
    """
    templates_dir = _get_templates_dir()
    template_path = templates_dir / f"{name}.md"

    if not template_path.exists():
        available = ", ".join(list_soul_templates())
        raise ValueError(f"Soul template not found: {name}. Available: {available or 'none'}")

    content = template_path.read_text()

    if user_name:
        template = Template(content)
        content = template.render(user_name=user_name)

    return content


__all__ = ["list_soul_templates", "load_soul_template"]
