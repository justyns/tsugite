"""Skill discovery system for Tsugite.

Scans directories for skill files (markdown with YAML frontmatter)
and builds an index for efficient discovery.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from tsugite.utils import parse_yaml_frontmatter

logger = logging.getLogger(__name__)


@dataclass
class SkillMeta:
    """Lightweight skill metadata from frontmatter."""

    name: str
    description: str
    path: Path


def get_builtin_skills_path() -> Path:
    """Get the built-in skills directory path.

    Returns:
        Path to builtin_skills directory in the package
    """
    return Path(__file__).parent / "builtin_skills"


def scan_skills() -> List[SkillMeta]:
    """Scan all skill directories and extract frontmatter.

    Only reads YAML frontmatter, not full file content.
    This is the key to token efficiency.

    Search order (highest to lowest priority):
    1. .tsugite/skills/     - Project-local
    2. skills/              - Project convention
    3. builtin_skills/      - Built-in (package)
    4. ~/.config/tsugite/skills/  - Global user

    Returns:
        List of SkillMeta objects for discovered skills
    """
    skill_paths = [
        Path(".tsugite/skills"),  # 1. Project-local
        Path("skills"),  # 2. Project convention
        get_builtin_skills_path(),  # 3. Built-in (package)
        Path.home() / ".config" / "tsugite" / "skills",  # 4. Global user
    ]

    skills = []
    seen_names = set()

    for skill_dir in skill_paths:
        skill_dir = skill_dir.resolve()

        if not skill_dir.exists():
            continue

        for skill_file in skill_dir.glob("**/*.md"):
            try:
                frontmatter, _ = parse_yaml_frontmatter(skill_file.read_text())

                if "name" not in frontmatter:
                    logger.debug(f"Skipping skill file {skill_file}: missing 'name' in frontmatter")
                    continue

                skill_name = frontmatter["name"]

                if skill_name in seen_names:
                    logger.debug(f"Skipping skill '{skill_name}' in {skill_file}: duplicate name")
                    continue

                # Warn if description is missing (not required, but recommended)
                if "description" not in frontmatter:
                    logger.warning(f"Skill '{skill_name}' in {skill_file} missing 'description' field")

                skill = SkillMeta(
                    name=skill_name,
                    description=frontmatter.get("description", ""),
                    path=skill_file,
                )

                skills.append(skill)
                seen_names.add(skill_name)

            except Exception as e:
                # Log parse failures for debugging
                logger.warning(f"Failed to parse skill file {skill_file}: {e}")
                continue

    return skills


def build_skill_index(skills: List[SkillMeta]) -> str:
    """Build skill index for LLM.

    Format: - name: description

    This is what the LLM sees at session start.
    Total: ~20 tokens per skill.

    Args:
        skills: List of SkillMeta objects

    Returns:
        Formatted string index of skills
    """
    if not skills:
        return ""

    lines = []
    for skill in skills:
        line = f"- {skill.name}: {skill.description}"
        lines.append(line)

    return "\n".join(lines)
