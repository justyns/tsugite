"""Skill discovery system for Tsugite.

Scans directories for skill files (markdown with YAML frontmatter)
and builds an index for efficient discovery.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

from tsugite.utils import parse_yaml_frontmatter

logger = logging.getLogger(__name__)


@dataclass
class SkillMeta:
    """Lightweight skill metadata from frontmatter."""

    name: str
    description: str
    path: Path
    triggers: List[str] = field(default_factory=list)


@dataclass
class Skill:
    """A loaded skill with its full content.

    Attributes:
        name: Skill name/identifier
        content: Full rendered content of the skill
        source_path: Optional path where the skill was loaded from
    """

    name: str
    content: str
    source_path: str | None = None


def get_builtin_skills_path() -> Path:
    """Get the built-in skills directory path.

    Returns:
        Path to builtin_skills directory in the package
    """
    return Path(__file__).parent / "builtin_skills"


def scan_skills(workspace=None, extra_paths: Optional[List[str]] = None) -> List[SkillMeta]:
    """Scan all skill directories and extract frontmatter with workspace priority.

    Only reads YAML frontmatter, not full file content.
    This is the key to token efficiency.

    Search order (highest to lowest priority):
    0. workspace/skills/    - Workspace-specific (if workspace provided)
    1. extra_paths          - User-configured additional paths
    2. .tsugite/skills/     - Project-local
    3. skills/              - Project convention
    4. builtin_skills/      - Built-in (package)
    5. ~/.config/tsugite/skills/  - Global user

    Args:
        workspace: Optional workspace to check for workspace-specific skills
        extra_paths: Optional list of additional directory paths to search

    Returns:
        List of SkillMeta objects for discovered skills
    """
    skill_paths = []

    # Workspace skills (highest priority)
    if workspace and hasattr(workspace, "skills_dir"):
        if workspace.skills_dir.exists():
            skill_paths.append(workspace.skills_dir)

    # User-configured extra paths
    if extra_paths:
        for p in extra_paths:
            skill_paths.append(Path(p).expanduser())

    # Project and system paths
    skill_paths.extend(
        [
            Path(".tsugite/skills"),
            Path("skills"),
            get_builtin_skills_path(),
            Path.home() / ".config" / "tsugite" / "skills",
        ]
    )

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
                    triggers=frontmatter.get("triggers", []),
                )

                skills.append(skill)
                seen_names.add(skill_name)

            except Exception as e:
                # Log parse failures for debugging
                logger.warning(f"Failed to parse skill file {skill_file}: {e}")
                continue

    return skills


_WORD_SPLIT = re.compile(r"\W+")


def match_triggered_skills(
    message: str,
    skills: List[SkillMeta],
    already_loaded: Set[str] | None = None,
    max_skills: int = 3,
) -> List[SkillMeta]:
    """Find skills whose trigger keywords appear in the message.

    Uses word-boundary matching (case-insensitive). Skills are ranked by
    number of matching triggers so more specific matches come first.

    Args:
        message: User message to scan for trigger keywords
        skills: All available skills to check
        already_loaded: Skill names to skip (already loaded)
        max_skills: Maximum number of skills to return

    Returns:
        List of matching SkillMeta objects, up to max_skills
    """
    already_loaded = already_loaded or set()
    message_words = set(_WORD_SPLIT.split(message.lower()))
    matches = []

    for skill in skills:
        if not skill.triggers or skill.name in already_loaded:
            continue

        match_count = sum(1 for t in skill.triggers if t.lower() in message_words)
        if match_count > 0:
            matches.append((match_count, skill))

    matches.sort(key=lambda x: x[0], reverse=True)
    return [skill for _, skill in matches[:max_skills]]


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
