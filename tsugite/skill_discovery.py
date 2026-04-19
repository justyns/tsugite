"""Skill discovery system for Tsugite.

Scans directories for skills (directory containing a SKILL.md with YAML frontmatter)
and builds an index for efficient discovery.

Skill layout follows the agentskills.io specification:

    skill-name/
        SKILL.md            # required; name/description frontmatter + body
        scripts/            # optional; executable code bundled with the skill
        references/         # optional; supplementary documentation
        assets/             # optional; templates, data, static resources

Progressive disclosure: only frontmatter is read at scan time; the body and
bundled resources are loaded on demand by `SkillManager.load_skill`.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

from tsugite.utils import parse_yaml_frontmatter

logger = logging.getLogger(__name__)

SKILL_FILENAME = "SKILL.md"
_VALID_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_MAX_NAME_LENGTH = 64


@dataclass
class SkillMeta:
    """Lightweight skill metadata from frontmatter.

    Attributes:
        name: Skill name/identifier.
        description: One-line description shown in the skill index.
        directory: Absolute path to the skill directory (contains SKILL.md).
        skill_md_path: Absolute path to the SKILL.md file.
        triggers: Optional keywords for auto-loading (tsugite extension).
        ttl: Optional time-to-live in turns for sticky persistence (tsugite
            extension). None means fall back to the global config default.
            0 or negative means never expire.
    """

    name: str
    description: str
    directory: Path
    skill_md_path: Path
    triggers: List[str] = field(default_factory=list)
    ttl: Optional[int] = None


@dataclass
class Skill:
    """A loaded skill with its full content.

    Attributes:
        name: Skill name/identifier.
        content: Rendered SKILL.md body, optionally with an appended
            bundled-resources block.
        source_path: Optional path where the skill was loaded from.
    """

    name: str
    content: str
    source_path: str | None = None


def get_builtin_skills_path() -> Path:
    """Get the built-in skills directory path."""
    return Path(__file__).parent / "builtin_skills"


def _validate_skill_name(name: str, directory_name: str) -> List[str]:
    """Validate a skill name against the agentskills.io spec.

    Returns a list of human-readable warning strings. An empty list means
    the name is fully compliant. Clients should emit the warnings but still
    load the skill (lenient behavior per the spec's implementation guide).
    """
    warnings: List[str] = []
    if len(name) > _MAX_NAME_LENGTH:
        warnings.append(f"name '{name}' exceeds {_MAX_NAME_LENGTH} characters")
    if not _VALID_NAME_RE.match(name):
        warnings.append(
            f"name '{name}' is not spec-compliant (must be lowercase alphanumeric + hyphens,"
            " no leading/trailing/consecutive hyphens)"
        )
    if name != directory_name:
        warnings.append(f"name '{name}' does not match directory '{directory_name}'")
    return warnings


def _collect_skill_roots(workspace, extra_paths: Optional[List[str]]) -> List[Path]:
    """Build the ordered list of directories to scan for skills.

    Priority (highest to lowest):
      1. workspace/.agents/skills, workspace/skills (if a workspace is provided)
      2. user-configured extra_paths
      3. <cwd>/.agents/skills, <cwd>/.tsugite/skills, <cwd>/skills
      4. builtin_skills (ships with the package)
      5. ~/.agents/skills, ~/.config/tsugite/skills
    """
    roots: List[Path] = []

    if workspace is not None:
        workspace_path = getattr(workspace, "path", None) or getattr(workspace, "root", None)
        if workspace_path:
            roots.append(Path(workspace_path) / ".agents" / "skills")
        skills_dir = getattr(workspace, "skills_dir", None)
        if skills_dir:
            roots.append(Path(skills_dir))

    if extra_paths:
        for p in extra_paths:
            roots.append(Path(p).expanduser())

    roots.extend(
        [
            Path(".agents/skills"),
            Path(".tsugite/skills"),
            Path("skills"),
            get_builtin_skills_path(),
            Path.home() / ".agents" / "skills",
            Path.home() / ".config" / "tsugite" / "skills",
        ]
    )
    return roots


def scan_skills(workspace=None, extra_paths: Optional[List[str]] = None) -> List[SkillMeta]:
    """Scan all skill roots and return metadata for every discovered skill.

    A skill is any immediate subdirectory of a root that contains a SKILL.md
    file. Only frontmatter is read here; bodies and bundled resources are
    loaded on demand by SkillManager.

    The first occurrence of a given skill name wins; later duplicates are
    logged and skipped so project-level skills reliably override built-ins.

    Args:
        workspace: Optional workspace object; checked first when provided.
        extra_paths: Optional user-configured directories to scan before the
            default project/user paths.

    Returns:
        Ordered list of SkillMeta objects.
    """
    skills: List[SkillMeta] = []
    seen_names: Set[str] = set()

    for root in _collect_skill_roots(workspace, extra_paths):
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if not resolved.is_dir():
            continue

        for skill_dir in sorted(p for p in resolved.iterdir() if p.is_dir()):
            skill_md = skill_dir / SKILL_FILENAME
            if not skill_md.is_file():
                continue

            try:
                frontmatter, _ = parse_yaml_frontmatter(skill_md.read_text())
            except Exception as exc:
                logger.warning(f"Failed to parse {skill_md}: {exc}")
                continue

            name = frontmatter.get("name")
            if not name:
                logger.debug(f"Skipping {skill_md}: missing 'name' in frontmatter")
                continue

            if name in seen_names:
                logger.debug(f"Skipping '{name}' at {skill_md}: already discovered with higher priority")
                continue

            for warning in _validate_skill_name(name, skill_dir.name):
                logger.warning(f"Skill {skill_md}: {warning}")

            description = frontmatter.get("description", "")
            if not description:
                logger.warning(f"Skill '{name}' at {skill_md} has no 'description' (recommended)")

            triggers_raw = frontmatter.get("triggers") or []
            if not isinstance(triggers_raw, list):
                logger.warning(f"Skill '{name}' at {skill_md}: 'triggers' must be a list; ignoring")
                triggers = []
            else:
                triggers = []
                for item in triggers_raw:
                    if isinstance(item, str):
                        triggers.append(item)
                    else:
                        logger.warning(
                            f"Skill '{name}' at {skill_md}: trigger {item!r} is not a string; ignoring"
                        )

            ttl_raw = frontmatter.get("ttl")
            ttl: Optional[int] = None
            if ttl_raw is not None:
                if isinstance(ttl_raw, bool) or not isinstance(ttl_raw, int):
                    logger.warning(f"Skill '{name}' at {skill_md}: 'ttl' must be an integer; ignoring")
                else:
                    ttl = ttl_raw

            skills.append(
                SkillMeta(
                    name=name,
                    description=description,
                    directory=skill_dir,
                    skill_md_path=skill_md,
                    triggers=triggers,
                    ttl=ttl,
                )
            )
            seen_names.add(name)

    return skills


_WORD_SPLIT = re.compile(r"\W+")


def _extract_words(text: str) -> Set[str]:
    """Split text into a set of lowercase words on non-word boundaries."""
    return set(_WORD_SPLIT.split(text.lower()))


def find_referenced_skills(text: str, skills: List[SkillMeta]) -> Set[str]:
    """Return names of skills whose name or any trigger appears in text.

    Uses the same word-boundary, case-insensitive matching as trigger dispatch
    so "keeping the skill alive" stays symmetric with "how it got loaded."
    """
    if not text:
        return set()
    words = _extract_words(text)
    referenced: Set[str] = set()
    for skill in skills:
        for candidate in (skill.name, *skill.triggers):
            if candidate and candidate.lower() in words:
                referenced.add(skill.name)
                break
    return referenced


def match_triggered_skills(
    message: str,
    skills: List[SkillMeta],
    already_loaded: Set[str] | None = None,
    max_skills: int = 3,
) -> List[SkillMeta]:
    """Find skills whose trigger keywords appear in the message.

    Uses word-boundary matching (case-insensitive). Skills are ranked by
    number of matching triggers so more specific matches come first.

    Triggers are a tsugite extension and not part of the agentskills.io spec.

    Args:
        message: User message to scan for trigger keywords.
        skills: All available skills to check.
        already_loaded: Skill names to skip (already loaded).
        max_skills: Maximum number of skills to return.

    Returns:
        List of matching SkillMeta objects, up to max_skills.
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

    This is what the LLM sees at session start (~20 tokens per skill).
    """
    if not skills:
        return ""

    lines = [f"- {skill.name}: {skill.description}" for skill in skills]
    return "\n".join(lines)
