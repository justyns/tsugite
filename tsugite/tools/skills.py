"""Skill loading tools for Tsugite agents."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from tsugite import renderer
from tsugite.renderer import AgentRenderer
from tsugite.skill_discovery import (
    SkillIssue,
    SkillIssueSource,
    SkillMeta,
    build_skill_index,
    match_triggered_skills,
    scan_skills_with_issues,
)
from tsugite.tools import tool
from tsugite.utils import parse_yaml_frontmatter

logger = logging.getLogger(__name__)

_BUNDLED_SUBDIRS = ("scripts", "references", "assets")
_MAX_RESOURCES_LISTED = 10


def _failed_skill_dict(
    name: Optional[str],
    source: SkillIssueSource,
    path,
    severity: str,
    message: str,
) -> Dict[str, str]:
    return {
        "name": name or "?",
        "source": source,
        "path": str(path),
        "severity": severity,
        "message": message,
    }


def _enumerate_bundled_resources(skill_dir: Path) -> List[str]:
    """List one-level-deep files under scripts/, references/, assets/.

    Returns skill-relative POSIX paths (e.g., "scripts/build.sh"), sorted
    first by subdirectory (scripts, references, assets) then by filename.
    """
    resources: List[str] = []
    for sub in _BUNDLED_SUBDIRS:
        sub_path = skill_dir / sub
        if not sub_path.is_dir():
            continue
        for entry in sorted(sub_path.iterdir(), key=lambda p: p.name):
            if entry.is_file():
                resources.append(f"{sub}/{entry.name}")
    return resources


def _append_resource_block(body: str, skill_dir: Path, resources: List[str]) -> str:
    """Append a <skill_resources> block when the skill bundles any resources.

    Wraps the enumerated file list in the tag recommended by the agentskills.io
    client-implementation doc so the block is identifiable by downstream tooling.
    Returns body unchanged when the skill has no scripts/, references/, or assets/.
    """
    if not resources:
        return body

    shown = resources[:_MAX_RESOURCES_LISTED]
    lines: List[str] = [
        body.rstrip(),
        "",
        f'<skill_resources dir="{skill_dir}">',
    ]
    lines.extend(f"- {path}" for path in shown)
    remaining = len(resources) - len(shown)
    if remaining > 0:
        lines.append(f"- ... (+{remaining} more)")
    lines.append("</skill_resources>")
    return "\n".join(lines) + "\n"


class SkillManager:
    """Manages skill discovery, loading, and lifecycle events.

    This class replaces module-level globals and integrates with the event system.
    An instance should be created per agent/session and passed to tools.
    """

    def __init__(self, workspace=None, extra_paths: Optional[List[str]] = None):
        """Initialize skill manager.

        Args:
            workspace: Optional workspace to check for workspace-specific skills
            extra_paths: Optional list of additional directory paths to search for skills
        """
        self._skill_registry: Dict[str, SkillMeta] = {}
        self._scan_issues: List[SkillIssue] = []
        self._load_failures: Dict[str, str] = {}
        self._loaded_skills: Dict[str, str] = {}
        self._registry_initialized = False
        self._workspace = workspace
        self._extra_paths = extra_paths
        self._executor = None

    def set_executor(self, executor):
        """Set executor reference for skill tracking.

        When set, loaded skills are registered with the executor for
        embedding in the observation of the current turn.

        Args:
            executor: LocalExecutor instance
        """
        self._executor = executor

    def _ensure_registry_initialized(self):
        """Initialize skill registry if not already initialized."""
        if not self._registry_initialized:
            skills, issues = scan_skills_with_issues(workspace=self._workspace, extra_paths=self._extra_paths)
            self._skill_registry = {skill.name: skill for skill in skills}
            self._scan_issues = issues
            self._registry_initialized = True

    def load_skill(self, skill_name: str) -> str:
        """Load a skill to gain additional capabilities.

        Reads SKILL.md from the skill directory, renders the body with Jinja2,
        and appends an absolute-path + bundled-resource summary so the agent
        can locate scripts/references/assets on demand.

        If the skill is already loaded, this still registers the call with the
        executor so the daemon's TTL bookkeeping can treat it as a renewal
        (reference resets the unused-turns counter to 0). Content is not
        re-read from disk in that case.

        Args:
            skill_name: Name of the skill to load (from available skills list)

        Returns:
            Success message or error description
        """
        self._ensure_registry_initialized()

        already_loaded = skill_name in self._loaded_skills
        if already_loaded:
            rendered_content = self._loaded_skills[skill_name]
            if self._executor and hasattr(self._executor, "register_loaded_skill"):
                self._executor.register_loaded_skill(skill_name, rendered_content)
            return f"Skill '{skill_name}' is already loaded (TTL renewed)"

        skill = self._skill_registry.get(skill_name)
        if not skill:
            available = ", ".join(sorted(self._skill_registry.keys()))
            if available:
                return f"Skill '{skill_name}' not found. Available skills: {available}"
            else:
                return f"Skill '{skill_name}' not found. No skills available."

        try:
            skill_text = skill.skill_md_path.read_text()
            frontmatter, content = parse_yaml_frontmatter(skill_text)

            if "name" not in frontmatter:
                return f"Failed to load skill '{skill_name}': missing 'name' in frontmatter"

            if "description" not in frontmatter or not frontmatter["description"]:
                logger.warning(f"Skill '{skill_name}' missing 'description' field (recommended)")

            agent_renderer = AgentRenderer()
            context = {
                "user_prompt": "",
                "today": renderer.today,
                "now": renderer.now,
                "env": os.environ,
            }
            rendered_body = agent_renderer.render(content, context)
            resources = _enumerate_bundled_resources(skill.directory)
            rendered_content = _append_resource_block(rendered_body, skill.directory, resources)

            self._loaded_skills[skill_name] = rendered_content
            self._load_failures.pop(skill_name, None)

            if self._executor and hasattr(self._executor, "register_loaded_skill"):
                self._executor.register_loaded_skill(skill_name, rendered_content)

            from tsugite.events.helpers import emit_skill_loaded_event

            emit_skill_loaded_event(skill_name=skill_name, description=skill.description)

            return f"✓ Successfully loaded skill: {skill_name}"

        except Exception as e:
            self._load_failures[skill_name] = str(e)
            logger.error(f"Failed to load skill '{skill_name}': {e}", exc_info=True)
            return f"Failed to load skill '{skill_name}': {str(e)}"

    def unload_skill(self, skill_name: str) -> str:
        """Unload a previously loaded skill.

        Args:
            skill_name: Name of the skill to unload

        Returns:
            Success message or error description
        """
        if skill_name not in self._loaded_skills:
            return f"Skill '{skill_name}' is not currently loaded"

        del self._loaded_skills[skill_name]

        if self._executor and hasattr(self._executor, "register_unloaded_skill"):
            self._executor.register_unloaded_skill(skill_name)

        from tsugite.events.helpers import emit_skill_unloaded_event

        emit_skill_unloaded_event(skill_name=skill_name)

        return f"✓ Successfully unloaded skill: {skill_name}"

    def list_available_skills(self) -> str:
        """List all discoverable skills.

        Returns:
            Formatted list of available skills with descriptions
        """
        self._ensure_registry_initialized()

        if not self._skill_registry:
            return "No skills available"

        skills = list(self._skill_registry.values())
        return build_skill_index(skills)

    def list_loaded_skills(self) -> str:
        """Show which skills are currently loaded in this session.

        Returns:
            List of loaded skill names
        """
        if not self._loaded_skills:
            return "No skills currently loaded"

        skill_names = sorted(self._loaded_skills.keys())
        return "Loaded skills:\n" + "\n".join(f"- {name}" for name in skill_names)

    def get_loaded_skills(self) -> Dict[str, str]:
        """Get all currently loaded skills.

        Returns:
            Dict mapping skill names to rendered content
        """
        return self._loaded_skills.copy()

    def get_failed_skills_list(self) -> List[Dict[str, str]]:
        """Return scan + load failures merged for user-facing surfaces."""
        self._ensure_registry_initialized()
        items: List[Dict[str, str]] = [
            _failed_skill_dict(issue.name, "scan", issue.path, issue.severity, issue.message)
            for issue in self._scan_issues
        ]
        for name, message in self._load_failures.items():
            meta = self._skill_registry.get(name)
            path = meta.skill_md_path if meta else "?"
            items.append(_failed_skill_dict(name, "load", path, "error", message))
        return items

    def get_triggered_skills(self, message: str, max_skills: int = 3) -> List[str]:
        """Find skills that should auto-load based on trigger keywords in the message.

        Args:
            message: User message to scan for trigger keywords
            max_skills: Maximum number of triggered skills to return

        Returns:
            List of skill names that matched
        """
        self._ensure_registry_initialized()
        already_loaded = set(self._loaded_skills.keys())
        matched = match_triggered_skills(message, list(self._skill_registry.values()), already_loaded, max_skills)
        return [skill.name for skill in matched]

    def clear_loaded_skills(self):
        """Clear all loaded skills.

        Useful for testing and session cleanup.
        """
        self._loaded_skills.clear()


# Global skill manager instance for backward compatibility
# This will be replaced with instance-based access in the future
_default_skill_manager: Optional[SkillManager] = None


def get_skill_manager(workspace=None) -> SkillManager:
    """Get or create the default skill manager instance.

    This is a temporary helper for backward compatibility.
    In the future, skill managers will be created per agent/session.

    Args:
        workspace: Optional workspace for skill discovery fallback

    Returns:
        Default SkillManager instance
    """
    global _default_skill_manager
    if _default_skill_manager is None:
        from tsugite.config import load_config

        config = load_config()
        _default_skill_manager = SkillManager(workspace=workspace, extra_paths=config.skill_paths or None)
    return _default_skill_manager


def set_skill_manager(manager: SkillManager):
    """Set the default skill manager instance.

    Args:
        manager: SkillManager instance to use as default
    """
    global _default_skill_manager
    _default_skill_manager = manager


@tool(parent_only=True)
def load_skill(skill_name: str) -> str:
    """Load a skill to gain additional capabilities.

    The skill content is rendered with Jinja2 and added to the agent's context.
    Once loaded, the skill persists for the session. Calling load_skill on an
    already-loaded skill renews its TTL (resets the unused-turn counter).

    Args:
        skill_name: Name of the skill to load (from available skills list)

    Returns:
        Success message or error description
    """
    manager = get_skill_manager()
    return manager.load_skill(skill_name)


@tool(parent_only=True)
def unload_skill(skill_name: str) -> str:
    """Drop a previously loaded skill from the session to free context tokens.

    Use this when a skill is no longer needed. The skill stays discoverable and
    can be loaded again later with load_skill. TTL auto-expiry calls this path
    internally after N turns of no reference.

    Args:
        skill_name: Name of the skill to unload

    Returns:
        Success message or error description
    """
    manager = get_skill_manager()
    return manager.unload_skill(skill_name)


@tool
def list_available_skills() -> str:
    """List all discoverable skills.

    Returns:
        Formatted list of available skills with descriptions
    """
    manager = get_skill_manager()
    return manager.list_available_skills()


@tool
def get_skills_for_template() -> List[Dict[str, str]]:
    """Get skills as a list of dicts for template rendering.

    Returns:
        List of dicts with 'name' and 'description' keys
    """
    return get_available_skills_list()


@tool
def get_failed_skills_for_template() -> List[Dict[str, str]]:
    """Get skill load failures as a list of dicts for template rendering.

    Returns:
        List of dicts with 'name', 'source', 'path', 'severity', 'message' keys.
        Empty list when all skills are healthy.
    """
    return get_failed_skills_list()


def get_available_skills_list() -> List[Dict[str, str]]:
    """Get list of all available skills as dicts (for template rendering).

    This is a non-@tool function used internally by prefetch to provide
    skill data to templates in a format Jinja2 can iterate over.

    Returns:
        List of dicts with 'name' and 'description' keys
    """
    manager = get_skill_manager()
    manager._ensure_registry_initialized()

    if not manager._skill_registry:
        return []

    # Convert SkillMeta objects to dicts for template use
    return [{"name": skill.name, "description": skill.description} for skill in manager._skill_registry.values()]


def get_failed_skills_list() -> List[Dict[str, str]]:
    """Get list of skill load failures (scan + runtime) as dicts.

    Returns:
        List of dicts with 'name', 'source', 'path', 'severity', 'message' keys.
    """
    manager = get_skill_manager()
    return manager.get_failed_skills_list()


def get_loaded_skills() -> Dict[str, str]:
    """Get all currently loaded skills.

    Returns:
        Dict mapping skill names to rendered content
    """
    manager = get_skill_manager()
    return manager.get_loaded_skills()


def clear_loaded_skills():
    """Clear all loaded skills.

    Useful for testing and session cleanup.
    """
    manager = get_skill_manager()
    manager.clear_loaded_skills()
