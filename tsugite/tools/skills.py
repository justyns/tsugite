"""Skill loading tools for Tsugite agents."""

import logging
import os
from typing import Dict, List, Optional

from tsugite import renderer
from tsugite.renderer import AgentRenderer
from tsugite.skill_discovery import SkillMeta, build_skill_index, scan_skills
from tsugite.tools import tool
from tsugite.utils import parse_yaml_frontmatter

logger = logging.getLogger(__name__)


class SkillManager:
    """Manages skill discovery, loading, and lifecycle events.

    This class replaces module-level globals and integrates with the event system.
    An instance should be created per agent/session and passed to tools.
    """

    def __init__(self, workspace=None):
        """Initialize skill manager.

        Args:
            workspace: Optional workspace to check for workspace-specific skills
        """
        self._skill_registry: Dict[str, SkillMeta] = {}
        self._loaded_skills: Dict[str, str] = {}
        self._registry_initialized = False
        self._workspace = workspace
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
            skills = scan_skills(workspace=self._workspace)
            self._skill_registry = {skill.name: skill for skill in skills}
            self._registry_initialized = True

    def load_skill(self, skill_name: str) -> str:
        """Load a skill to gain additional capabilities.

        The skill content is rendered with Jinja2 and added to the agent's context.
        Once loaded, the skill persists for the session.

        Args:
            skill_name: Name of the skill to load (from available skills list)

        Returns:
            Success message or error description
        """
        self._ensure_registry_initialized()

        if skill_name in self._loaded_skills:
            return f"Skill '{skill_name}' is already loaded in this session"

        skill = self._skill_registry.get(skill_name)
        if not skill:
            available = ", ".join(sorted(self._skill_registry.keys()))
            if available:
                return f"Skill '{skill_name}' not found. Available skills: {available}"
            else:
                return f"Skill '{skill_name}' not found. No skills available."

        try:
            skill_text = skill.path.read_text()
            frontmatter, content = parse_yaml_frontmatter(skill_text)

            # Validate frontmatter fields
            if "name" not in frontmatter:
                return f"Failed to load skill '{skill_name}': missing 'name' in frontmatter"

            if "description" not in frontmatter or not frontmatter["description"]:
                logger.warning(f"Skill '{skill_name}' missing 'description' field (recommended)")

            agent_renderer = AgentRenderer()
            context = {
                "user_prompt": "",  # Empty for skills
                "today": renderer.today,
                "now": renderer.now,
                "env": os.environ,
            }
            rendered_content = agent_renderer.render(content, context)

            self._loaded_skills[skill_name] = rendered_content

            # Register with executor for embedding in observation
            if self._executor and hasattr(self._executor, "register_loaded_skill"):
                self._executor.register_loaded_skill(skill_name, rendered_content)

            from tsugite.events.helpers import emit_skill_loaded_event

            emit_skill_loaded_event(skill_name=skill_name, description=skill.description)

            return f"✓ Successfully loaded skill: {skill_name}"

        except Exception as e:
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

    def clear_loaded_skills(self):
        """Clear all loaded skills.

        Useful for testing and session cleanup.
        """
        self._loaded_skills.clear()


# Global skill manager instance for backward compatibility
# This will be replaced with instance-based access in the future
_default_skill_manager: Optional[SkillManager] = None


def get_skill_manager() -> SkillManager:
    """Get or create the default skill manager instance.

    This is a temporary helper for backward compatibility.
    In the future, skill managers will be created per agent/session.

    Returns:
        Default SkillManager instance
    """
    global _default_skill_manager
    if _default_skill_manager is None:
        _default_skill_manager = SkillManager()
    return _default_skill_manager


def set_skill_manager(manager: SkillManager):
    """Set the default skill manager instance.

    Args:
        manager: SkillManager instance to use as default
    """
    global _default_skill_manager
    _default_skill_manager = manager


@tool
def load_skill(skill_name: str) -> str:
    """Load a skill to gain additional capabilities.

    The skill content is rendered with Jinja2 and added to the agent's context.
    Once loaded, the skill persists for the session.

    Args:
        skill_name: Name of the skill to load (from available skills list)

    Returns:
        Success message or error description
    """
    manager = get_skill_manager()
    return manager.load_skill(skill_name)


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
