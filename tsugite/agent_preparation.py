"""Agent preparation pipeline - unified logic for render and execution."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from tsugite.attachments.base import Attachment  # noqa: E402
from tsugite.core.tools import Tool  # noqa: E402
from tsugite.md_agents import Agent, AgentConfig  # noqa: E402
from tsugite.skill_discovery import Skill  # noqa: E402

if TYPE_CHECKING:
    from tsugite.workspace import Workspace


def resolve_agent_config_attachments(
    attachment_templates: List[str],
    workspace_path: Optional[Path] = None,
) -> List[Attachment]:
    """Resolve agent config attachment templates to Attachment objects.

    Takes raw Jinja template strings from agent_config.attachments, renders them,
    resolves paths relative to workspace, and skips missing files.

    Args:
        attachment_templates: List of Jinja template strings for attachment paths
        workspace_path: Optional workspace path for resolving relative paths

    Returns:
        List of resolved Attachment objects
    """
    if not attachment_templates:
        return []

    from tsugite.attachments.file import FileHandler
    from tsugite.renderer import AgentRenderer

    renderer = AgentRenderer()
    file_handler = FileHandler()
    attachments: List[Attachment] = []

    for att_path_template in attachment_templates:
        try:
            rendered_path = renderer.env.from_string(att_path_template).render()
        except Exception as e:
            logger.debug("Failed to render attachment path %r: %s", att_path_template, e)
            continue

        resolved = Path(rendered_path)
        if not resolved.is_absolute() and workspace_path:
            resolved = workspace_path / resolved

        if not resolved.exists():
            logger.debug("Agent attachment not found (skipped): %s", resolved)
            continue

        att = file_handler.fetch(str(resolved))
        if att:
            attachments.append(att)
            logger.debug("Loaded agent attachment: %s", resolved)

    return attachments


@dataclass
class PreparedAgent:
    """Fully prepared agent ready for execution or display.

    This dataclass contains everything needed to either:
    1. Display what will be sent to the LLM (render command)
    2. Execute the agent (run command)

    Attributes:
        agent: Parsed agent object with content and config
        agent_config: Agent configuration (model, tools, etc.)
        system_message: Complete system message sent to LLM
        user_message: Complete user message sent to LLM
        rendered_prompt: Rendered template (before building system message)
        original_prompt: The user's prompt as typed (pre-template-rendering)
        tools: List of Tool objects ready for agent execution
        context: Full template rendering context
        combined_instructions: Combined default + agent instructions
        prefetch_results: Results from prefetch tool execution
        attachments: List of Attachment objects for multi-modal inputs
        skills: List of Skill objects for loaded skills
    """

    agent: Agent
    agent_config: AgentConfig
    system_message: str
    user_message: str
    rendered_prompt: str
    tools: List[Tool]
    context: Dict[str, Any]
    combined_instructions: str
    prefetch_results: Dict[str, Any]
    attachments: List[Attachment]
    original_prompt: str = ""
    skills: List[Skill] = field(default_factory=list)
    # Map of sticky skill name -> turns remaining before auto-unload. Only
    # populated on the turn a skill is about to expire (turns_remaining <= 0).
    # The agent surfaces these as <skill_expiring> blocks in the context turn.
    expiring_skills: Dict[str, int] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str | None = None


class AgentPreparer:
    """Prepares agents for execution or rendering.

    This class consolidates all agent preparation logic that was previously
    duplicated across render command, run_agent, and _execute_agent_with_prompt.

    The preparation pipeline:
    1. Parse agent file (with inheritance resolution)
    2. Execute prefetch tools
    3. Execute tool directives (optional)
    4. Build template context
    5. Render template
    6. Build instructions
    7. Expand and create tools
    8. Build system prompt

    This ensures that render shows EXACTLY what run executes.
    """

    def _extract_tool_directive_placeholders(self, content: str) -> Dict[str, str]:
        """Extract variable names from tool directives and return placeholders.

        When rendering without executing directives, we still need variables to
        be defined so the template doesn't fail. This extracts all assign="var"
        names and creates placeholder values.

        Args:
            content: Markdown content with tool directives

        Returns:
            Dict mapping variable names to placeholder values
        """
        from tsugite.md_agents import extract_tool_directives

        try:
            directives = extract_tool_directives(content)
        except Exception:
            # If extraction fails, return empty dict
            return {}

        placeholders = {}
        for directive in directives:
            if directive.assign_var:
                # Create a descriptive placeholder showing what would be executed
                placeholders[directive.assign_var] = (
                    f"[Tool directive: {directive.name}(...) - not executed in render mode]"
                )

        return placeholders

    def prepare(
        self,
        agent: Agent,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        workspace: Optional["Workspace"] = None,
        skip_tool_directives: bool = False,
        attachments: Optional[List[Attachment]] = None,
        event_bus: Optional[Any] = None,
        path_context: Optional[Any] = None,
    ) -> PreparedAgent:
        """Prepare agent with all context, tools, and instructions.

        Args:
            agent: Parsed agent object
            prompt: User prompt/task
            context: Additional context variables
            workspace: Optional workspace for context files and persistent sessions
            skip_tool_directives: Skip executing tool directives (for render)
            attachments: List of Attachment objects for multi-modal inputs
            event_bus: Optional event bus for emitting skill load events
            path_context: Optional PathContext with invoked_from, workspace_dir, effective_cwd

        Returns:
            PreparedAgent ready for execution or display

        Raises:
            RuntimeError: If preparation fails
        """
        from tsugite.agent_runner import (
            _combine_instructions,
            execute_prefetch,
            execute_tool_directives,
            get_default_instructions,
        )
        from tsugite.core.agent import build_system_prompt
        from tsugite.core.tools import create_tool_from_tsugite
        from tsugite.renderer import AgentRenderer
        from tsugite.tools import expand_tool_specs
        from tsugite.utils import is_interactive

        if context is None:
            context = {}

        agent_config = agent.config

        # Step 0: Resolve workspace files using convention-based discovery
        workspace_attachments: List[Attachment] = []
        if workspace:
            from tsugite.workspace.context import build_workspace_attachments

            workspace_attachments = build_workspace_attachments(workspace)

        # Merge workspace attachments with explicit attachments (explicit first)
        all_attachments = (attachments or []) + workspace_attachments

        # Step 0b: Load agent-config attachments (Jinja-rendered paths)
        # Support "-filename" prefix to remove a workspace default
        workspace_path = workspace.path if workspace else None
        removals = {t.lstrip("-") for t in (agent_config.attachments or []) if t.startswith("-")}
        keep_templates = [t for t in (agent_config.attachments or []) if not t.startswith("-")]
        if removals:
            all_attachments = [a for a in all_attachments if a.name not in removals]
        all_attachments.extend(resolve_agent_config_attachments(keep_templates, workspace_path))

        # Deduplicate by name (keep first occurrence)
        seen_names: set[str] = set()
        deduped: List[Attachment] = []
        for att in all_attachments:
            if att.name not in seen_names:
                seen_names.add(att.name)
                deduped.append(att)
        all_attachments = deduped

        # Step 0c: Set up workspace-aware skill manager. Fall back to
        # path_context.workspace_dir when no Workspace was passed so daemon/chat
        # callers still get workspace skills.
        from tsugite.config import load_config as _load_config
        from tsugite.tools.skills import SkillManager, set_skill_manager
        from tsugite.workspace.models import Workspace

        effective_workspace = workspace or Workspace.try_load(path_context.workspace_dir if path_context else None)

        _config = _load_config()
        extra_skill_paths = (agent_config.skill_paths or []) + (_config.skill_paths or [])
        _skill_manager = SkillManager(workspace=effective_workspace, extra_paths=extra_skill_paths or None)
        set_skill_manager(_skill_manager)

        # Step 1: Execute prefetch tools
        prefetch_context = {}
        if agent_config.prefetch:
            try:
                prefetch_context = execute_prefetch(agent_config.prefetch)
            except Exception:
                # Silently continue if prefetch fails
                prefetch_context = {}

        # Step 2: Execute tool directives (unless skip_tool_directives=True for render)
        if skip_tool_directives:
            modified_content = agent.content
            # Extract tool directive variable names and provide placeholders
            tool_context = self._extract_tool_directive_placeholders(agent.content)
        else:
            modified_content, tool_context = execute_tool_directives(agent.content, prefetch_context)

        # Step 3: Build template context
        interactive_mode = is_interactive()

        # Extract path context values if available
        # Use effective_cwd from path_context (for daemon) or fall back to actual cwd
        if path_context and path_context.effective_cwd:
            cwd = str(path_context.effective_cwd)
        else:
            cwd = str(Path.cwd())
        invoked_from = str(path_context.invoked_from) if path_context else None
        workspace_dir = str(path_context.workspace_dir) if path_context and path_context.workspace_dir else None

        full_context = {
            **context,
            **prefetch_context,
            **tool_context,
            "user_prompt": prompt,
            "agent_name": agent_config.name,
            "is_interactive": interactive_mode,
            "is_daemon": context.get("is_daemon", False),
            "is_scheduled": context.get("is_scheduled", False),
            "schedule_id": context.get("schedule_id", ""),
            "has_notify_tool": context.get("has_notify_tool", False),
            "running_tasks": context.get("running_tasks", []),
            "tsugite_url": context.get("tsugite_url", ""),
            "tsugite_token": context.get("tsugite_token", ""),
            "tools": agent_config.tools,
            # Subagent context
            "is_subagent": context.get("is_subagent", False),
            "parent_agent": context.get("parent_agent", None),
            # Chat history (for chat agents)
            "chat_history": context.get("chat_history", []),
            # Path context for workspace-aware agents
            "CWD": cwd,
            "INVOKED_FROM": invoked_from,
            "WORKSPACE_DIR": workspace_dir,
        }

        renderer = AgentRenderer()

        # Step 3b: Evaluate run_if guard (skip agent if expression is falsy)
        if agent_config.run_if:
            skip_reason = None
            try:
                guard_result = renderer.render("{{ " + agent_config.run_if + " }}", full_context)
                if guard_result.strip().lower() in ("", "false", "none", "0"):
                    skip_reason = f"run_if guard '{agent_config.run_if}' evaluated to false"
            except Exception as e:
                skip_reason = f"run_if guard error: {e}"
            if skip_reason:
                return PreparedAgent(
                    agent=agent,
                    agent_config=agent_config,
                    system_message="",
                    user_message="",
                    rendered_prompt="",
                    tools=[],
                    context=full_context,
                    combined_instructions="",
                    prefetch_results=prefetch_context,
                    attachments=all_attachments,
                    skipped=True,
                    skip_reason=skip_reason,
                )

        # Step 4: Render template
        try:
            rendered_prompt = renderer.render(modified_content, full_context)
        except Exception as e:
            raise RuntimeError(f"Template rendering failed: {e}") from e

        # Step 5: Build instructions
        base_instructions = get_default_instructions()
        agent_instructions = getattr(agent_config, "instructions", "")

        # Render agent instructions as Jinja2 template
        if agent_instructions:
            try:
                agent_instructions = renderer.render(agent_instructions, full_context)
            except Exception as e:
                raise RuntimeError(f"Failed to render agent instructions: {e}") from e

        combined_instructions = _combine_instructions(base_instructions, agent_instructions)

        # Step 6: Expand and create tools
        try:
            # Expand tool specifications (categories, globs, regular names)
            expanded_tools = expand_tool_specs(agent_config.tools) if agent_config.tools else []

            # Auto-inject or filter interactive tools based on interaction capability.
            from tsugite.interaction import get_interaction_backend
            from tsugite.tools import _tools

            interactive_tool_names = ["ask_user", "ask_user_batch"]
            has_interaction = (
                interactive_mode or get_interaction_backend() is not None or full_context.get("is_daemon", False)
            )
            if has_interaction:
                for name in interactive_tool_names:
                    if name not in expanded_tools and name in _tools:
                        expanded_tools.append(name)
            else:
                expanded_tools = [t for t in expanded_tools if t not in interactive_tool_names]

            # Auto-inject notify_user when has_notify_tool is set (scheduled tasks)
            if full_context.get("has_notify_tool", False):
                if "notify_user" not in expanded_tools and "notify_user" in _tools:
                    expanded_tools.append("notify_user")

            # Convert to Tool objects
            tools = [create_tool_from_tsugite(name) for name in expanded_tools]
        except Exception as e:
            raise RuntimeError(f"Failed to create tools: {e}") from e

        # Step 7: Load auto_load_skills, sticky skills, and trigger-matched skills
        from tsugite.events.events import SkillLoadFailedEvent

        # Skills the user explicitly removed this session (populated by the daemon).
        suppressed_skills = set(full_context.get("suppressed_skills") or [])

        # Sticky skills carried over from prior turns on this session (daemon-only).
        # Shape: {skill_name: turns_unused_counter}
        sticky_counters: Dict[str, int] = dict(full_context.get("sticky_skills") or {})
        ttl_default = int(full_context.get("skill_ttl_default") or 10)

        # Step 7a: auto_load_skills (exempt from TTL, re-loaded every turn from frontmatter)
        auto_load_skills = [s for s in (agent_config.auto_load_skills or []) if s not in suppressed_skills]

        for skill_name in auto_load_skills:
            result = _skill_manager.load_skill(skill_name)
            if result.startswith("Failed") or result.startswith("Skill '"):
                if event_bus:
                    event_bus.emit(SkillLoadFailedEvent(skill_name=skill_name, error_message=result))

        # Step 7b: sticky skills — re-load whatever carried over from prior turns,
        # unless the counter already exceeded the skill's effective TTL (expired).
        _skill_manager._ensure_registry_initialized()
        registry = _skill_manager._skill_registry
        expiring_skills: Dict[str, int] = {}
        expired_sticky: List[str] = []
        for name, counter in sticky_counters.items():
            if name in suppressed_skills:
                expired_sticky.append(name)
                continue
            meta = registry.get(name)
            if meta is None:
                # Skill vanished (renamed/removed) between turns — drop it.
                expired_sticky.append(name)
                continue
            effective_ttl = meta.ttl if meta.ttl is not None else ttl_default
            if effective_ttl > 0 and counter > effective_ttl:
                expired_sticky.append(name)
                continue
            if name in auto_load_skills:
                # Already loaded above; no need to double-load but it's still sticky.
                continue
            result = _skill_manager.load_skill(name)
            if result.startswith("Failed") or result.startswith("Skill '"):
                if event_bus:
                    event_bus.emit(SkillLoadFailedEvent(skill_name=name, error_message=result))
                continue
            if effective_ttl > 0:
                remaining = effective_ttl - counter
                if remaining <= 1:
                    expiring_skills[name] = max(remaining, 0)

        # Step 7c: trigger-matched skills (new stickies start here when daemon adds them).
        triggered_skill_names = [
            name for name in _skill_manager.get_triggered_skills(prompt) if name not in suppressed_skills
        ]
        for skill_name in triggered_skill_names:
            logger.info(f"Trigger-loading skill '{skill_name}' based on user prompt")
            result = _skill_manager.load_skill(skill_name)
            if result.startswith("Failed") or result.startswith("Skill '"):
                if event_bus:
                    event_bus.emit(SkillLoadFailedEvent(skill_name=skill_name, error_message=result))

        # Stash expired/triggered names so the daemon can update its sticky state post-turn.
        full_context["_expired_sticky_skills"] = expired_sticky
        full_context["_triggered_skill_names"] = list(triggered_skill_names)
        full_context["_auto_loaded_skill_names"] = list(auto_load_skills)

        # Get all successfully loaded skills as Skill objects
        loaded_skills_dict = _skill_manager.get_loaded_skills()
        skills = [Skill(name=name, content=content) for name, content in loaded_skills_dict.items()]

        # Step 8: Build system message (what LLM actually sees)
        system_message = build_system_prompt(tools, combined_instructions)

        # Add environment context when invoked_from differs from CWD
        if invoked_from and invoked_from != cwd:
            env_block = f"""
## Environment

Working directory: {cwd}
Invoked from: {invoked_from}

When the user refers to "this folder", "current directory", or "here",
they typically mean the invoked location ({invoked_from}).
"""
            system_message = system_message + env_block

        # User message is the rendered prompt
        user_message = rendered_prompt

        return PreparedAgent(
            agent=agent,
            agent_config=agent_config,
            system_message=system_message,
            user_message=user_message,
            rendered_prompt=rendered_prompt,
            original_prompt=prompt,
            tools=tools,
            context=full_context,
            combined_instructions=combined_instructions,
            prefetch_results=prefetch_context,
            attachments=all_attachments,
            skills=skills,
            expiring_skills=expiring_skills,
        )
