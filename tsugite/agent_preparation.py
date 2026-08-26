"""Agent preparation pipeline - unified logic for render and execution."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tsugite.cli.helpers import PathContext
    from tsugite.tools.skills import SkillManager

from tsugite.attachments.agent_config import (  # noqa: E402
    resolve_agent_config_attachments,
    split_attachment_removals,
)
from tsugite.attachments.base import Attachment  # noqa: E402
from tsugite.core.tools import Tool  # noqa: E402
from tsugite.md_agents import Agent, AgentConfig  # noqa: E402
from tsugite.skill_discovery import Skill  # noqa: E402


@dataclass(frozen=True)
class SkillLoad:
    """What one turn's skill resolution produced.

    `expired`/`triggered`/`auto_loaded` are read back by the daemon to update its
    sticky state after the turn, so they are real outputs rather than bookkeeping;
    `prepare()` publishes them onto the template context explicitly.
    """

    skills: List[Skill]
    expiring: Dict[str, int]
    expired: List[str]
    triggered: List[str]
    auto_loaded: List[str]


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

    `prepare()` orchestrates, one named step per phase:

        _resolve_attachments     caller + frontmatter attachments, removals, dedupe
        _install_skill_manager   workspace-aware SkillManager, made active
        _run_prefetch            frontmatter prefetch tools + agent-list injection
        _resolve_paths           cwd / invoked_from / workspace_dir
        _build_template_context  the Jinja context and its framework defaults
        _expand_tools            tool specs + capability-based auto-injection
        _load_skills             auto-load, sticky TTL, trigger matching

    Both `render` and `run` go through here, so render shows EXACTLY what run
    executes.
    """

    def _build_directive_placeholders(
        self,
        content: str,
        extractor: Callable[[str], List[Any]],
        kind: str,
        rewrite_to: Optional[Callable[[Any], str]] = None,
    ) -> Tuple[str, Dict[str, str]]:
        """Build {assign_var: placeholder} for directives that won't actually run.

        Used by render-mode shortcuts so the template still has all its variables
        defined. When `rewrite_to` is given, the directive substring is also
        replaced in `content` with the callable's per-directive output.
        """
        try:
            directives = extractor(content)
        except Exception:
            return content, {}

        placeholders: Dict[str, str] = {}
        modified = content
        for d in directives:
            if d.assign_var:
                placeholders[d.assign_var] = f"[{kind} directive: {d.name}(...) - not executed in render mode]"
            if rewrite_to is not None:
                modified = modified.replace(d.raw_match, rewrite_to(d))
        return modified, placeholders

    def _extract_tool_directive_placeholders(self, content: str) -> Dict[str, str]:
        from tsugite.md_agents import extract_tool_directives

        _, placeholders = self._build_directive_placeholders(content, extract_tool_directives, "Tool")
        return placeholders

    def _extract_exec_directive_placeholders(self, content: str) -> Tuple[str, Dict[str, str]]:
        from tsugite.md_agents import extract_exec_directives

        return self._build_directive_placeholders(
            content,
            extract_exec_directives,
            "Exec",
            rewrite_to=lambda d: f"<!-- Exec '{d.name}' skipped (--no-exec) -->",
        )

    def _resolve_attachments(
        self, agent_config: AgentConfig, attachments: Optional[List[Attachment]]
    ) -> Tuple[List[Attachment], Dict[str, Any]]:
        """Merge caller attachments with the agent's own, honoring removals.

        Front-matter attachments carry the cache tiers and are the intended
        source, so they dedupe ahead of any same-named attachment the caller
        passed in. Legacy `-filename` string entries drop a same-named entry.
        Front-matter paths resolve via cwd (workspace_path is None in production).
        """
        all_attachments = list(attachments or [])

        removals, keep_items = split_attachment_removals(agent_config.attachments or [])
        if removals:
            all_attachments = [a for a in all_attachments if a.name not in removals]

        loaded, bindings = resolve_agent_config_attachments(keep_items, None)
        all_attachments = loaded + all_attachments

        seen_names: set[str] = set()
        deduped: List[Attachment] = []
        for att in all_attachments:
            if att.name not in seen_names:
                seen_names.add(att.name)
                deduped.append(att)
        return deduped, bindings

    def _install_skill_manager(
        self, agent_config: AgentConfig, path_context: Optional["PathContext"]
    ) -> "SkillManager":
        """Build the workspace-aware skill manager and make it the active one.

        The workspace comes from path_context.workspace_dir so daemon and chat
        callers still get workspace skills.
        """
        from tsugite.config import load_config
        from tsugite.tools.skills import SkillManager, set_skill_manager
        from tsugite.workspace.models import Workspace

        workspace = Workspace.try_load(path_context.workspace_dir if path_context else None)
        extra_paths = (agent_config.skill_paths or []) + (load_config().skill_paths or [])
        manager = SkillManager(workspace=workspace, extra_paths=extra_paths or None)
        set_skill_manager(manager)
        return manager

    def _run_prefetch(self, agent_config: AgentConfig) -> Dict[str, Any]:
        """Run frontmatter prefetch tools, plus the opt-in agent-list injection.

        A failing prefetch yields an empty context rather than failing the run:
        prefetch is supplementary context, not a precondition.
        """
        from tsugite.agent_runner import execute_prefetch

        prefetch_context: Dict[str, Any] = {}
        if agent_config.prefetch:
            try:
                prefetch_context = execute_prefetch(agent_config.prefetch)
            except Exception:
                prefetch_context = {}

        # Default agents call list_available_agents() on demand instead of
        # carrying the full list, so this stays opt-in.
        if "available_agents" not in prefetch_context and (
            agent_config.auto_load_agent_list or agent_config.auto_load_agents
        ):
            from tsugite.tools.agents import discover_agents, format_agents_markdown

            agents = discover_agents()
            if agent_config.auto_load_agents:
                wanted = set(agent_config.auto_load_agents)
                agents = [a for a in agents if a["name"] in wanted]
            prefetch_context["available_agents"] = format_agents_markdown(agents)

        return prefetch_context

    def _load_skills(
        self,
        agent_config: AgentConfig,
        prompt: str,
        full_context: Dict[str, Any],
        skill_manager: "SkillManager",
        event_bus: Optional[Any],
    ) -> SkillLoad:
        """Load auto, sticky and trigger-matched skills for this turn."""
        from tsugite.events.events import SkillLoadFailedEvent

        # Skills the user explicitly removed this session (populated by the daemon).
        suppressed_skills = set(full_context.get("suppressed_skills") or [])

        # Sticky skills carried over from prior turns on this session (daemon-only).
        # Shape: {skill_name: turns_unused_counter}
        sticky_counters: Dict[str, int] = dict(full_context.get("sticky_skills") or {})
        ttl_default = int(full_context.get("skill_ttl_default") or 10)

        auto_load_skills = [s for s in (agent_config.auto_load_skills or []) if s not in suppressed_skills]

        for skill_name in auto_load_skills:
            result = skill_manager.load_skill(skill_name)
            if result.startswith("Failed") or result.startswith("Skill '"):
                if event_bus:
                    event_bus.emit(SkillLoadFailedEvent(skill_name=skill_name, error_message=result))

        # unless the counter already exceeded the skill's effective TTL (expired).
        skill_manager._ensure_registry_initialized()
        registry = skill_manager._skill_registry
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
            result = skill_manager.load_skill(name)
            if result.startswith("Failed") or result.startswith("Skill '"):
                if event_bus:
                    event_bus.emit(SkillLoadFailedEvent(skill_name=name, error_message=result))
                continue
            if effective_ttl > 0:
                remaining = effective_ttl - counter
                if remaining <= 1:
                    expiring_skills[name] = max(remaining, 0)

        triggered_skill_names = [
            name for name in skill_manager.get_triggered_skills(prompt) if name not in suppressed_skills
        ]
        for skill_name in triggered_skill_names:
            logger.info(f"Trigger-loading skill '{skill_name}' based on user prompt")
            result = skill_manager.load_skill(skill_name)
            if result.startswith("Failed") or result.startswith("Skill '"):
                if event_bus:
                    event_bus.emit(SkillLoadFailedEvent(skill_name=skill_name, error_message=result))

        loaded_skills_dict = skill_manager.get_loaded_skills()
        return SkillLoad(
            skills=[Skill(name=name, content=content) for name, content in loaded_skills_dict.items()],
            expiring=expiring_skills,
            expired=expired_sticky,
            triggered=list(triggered_skill_names),
            auto_loaded=list(auto_load_skills),
        )

    def _expand_tools(
        self, agent_config: AgentConfig, full_context: Dict[str, Any], interactive_mode: bool
    ) -> List[Tool]:
        """Expand the agent's tool specs and apply the capability-based auto-injections.

        Interactive tools are added only when something can actually answer, and
        stripped otherwise, so a scheduled run cannot offer the model a prompt
        nobody will see.
        """
        from tsugite.core.tools import create_tool_from_tsugite
        from tsugite.interaction import get_interaction_backend
        from tsugite.tools import _tools, expand_tool_specs

        try:
            expanded = (
                expand_tool_specs(agent_config.tools, strict=agent_config.strict_tools) if agent_config.tools else []
            )

            interactive_tool_names = ["ask_user", "ask_user_batch"]
            has_interaction = (
                interactive_mode or get_interaction_backend() is not None or full_context.get("is_daemon", False)
            )
            if has_interaction:
                for name in interactive_tool_names:
                    if name not in expanded and name in _tools:
                        expanded.append(name)
            else:
                expanded = [t for t in expanded if t not in interactive_tool_names]

            # Scheduled tasks get a way to reach the user even if the agent didn't ask.
            if full_context.get("has_notify_tool", False):
                if "notify_user" not in expanded and "notify_user" in _tools:
                    expanded.append("notify_user")

            return [create_tool_from_tsugite(name) for name in expanded]
        except Exception as e:
            raise RuntimeError(f"Failed to create tools: {e}") from e

    @staticmethod
    def _resolve_paths(path_context: Optional["PathContext"]) -> Tuple[str, Optional[str], Optional[str]]:
        """Return (cwd, invoked_from, workspace_dir) as strings for the template context.

        The daemon supplies an effective_cwd that differs from the process cwd.
        """
        if path_context and path_context.effective_cwd:
            cwd = str(path_context.effective_cwd)
        else:
            cwd = str(Path.cwd())
        invoked_from = str(path_context.invoked_from) if path_context else None
        workspace_dir = str(path_context.workspace_dir) if path_context and path_context.workspace_dir else None
        return cwd, invoked_from, workspace_dir

    def _build_template_context(
        self,
        *,
        agent_config: AgentConfig,
        prompt: str,
        context: Dict[str, Any],
        directive_context: Dict[str, Any],
        attachment_bindings: Dict[str, Any],
        interactive_mode: bool,
        cwd: str,
        invoked_from: Optional[str],
        workspace_dir: Optional[str],
    ) -> Dict[str, Any]:
        """Assemble the Jinja context every agent template renders against.

        Framework flags default here rather than in the template so a caller that
        supplies none of them still renders under StrictUndefined.
        """
        from tsugite.tools import list_tools

        full_context = {
            **context,
            **directive_context,
            "user_prompt": prompt,
            "agent_name": agent_config.name,
            "is_interactive": interactive_mode,
            "is_daemon": context.get("is_daemon", False),
            "is_scheduled": context.get("is_scheduled", False),
            "schedule_id": context.get("schedule_id", ""),
            "conversation_id": context.get("conversation_id", ""),
            "has_notify_tool": context.get("has_notify_tool", False),
            "running_tasks": context.get("running_tasks", []),
            "tsugite_url": context.get("tsugite_url", ""),
            "tsugite_token": context.get("tsugite_token", ""),
            "tools": agent_config.tools,
            # Tool names actually installed, so templates can conditionally mention optional
            # tools, e.g. `{% if 'web_search' in available_tools %}`.
            "available_tools": list_tools(),
            "is_subagent": context.get("is_subagent", False),
            "parent_agent": context.get("parent_agent", None),
            "chat_history": context.get("chat_history", []),
            "CWD": cwd,
            "INVOKED_FROM": invoked_from,
            "WORKSPACE_DIR": workspace_dir,
        }

        # User-specified attachment `assign:` bindings win over built-in or
        # prefetch values; warn so collisions surface rather than confuse.
        for name, value in attachment_bindings.items():
            if name in full_context:
                logger.warning("Attachment binding %r overrides existing context variable", name)
            full_context[name] = value

        return full_context

    def prepare(
        self,
        agent: Agent,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        skip_tool_directives: bool = False,
        skip_exec_directives: bool = False,
        attachments: Optional[List[Attachment]] = None,
        event_bus: Optional[Any] = None,
        path_context: Optional["PathContext"] = None,
    ) -> PreparedAgent:
        """Prepare agent with all context, tools, and instructions.

        Args:
            agent: Parsed agent object
            prompt: User prompt/task
            context: Additional context variables
            skip_tool_directives: Skip executing tool directives (for render)
            skip_exec_directives: Skip executing exec directives (for render --no-exec)
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
            execute_exec_directives,
            execute_tool_directives,
            get_default_instructions,
        )
        from tsugite.core.agent import build_system_prompt
        from tsugite.renderer import AgentRenderer
        from tsugite.utils import is_interactive

        if context is None:
            context = {}

        agent_config = agent.config

        all_attachments, attachment_bindings = self._resolve_attachments(agent_config, attachments)

        _skill_manager = self._install_skill_manager(agent_config, path_context)

        prefetch_context = self._run_prefetch(agent_config)

        if skip_tool_directives:
            modified_content = agent.content
            # Extract tool directive variable names and provide placeholders
            tool_context = self._extract_tool_directive_placeholders(agent.content)
        else:
            modified_content, tool_context = execute_tool_directives(agent.content, prefetch_context)

        # reflects what the LLM would see; `--no-exec` opts out for side-effecty blocks.
        if skip_exec_directives:
            modified_content, exec_context = self._extract_exec_directive_placeholders(modified_content)
        else:
            exec_locals: Dict[str, Any] = {**context, **prefetch_context, **tool_context}
            modified_content, exec_context = execute_exec_directives(
                modified_content,
                existing_context=exec_locals,
                event_bus=event_bus,
            )

        interactive_mode = is_interactive()
        cwd, invoked_from, workspace_dir = self._resolve_paths(path_context)
        full_context = self._build_template_context(
            agent_config=agent_config,
            prompt=prompt,
            context=context,
            directive_context={**prefetch_context, **tool_context, **exec_context},
            attachment_bindings=attachment_bindings,
            interactive_mode=interactive_mode,
            cwd=cwd,
            invoked_from=invoked_from,
            workspace_dir=workspace_dir,
        )

        renderer = AgentRenderer()

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
                    attachments=all_attachments,
                    skipped=True,
                    skip_reason=skip_reason,
                )

        try:
            rendered_prompt = renderer.render(modified_content, full_context)
        except Exception as e:
            raise RuntimeError(f"Template rendering failed: {e}") from e

        base_instructions = get_default_instructions()
        agent_instructions = getattr(agent_config, "instructions", "")

        # Render agent instructions as Jinja2 template
        if agent_instructions:
            try:
                agent_instructions = renderer.render(agent_instructions, full_context)
            except Exception as e:
                raise RuntimeError(f"Failed to render agent instructions: {e}") from e

        combined_instructions = _combine_instructions(base_instructions, agent_instructions)

        tools = self._expand_tools(agent_config, full_context, interactive_mode)

        skill_load = self._load_skills(agent_config, prompt, full_context, _skill_manager, event_bus)
        skills, expiring_skills = skill_load.skills, skill_load.expiring
        # The daemon reads these back off the context to update sticky state.
        full_context["_expired_sticky_skills"] = skill_load.expired
        full_context["_triggered_skill_names"] = skill_load.triggered
        full_context["_auto_loaded_skill_names"] = skill_load.auto_loaded

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
            attachments=all_attachments,
            skills=skills,
            expiring_skills=expiring_skills,
        )
