"""Agent preparation pipeline - unified logic for render and execution."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tsugite.core.tools import Tool
from tsugite.md_agents import Agent, AgentConfig


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
        tools: List of Tool objects ready for agent execution
        context: Full template rendering context
        combined_instructions: Combined default + agent instructions
        prefetch_results: Results from prefetch tool execution
        attachments: List of (name, content) tuples for prompt caching
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
    attachments: List[tuple[str, str]]


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
        delegation_agents: Optional[List[tuple[str, Path]]] = None,
        skip_tool_directives: bool = False,
        task_summary: str = "## Current Tasks\nNo tasks yet.",
        tasks: Optional[List[Dict[str, Any]]] = None,
        attachments: Optional[List[tuple[str, str]]] = None,
    ) -> PreparedAgent:
        """Prepare agent with all context, tools, and instructions.

        Args:
            agent: Parsed agent object
            prompt: User prompt/task
            context: Additional context variables
            delegation_agents: List of (name, path) tuples for delegation
            skip_tool_directives: Skip executing tool directives (for render)
            task_summary: Current task summary (from task manager)
            tasks: List of task dicts for template iteration (from task manager)
            attachments: List of (name, content) tuples for prompt caching

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

        # Step 1: Execute prefetch tools
        prefetch_context = {}
        if agent_config.prefetch:
            try:
                from tsugite.agent_runner import execute_prefetch

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
        full_context = {
            **context,
            **prefetch_context,
            **tool_context,
            "user_prompt": prompt,
            "task_summary": task_summary,
            "tasks": tasks or [],
            "is_interactive": interactive_mode,
            "text_mode": agent_config.text_mode,
            "tools": agent_config.tools,
            # Subagent context
            "is_subagent": context.get("is_subagent", False),
            "parent_agent": context.get("parent_agent", None),
            # Chat history (for chat agents)
            "chat_history": context.get("chat_history", []),
        }

        # Step 4: Render template
        renderer = AgentRenderer()
        try:
            rendered_prompt = renderer.render(modified_content, full_context)
        except Exception as e:
            raise RuntimeError(f"Template rendering failed: {e}") from e

        # Step 5: Build instructions
        base_instructions = get_default_instructions(text_mode=agent_config.text_mode)
        agent_instructions = getattr(agent_config, "instructions", "")

        # Render agent instructions as Jinja2 template (they may contain {% if text_mode %}, etc.)
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

            # Add task management tools
            task_tools = ["task_add", "task_update", "task_complete", "task_list", "task_get"]
            all_tool_names = expanded_tools + task_tools

            if delegation_agents:
                all_tool_names.append("spawn_agent")

            # Filter out interactive tools in non-interactive mode
            if not interactive_mode and "ask_user" in all_tool_names:
                all_tool_names.remove("ask_user")

            # Convert to Tool objects
            tools = [create_tool_from_tsugite(name) for name in all_tool_names]
        except Exception as e:
            raise RuntimeError(f"Failed to create tools: {e}") from e

        # Step 7: Build system message (what LLM actually sees)
        system_message = build_system_prompt(tools, combined_instructions, agent_config.text_mode)

        # User message is the rendered prompt
        user_message = rendered_prompt

        return PreparedAgent(
            agent=agent,
            agent_config=agent_config,
            system_message=system_message,
            user_message=user_message,
            rendered_prompt=rendered_prompt,
            tools=tools,
            context=full_context,
            combined_instructions=combined_instructions,
            prefetch_results=prefetch_context,
            attachments=attachments or [],
        )
