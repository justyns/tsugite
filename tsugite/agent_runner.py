"""Agent execution engine using smolagents."""

from pathlib import Path
from typing import Dict, Any, Optional, List
from smolagents import CodeAgent

from tsugite.md_agents import parse_agent, AgentConfig, validate_agent_execution
from tsugite.renderer import AgentRenderer
from tsugite.tool_adapter import get_smolagents_tools
from tsugite.models import get_model
from tsugite.tools import call_tool
from tsugite.tools.tasks import reset_task_manager, get_task_manager


TSUGITE_DEFAULT_INSTRUCTIONS = (
    "You are operating inside the Tsugite micro-agent runtime. Follow the rendered task faithfully, use the available "
    "tools when they meaningfully advance the work, and maintain a living plan via the task_* tools. Create or update "
    "tasks whenever you define new sub-work, mark progress as you go, and rely on the task summary to decide the next "
    "action. Provide a clear, actionable final response without unnecessary filler."
)


def _combine_instructions(*segments: str) -> str:
    """Join instruction segments, skipping empties."""

    parts = [segment.strip() for segment in segments if segment and segment.strip()]
    return "\n\n".join(parts)


def execute_prefetch(prefetch_config: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Execute prefetch tools and return context.

    Args:
        prefetch_config: List of prefetch tool configurations

    Returns:
        Dictionary mapping assign names to tool results
    """
    context = {}

    for config in prefetch_config:
        tool_name = config.get("tool")
        args = config.get("args", {})
        assign_name = config.get("assign")

        if not tool_name or not assign_name:
            continue

        try:
            result = call_tool(tool_name, **args)
            context[assign_name] = result
        except Exception as e:
            print(f"Warning: Prefetch tool '{tool_name}' failed: {e}")
            context[assign_name] = None

    return context


def run_agent(
    agent_path: Path,
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
    debug: bool = False,
    custom_logger: Optional[Any] = None,
    trust_mcp_code: bool = False,
) -> str:
    """Run a Tsugite agent using smolagents.

    Args:
        agent_path: Path to agent markdown file
        prompt: User prompt/task for the agent
        context: Additional context variables
        model_override: Override agent's default model
        debug: Enable debug output (rendered prompt)
        custom_logger: Custom logger for agent output
        trust_mcp_code: Whether to trust remote code from MCP servers

    Returns:
        Agent execution result as string

    Raises:
        ValueError: If agent file is invalid
        RuntimeError: If agent execution fails
    """
    if context is None:
        context = {}

    # Initialize task manager for this agent session
    reset_task_manager()
    task_manager = get_task_manager()

    # Parse agent configuration
    try:
        agent_text = agent_path.read_text()
        agent = parse_agent(agent_text, agent_path)
        agent_config = agent.config
    except Exception as e:
        raise ValueError(f"Failed to parse agent file: {e}")

    combined_instructions = _combine_instructions(
        TSUGITE_DEFAULT_INSTRUCTIONS,
        getattr(agent_config, "instructions", ""),
    )

    # Execute prefetch tools if any
    prefetch_context = {}
    if agent_config.prefetch:
        try:
            prefetch_context = execute_prefetch(agent_config.prefetch)
        except Exception as e:
            print(f"Warning: Prefetch execution failed: {e}")

    # Add task context (will be updated as agent creates tasks)
    task_context = task_manager.get_task_summary()

    # Prepare full context for template rendering
    full_context = {
        **context,
        **prefetch_context,
        "user_prompt": prompt,
        "task_summary": task_context,
    }

    # Render agent template
    renderer = AgentRenderer()
    try:
        rendered_prompt = renderer.render(agent.content, full_context)

        if debug:
            print("\n" + "=" * 60)
            print("DEBUG: Rendered Prompt")
            print("=" * 60)
            print(rendered_prompt)
            print("=" * 60 + "\n")

    except Exception as e:
        raise ValueError(f"Template rendering failed: {e}")

    # Create smolagents tools (automatically include task tracking tools)
    try:
        # Always include task tracking tools for agents
        task_tools = ["task_add", "task_update", "task_complete", "task_list", "task_get"]
        all_tools = list(agent_config.tools) + task_tools
        tools = get_smolagents_tools(all_tools)
    except Exception as e:
        raise RuntimeError(f"Failed to create tools: {e}")

    # Load MCP tools if configured
    if agent_config.mcp_servers:
        try:
            from tsugite.mcp_config import load_mcp_config
            from tsugite.mcp_integration import load_all_mcp_tools

            global_mcp_config = load_mcp_config()
            mcp_tools = load_all_mcp_tools(agent_config.mcp_servers, global_mcp_config, trust_mcp_code)
            tools.extend(mcp_tools)
        except Exception as e:
            print(f"Warning: Failed to load MCP tools: {e}")
            print("Continuing without MCP tools.")

    # Create model
    model_string = model_override or agent_config.model
    try:
        model = get_model(model_string)
    except Exception as e:
        raise RuntimeError(f"Failed to create model '{model_string}': {e}")

    # Create and run smolagents agent with task tracking awareness
    try:
        # Build agent kwargs
        agent_kwargs = {
            "tools": tools,
            "model": model,
            "max_steps": agent_config.max_steps,
            "instructions": combined_instructions or None,
        }

        # Add custom logger if provided
        if custom_logger is not None:
            agent_kwargs["logger"] = custom_logger
            agent_kwargs["verbosity_level"] = -1  # Suppress default output

        agent = CodeAgent(**agent_kwargs)

        result = agent.run(rendered_prompt)
        return str(result)

    except Exception as e:
        raise RuntimeError(f"Agent execution failed: {e}")


def validate_agent_file(agent_path: Path) -> tuple[bool, str]:
    """Validate that an agent file can be executed.

    Args:
        agent_path: Path to agent markdown file

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Parse agent
        agent_text = agent_path.read_text()
        agent = parse_agent(agent_text, agent_path)

        # Use centralized validation
        return validate_agent_execution(agent)

    except Exception as e:
        return False, f"Agent file validation failed: {e}"


def get_agent_info(agent_path: Path) -> Dict[str, Any]:
    """Get information about an agent without executing it.

    Args:
        agent_path: Path to agent markdown file

    Returns:
        Dictionary with agent information
    """
    try:
        agent_text = agent_path.read_text()
        agent = parse_agent(agent_text, agent_path)
        agent_config = agent.config

        return {
            "name": agent_config.name,
            "description": getattr(agent_config, "description", "No description"),
            "model": agent_config.model,
            "max_steps": agent_config.max_steps,
            "tools": agent_config.tools,
            "prefetch_count": (len(agent_config.prefetch) if agent_config.prefetch else 0),
            "permissions_profile": getattr(agent_config, "permissions_profile", None),
            "valid": validate_agent_file(agent_path)[0],
            "instructions": getattr(agent_config, "instructions", ""),
        }
    except Exception as e:
        return {
            "error": str(e),
            "valid": False,
        }
