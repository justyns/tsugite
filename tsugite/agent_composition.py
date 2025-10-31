"""Agent composition utilities for multi-agent delegation."""

from pathlib import Path
from typing import List, Tuple

from tsugite.core.tools import Tool


def resolve_agent_reference(ref: str, base_dir: Path) -> Path:
    """Resolve agent reference to file path.

    Args:
        ref: Agent reference (path or +name shorthand)
        base_dir: Base directory for relative path resolution

    Returns:
        Resolved Path to agent file

    Raises:
        ValueError: If agent not found or invalid
    """
    # Handle +name shorthand
    if ref.startswith("+"):
        agent_name = ref[1:]
        from .agent_inheritance import find_agent_file

        found = find_agent_file(agent_name, base_dir)
        if not found:
            raise ValueError(f"Agent not found: {ref} (searched standard locations)")
        return found

    # Handle regular path
    agent_path = Path(ref)
    if not agent_path.is_absolute():
        agent_path = base_dir / agent_path

    if agent_path.exists():
        return agent_path

    # If path doesn't exist, try as agent name lookup
    # This allows "helper" or "helper.md" to work like "+helper"
    from .agent_inheritance import find_agent_file

    found = find_agent_file(ref, base_dir)
    if found:
        return found

    raise ValueError(f"Agent file not found: {ref}")


def create_delegation_tool(agent_name: str, agent_path: Path) -> Tool:
    """Create a delegation tool for a specific agent.

    Args:
        agent_name: Name for the tool (e.g., "jira" -> spawn_jira)
        agent_path: Path to the agent file

    Returns:
        Tool that wraps spawn_agent with pre-filled path
    """

    def delegation_function(prompt: str) -> str:
        """Execute the delegated agent.

        Args:
            prompt: Task/prompt to give the agent

        Returns:
            str: Result from the delegated agent
        """
        from .agent_runner import run_agent

        try:
            result = run_agent(
                agent_path=agent_path,
                prompt=prompt,
                context={},
                model_override=None,
                debug=False,
            )
            return str(result)
        except Exception as e:
            return f"Error executing {agent_name} agent: {e}"

    # Create Tool instance with proper schema
    return Tool(
        name=f"spawn_{agent_name}",
        description=f"Delegate a task to the {agent_name} agent and get the result.",
        parameters={
            "type": "object",
            "properties": {"prompt": {"type": "string", "description": "Task/prompt to give the agent"}},
            "required": ["prompt"],
        },
        function=delegation_function,
    )


def create_delegation_tools(delegation_agents: List[Tuple[str, Path]]) -> List[Tool]:
    """Create delegation tools for multiple agents.

    Args:
        delegation_agents: List of (agent_name, agent_path) tuples

    Returns:
        List of delegation tools
    """
    tools = []
    for agent_name, agent_path in delegation_agents:
        tool = create_delegation_tool(agent_name, agent_path)
        tools.append(tool)
    return tools


def parse_agent_references(
    agent_refs: List[str], with_agents_str: str | None, base_dir: Path
) -> Tuple[Path, List[Tuple[str, Path]]]:
    """Parse agent references into primary agent and delegation agents.

    Args:
        agent_refs: List of agent references from CLI args
        with_agents_str: Comma or space separated string from --with-agents option
        base_dir: Base directory for resolution

    Returns:
        Tuple of (primary_agent_path, [(name, path), ...])

    Raises:
        ValueError: If no agents specified or resolution fails
    """
    if not agent_refs:
        raise ValueError("No agent specified")

    # Resolve primary agent (first in list)
    primary_ref = agent_refs[0]
    primary_path = resolve_agent_reference(primary_ref, base_dir)

    # Collect delegation agent references
    delegation_refs = []

    # Add remaining positional agents
    if len(agent_refs) > 1:
        delegation_refs.extend(agent_refs[1:])

    # Add --with-agents if provided
    if with_agents_str:
        # Support both comma and space separated
        import re

        refs = re.split(r"[,\s]+", with_agents_str.strip())
        delegation_refs.extend([ref for ref in refs if ref])

    # Resolve delegation agents
    delegation_agents = []
    for ref in delegation_refs:
        agent_path = resolve_agent_reference(ref, base_dir)
        # Extract name from reference (strip + if present, remove .md extension)
        agent_name = ref.lstrip("+").removesuffix(".md")
        # If it's a path, use the filename without extension
        if "/" in agent_name:
            agent_name = Path(agent_name).stem
        delegation_agents.append((agent_name, agent_path))

    return primary_path, delegation_agents
