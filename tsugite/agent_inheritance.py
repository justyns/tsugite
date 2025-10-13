"""Agent inheritance resolution for Tsugite."""

import os
from pathlib import Path
from typing import List, Optional, Set


def get_global_agents_paths() -> List[Path]:
    """Get global agent directory paths in precedence order.

    Returns:
        List of paths to check for global agents (XDG-compliant)
    """
    paths = []

    # ~/.tsugite/agents/
    home_tsugite = Path.home() / ".tsugite" / "agents"
    paths.append(home_tsugite)

    # $XDG_CONFIG_HOME/tsugite/agents/
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        xdg_path = Path(xdg_config) / "tsugite" / "agents"
        paths.append(xdg_path)

    # ~/.config/tsugite/agents/
    default_path = Path.home() / ".config" / "tsugite" / "agents"
    paths.append(default_path)

    return paths


def find_agent_file(agent_ref: str, current_agent_dir: Path) -> Optional[Path]:
    """Find an agent file by reference.

    Args:
        agent_ref: Agent reference (name or path)
        current_agent_dir: Directory of the agent doing the extending

    Returns:
        Path to agent file if found, None otherwise

    Search order:
    1. Built-in agents (e.g., "builtin-default")
    2. If path-like (contains / or .md), resolve relative to current agent
    3. .tsugite/{name}.md (project-local shared)
    4. ./agents/{name}.md (project convention)
    5. ./{name}.md (current directory)
    6. Global agent directories (XDG order)
    """
    # Check if it's a built-in agent
    from .builtin_agents import is_builtin_agent

    if is_builtin_agent(agent_ref):
        return Path(f"<{agent_ref}>")

    # If it looks like a path, resolve it relative to current agent
    if "/" in agent_ref or agent_ref.endswith(".md"):
        path = current_agent_dir / agent_ref
        if path.exists():
            return path.resolve()
        # Also try as absolute path
        abs_path = Path(agent_ref).expanduser()
        if abs_path.exists():
            return abs_path.resolve()
        return None

    # Ensure .md extension for name-based lookup
    agent_name = agent_ref if agent_ref.endswith(".md") else f"{agent_ref}.md"

    # Search project-local locations
    search_paths = [
        current_agent_dir / ".tsugite" / agent_name,
        current_agent_dir / "agents" / agent_name,
        current_agent_dir / agent_name,
    ]

    # Add global locations
    for global_dir in get_global_agents_paths():
        search_paths.append(global_dir / agent_name)

    # Return first existing path
    for path in search_paths:
        if path.exists():
            return path.resolve()

    return None


def detect_circular_inheritance(agent_path: Path, inheritance_chain: Set[Path]) -> bool:
    """Detect circular inheritance.

    Args:
        agent_path: Path of agent being loaded
        inheritance_chain: Set of agent paths already in the chain

    Returns:
        True if circular inheritance detected, False otherwise
    """
    resolved_path = agent_path.resolve()
    return resolved_path in inheritance_chain


def load_extended_agent(extends_ref: str, current_agent_path: Path, inheritance_chain: Set[Path]):
    """Load an extended (parent) agent.

    Args:
        extends_ref: Reference to parent agent
        current_agent_path: Path to current agent file
        inheritance_chain: Set of agent paths in current inheritance chain

    Returns:
        Parsed Agent object

    Raises:
        ValueError: If agent not found or circular inheritance detected
    """
    from .md_agents import parse_agent

    current_agent_dir = current_agent_path.parent if current_agent_path.is_file() else current_agent_path

    agent_path = find_agent_file(extends_ref, current_agent_dir)

    if agent_path is None:
        raise ValueError(f"Extended agent not found: {extends_ref}")

    # Check for circular inheritance
    if detect_circular_inheritance(agent_path, inheritance_chain):
        chain_str = " -> ".join(str(p) for p in inheritance_chain)
        raise ValueError(f"Circular inheritance detected: {chain_str} -> {agent_path}")

    # Load the parent agent WITHOUT resolving its inheritance yet
    # (we'll do that recursively in resolve_agent_inheritance)

    # Handle built-in agents
    if str(agent_path).startswith("<builtin-"):
        from .builtin_agents import get_builtin_chat_assistant, get_builtin_default_agent

        if "builtin-default" in str(agent_path):
            return get_builtin_default_agent()
        elif "builtin-chat-assistant" in str(agent_path):
            return get_builtin_chat_assistant()
        else:
            raise ValueError(f"Unknown built-in agent: {agent_path}")

    if not agent_path.exists():
        raise ValueError(f"Agent file not found: {agent_path}")

    content = agent_path.read_text(encoding="utf-8")
    return parse_agent(content, agent_path)


def merge_agent_configs(parent, child):
    """Merge parent and child agent configurations.

    Args:
        parent: Parent AgentConfig
        child: Child AgentConfig

    Returns:
        Merged AgentConfig with child taking precedence

    Merge rules:
    - Scalars (model, max_steps, etc.): child overwrites parent
    - Lists (tools, prefetch): merge and deduplicate
    - Dicts (mcp_servers): merge, child keys override
    - Strings (instructions): concatenate with newline
    """
    merged_data = {}

    # Scalars - child overwrites parent if set
    merged_data["name"] = child.name if child.name else parent.name
    merged_data["description"] = child.description if child.description else parent.description
    merged_data["model"] = child.model if child.model else parent.model
    merged_data["max_steps"] = child.max_steps if child.max_steps != 5 else parent.max_steps
    merged_data["permissions_profile"] = (
        child.permissions_profile if child.permissions_profile != "default" else parent.permissions_profile
    )

    # Lists - merge and deduplicate
    parent_tools = parent.tools if parent.tools else []
    child_tools = child.tools if child.tools else []
    merged_tools = list(dict.fromkeys(parent_tools + child_tools))  # Preserves order, deduplicates
    merged_data["tools"] = merged_tools

    parent_prefetch = parent.prefetch if parent.prefetch else []
    child_prefetch = child.prefetch if child.prefetch else []
    merged_data["prefetch"] = parent_prefetch + child_prefetch

    # Dicts - merge with child override
    parent_mcp = parent.mcp_servers if parent.mcp_servers else {}
    child_mcp = child.mcp_servers if child.mcp_servers else {}
    merged_mcp = {**parent_mcp, **child_mcp}
    merged_data["mcp_servers"] = merged_mcp if merged_mcp else None

    parent_context = parent.context_budget if parent.context_budget else {}
    child_context = child.context_budget if child.context_budget else {}
    merged_context = {**parent_context, **child_context}
    merged_data["context_budget"] = merged_context

    # Strings - concatenate instructions
    parent_instructions = parent.instructions if parent.instructions else ""
    child_instructions = child.instructions if child.instructions else ""

    if parent_instructions and child_instructions:
        merged_data["instructions"] = f"{parent_instructions}\n\n{child_instructions}"
    elif parent_instructions:
        merged_data["instructions"] = parent_instructions
    elif child_instructions:
        merged_data["instructions"] = child_instructions
    else:
        merged_data["instructions"] = ""

    from .md_agents import AgentConfig

    return AgentConfig(**merged_data)


def resolve_agent_inheritance(agent, agent_path: Path, inheritance_chain: Optional[Set[Path]] = None):
    """Resolve agent inheritance chain.

    Args:
        agent: Parsed Agent object
        agent_path: Path to agent file
        inheritance_chain: Set of agent paths already processed (for cycle detection)

    Returns:
        Agent object with resolved inheritance

    Inheritance chain:
    1. Default base agent (from config, usually "default")
    2. Extended agent (if extends specified)
    3. Current agent

    Raises:
        ValueError: If circular inheritance or missing agent
    """
    from .md_agents import Agent

    if inheritance_chain is None:
        inheritance_chain = set()

    # Add current agent to chain
    resolved_path = agent_path.resolve()
    inheritance_chain.add(resolved_path)

    # Start with current agent's config
    current_config = agent.config

    # Check if explicitly opted out
    if current_config.extends == "none":
        return agent

    # Build inheritance chain
    configs_to_merge = []

    # 1. Load default base agent (if not the current agent and not opted out)
    default_base_name = _get_default_base_agent_name()
    if default_base_name and current_config.extends != "none":
        # Don't load default if current agent IS the default
        default_path = find_agent_file(default_base_name, agent_path.parent)
        if default_path and default_path.resolve() != resolved_path:
            # Only load if not already in chain (prevents infinite recursion)
            if default_path.resolve() not in inheritance_chain:
                try:
                    default_agent = load_extended_agent(default_base_name, agent_path, inheritance_chain.copy())
                    # Recursively resolve default agent's inheritance
                    default_agent = resolve_agent_inheritance(default_agent, default_path, inheritance_chain.copy())
                    configs_to_merge.append(default_agent.config)
                except ValueError:
                    pass
        elif default_base_name == "default" and default_path is None:
            # No user default.md found, use built-in fallback
            from .builtin_agents import get_builtin_default_agent

            try:
                builtin_agent = get_builtin_default_agent()
                configs_to_merge.append(builtin_agent.config)
            except Exception:
                pass  # Silently fail if built-in can't be loaded

    # 2. Load explicitly extended agent
    if current_config.extends and current_config.extends != "none":
        parent_agent = load_extended_agent(current_config.extends, agent_path, inheritance_chain.copy())
        parent_path = parent_agent.file_path

        # Recursively resolve parent's inheritance
        parent_agent = resolve_agent_inheritance(parent_agent, parent_path, inheritance_chain.copy())
        configs_to_merge.append(parent_agent.config)

    # 3. Current agent is last (highest precedence)
    configs_to_merge.append(current_config)

    # Merge all configs in order (earlier = lower precedence)
    if len(configs_to_merge) == 1:
        # No inheritance, return as-is
        return agent

    merged_config = configs_to_merge[0]
    for config in configs_to_merge[1:]:
        merged_config = merge_agent_configs(merged_config, config)

    # Return new Agent with merged config
    return Agent(config=merged_config, content=agent.content, file_path=agent_path)


def _get_default_base_agent_name() -> Optional[str]:
    """Get the default base agent name from config.

    Returns:
        Default base agent name, or "default" as fallback, or None if disabled
    """
    from .config import load_config

    try:
        config = load_config()
        if config.default_base_agent is None:
            return "default"
        return config.default_base_agent
    except Exception:
        return "default"
