"""Agent inheritance resolution for Tsugite."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tsugite.utils import ensure_file_exists


def get_builtin_agents_path() -> Path:
    """Get the built-in agents directory path.

    Returns:
        Path to built-in agents directory within the package
    """
    return Path(__file__).parent / "builtin_agents"


def get_global_agents_paths() -> List[Path]:
    """Get global agent directory paths in precedence order.

    Returns:
        List of paths to check for global agents (XDG-compliant)
    """
    paths = []

    # $XDG_CONFIG_HOME/tsugite/agents/
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        xdg_path = Path(xdg_config) / "tsugite" / "agents"
        paths.append(xdg_path)

    # ~/.config/tsugite/agents/
    default_path = Path.home() / ".config" / "tsugite" / "agents"
    paths.append(default_path)

    return paths


def find_agent_file(
    agent_ref: str,
    current_agent_dir: Path,
    workspace: Optional[Any] = None,
) -> Optional[Path]:
    """Find an agent file by reference with workspace priority.

    Args:
        agent_ref: Agent reference (name or path)
        current_agent_dir: Directory of the agent doing the extending
        workspace: Optional workspace to check for workspace-specific agents

    Returns:
        Path to agent file if found, None otherwise

    Search order:
    1. If path-like (contains / or .md), resolve relative to current agent
    2. Workspace agents directory (if workspace provided)
    3. .tsugite/{name}.md (project-local shared)
    4. ./agents/{name}.md (project convention)
    5. ./{name}.md (current directory)
    6. Built-in agents directory (tsugite/builtin_agents/)
    7. Global agent directories (XDG order)
    """
    # If it looks like a path, resolve it relative to current agent
    if "/" in agent_ref or agent_ref.endswith(".md"):
        path = current_agent_dir / agent_ref
        if path.exists():
            return path.resolve()
        abs_path = Path(agent_ref).expanduser()
        if abs_path.exists():
            return abs_path.resolve()
        return None

    # Ensure .md extension for name-based lookup
    agent_name = agent_ref if agent_ref.endswith(".md") else f"{agent_ref}.md"

    search_paths = []

    # Workspace agents directory (highest priority)
    if workspace and hasattr(workspace, "agents_dir"):
        workspace_agents = workspace.agents_dir
        if workspace_agents.exists():
            search_paths.append(workspace_agents / agent_name)

    # Project-local locations
    search_paths.extend(
        [
            current_agent_dir / ".tsugite" / agent_name,
            current_agent_dir / "agents" / agent_name,
            current_agent_dir / agent_name,
        ]
    )

    # Built-in agents directory
    builtin_path = get_builtin_agents_path() / agent_name
    search_paths.append(builtin_path)

    # Global locations
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
    ensure_file_exists(agent_path, "Agent file")

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
    - Scalars (model, max_turns, reasoning_effort, etc.): child overwrites parent
    - Lists (tools, attachments): merge and deduplicate
    - Lists (prefetch): concatenate (parent first, no deduplication)
    - Lists of dicts (custom_tools): merge and deduplicate by "name" field
    - Dicts (mcp_servers, context_budget): merge, child keys override
    - Strings (instructions): concatenate with newline
    """
    from .md_agents import AgentConfig

    # Merge all field types
    merged_data = {}
    merged_data.update(merge_scalar_fields(parent, child))
    merged_data.update(merge_list_fields(parent, child))
    merged_data.update(merge_dict_fields(parent, child))
    merged_data["instructions"] = merge_instructions(parent, child)

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

    # Check if explicitly opted out
    if agent.config.extends == "none":
        return agent

    # Build inheritance chain
    configs_to_merge = []
    contents_to_merge = []

    # 1. Load default base agent (only for implicit inheritance)
    if agent.config.extends is None:
        default_config, default_content = load_default_base_agent(agent_path, resolved_path, inheritance_chain)
        if default_config:
            configs_to_merge.append(default_config)
            contents_to_merge.append(default_content)

    # 2. Load explicitly extended agent
    if agent.config.extends and agent.config.extends != "none":
        parent_config, parent_content = load_explicit_parent_agent(agent.config.extends, agent_path, inheritance_chain)
        configs_to_merge.append(parent_config)
        contents_to_merge.append(parent_content)

    # 3. Current agent is last (highest precedence)
    configs_to_merge.append(agent.config)
    contents_to_merge.append(agent.content)

    # Merge all configs and content
    if len(configs_to_merge) == 1:
        return agent

    merged_config, merged_content = merge_configs_and_content(configs_to_merge, contents_to_merge)
    return Agent(config=merged_config, content=merged_content, file_path=agent_path)


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


def merge_scalar_fields(parent, child) -> Dict[str, Any]:
    """Merge scalar fields from parent and child configs.

    Child values take precedence when explicitly set.

    Args:
        parent: Parent AgentConfig
        child: Child AgentConfig

    Returns:
        Dict of merged scalar fields
    """
    return {
        "name": child.name if child.name else parent.name,
        "description": child.description if child.description else parent.description,
        "model": child.model if child.model else parent.model,
        "max_turns": child.max_turns if child.max_turns != 5 else parent.max_turns,
        "permissions_profile": (
            child.permissions_profile if child.permissions_profile != "default" else parent.permissions_profile
        ),
        "reasoning_effort": child.reasoning_effort if child.reasoning_effort else parent.reasoning_effort,
        "memory_enabled": child.memory_enabled if child.memory_enabled is not None else parent.memory_enabled,
    }


def merge_list_fields(parent, child) -> Dict[str, List]:
    """Merge list fields from parent and child configs.

    Different list types have different merge strategies:
    - tools, attachments: merge and deduplicate
    - prefetch, initial_tasks: concatenate (parent first)
    - custom_tools: deduplicate by "name" field

    Args:
        parent: Parent AgentConfig
        child: Child AgentConfig

    Returns:
        Dict of merged list fields
    """
    # Simple lists - merge and deduplicate while preserving order
    parent_tools = parent.tools if parent.tools else []
    child_tools = child.tools if child.tools else []
    merged_tools = list(dict.fromkeys(parent_tools + child_tools))

    parent_attachments = parent.attachments if parent.attachments else []
    child_attachments = child.attachments if child.attachments else []
    merged_attachments = list(dict.fromkeys(parent_attachments + child_attachments))

    parent_skills = parent.auto_load_skills if parent.auto_load_skills else []
    child_skills = child.auto_load_skills if child.auto_load_skills else []
    merged_skills = list(dict.fromkeys(parent_skills + child_skills))

    # Lists that concatenate without deduplication
    parent_prefetch = parent.prefetch if parent.prefetch else []
    child_prefetch = child.prefetch if child.prefetch else []

    parent_initial_tasks = parent.initial_tasks if parent.initial_tasks else []
    child_initial_tasks = child.initial_tasks if child.initial_tasks else []

    # Custom tools - deduplicate by "name" field (child overrides parent)
    parent_custom = parent.custom_tools if parent.custom_tools else []
    child_custom = child.custom_tools if child.custom_tools else []
    custom_tool_dict = {}
    for tool in parent_custom + child_custom:
        custom_tool_dict[tool["name"]] = tool

    return {
        "tools": merged_tools,
        "attachments": merged_attachments,
        "prefetch": parent_prefetch + child_prefetch,
        "initial_tasks": parent_initial_tasks + child_initial_tasks,
        "custom_tools": list(custom_tool_dict.values()),
        "auto_load_skills": merged_skills,
    }


def merge_dict_fields(parent, child) -> Dict[str, Any]:
    """Merge dictionary fields from parent and child configs.

    Child keys override parent keys.

    Args:
        parent: Parent AgentConfig
        child: Child AgentConfig

    Returns:
        Dict of merged dict fields
    """
    # MCP servers
    parent_mcp = parent.mcp_servers if parent.mcp_servers else {}
    child_mcp = child.mcp_servers if child.mcp_servers else {}
    merged_mcp = {**parent_mcp, **child_mcp}

    # Context budget
    parent_context = parent.context_budget if parent.context_budget else {}
    child_context = child.context_budget if child.context_budget else {}
    merged_context = {**parent_context, **child_context}

    return {
        "mcp_servers": merged_mcp,
        "context_budget": merged_context,
    }


def merge_instructions(parent, child) -> str:
    """Merge instructions from parent and child configs.

    Concatenates with double newline separator.

    Args:
        parent: Parent AgentConfig
        child: Child AgentConfig

    Returns:
        Merged instructions string
    """
    parent_instructions = parent.instructions if parent.instructions else ""
    child_instructions = child.instructions if child.instructions else ""

    if parent_instructions and child_instructions:
        return f"{parent_instructions}\n\n{child_instructions}"
    elif parent_instructions:
        return parent_instructions
    elif child_instructions:
        return child_instructions
    else:
        return ""


def load_default_base_agent(
    agent_path: Path, resolved_path: Path, inheritance_chain: Set[Path]
) -> tuple[Optional[Any], Optional[str]]:
    """Load default base agent for implicit inheritance.

    Search priority:
    1. User's project-local default.md (if exists)
    2. Config's default_base_agent
    3. Builtin fallback

    Args:
        agent_path: Path to current agent file
        resolved_path: Resolved path to current agent
        inheritance_chain: Set of agent paths already in chain

    Returns:
        Tuple of (loaded_agent, loaded_content) or (None, None) if not loaded
    """
    # Try project-local default.md first
    user_default_path = agent_path.parent / ".tsugite" / "default.md"

    if user_default_path.exists() and user_default_path.resolve() != resolved_path:
        if user_default_path.resolve() not in inheritance_chain:
            try:
                default_agent = load_extended_agent("default", agent_path, inheritance_chain.copy())
                # User's default.md should opt out of builtin inheritance unless explicit
                if default_agent.config.extends is None:
                    # Standalone user default
                    return default_agent.config, default_agent.content
                else:
                    # User explicitly extended something, resolve it
                    default_agent = resolve_agent_inheritance(
                        default_agent, user_default_path, inheritance_chain.copy()
                    )
                    return default_agent.config, default_agent.content
            except ValueError:
                pass

    # Fall back to config or builtin
    default_base_name = _get_default_base_agent_name()
    if not default_base_name:
        return None, None

    # Try to find the configured default agent (including built-ins via find_agent_file)
    default_path = find_agent_file(default_base_name, agent_path.parent)
    if default_path and default_path.resolve() != resolved_path:
        if default_path.resolve() not in inheritance_chain:
            try:
                default_agent = load_extended_agent(default_base_name, agent_path, inheritance_chain.copy())
                default_agent = resolve_agent_inheritance(default_agent, default_path, inheritance_chain.copy())
                return default_agent.config, default_agent.content
            except ValueError:
                pass

    return None, None


def load_explicit_parent_agent(extends_ref: str, agent_path: Path, inheritance_chain: Set[Path]) -> tuple[Any, str]:
    """Load explicitly extended parent agent.

    Args:
        extends_ref: Reference to parent agent
        agent_path: Path to current agent file
        inheritance_chain: Set of agent paths already in chain

    Returns:
        Tuple of (parent_config, parent_content)
    """
    parent_agent = load_extended_agent(extends_ref, agent_path, inheritance_chain.copy())
    parent_path = parent_agent.file_path

    # Recursively resolve parent's inheritance
    parent_agent = resolve_agent_inheritance(parent_agent, parent_path, inheritance_chain.copy())
    return parent_agent.config, parent_agent.content


def merge_configs_and_content(configs: List, contents: List):
    """Merge a list of configs and contents.

    Args:
        configs: List of AgentConfig objects (earlier = lower precedence)
        contents: List of content strings

    Returns:
        Tuple of (merged_config, merged_content)
    """
    if len(configs) == 1:
        return configs[0], contents[0]

    # Merge configs in order
    merged_config = configs[0]
    for config in configs[1:]:
        merged_config = merge_agent_configs(merged_config, config)

    # Merge contents
    merged_content = "\n\n".join(contents)

    return merged_config, merged_content
