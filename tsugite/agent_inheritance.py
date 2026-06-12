"""Agent inheritance resolution for Tsugite."""

import importlib.metadata
import importlib.util
import logging
import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tsugite.utils import ensure_file_exists

logger = logging.getLogger(__name__)


class AgentDirSource(Enum):
    """Origin of an agent search directory."""

    WORKSPACE = "workspace"  # caller-passed workspace agents/ (mutable)
    PROJECT = "project"  # caller-passed project agents/ (mutable; today only http.py)
    BUILTIN = "builtin"  # tsugite/builtin_agents/ (read-only)
    PLUGIN = "plugin"  # plugin <pkg>/agents/ via tsugite.plugins entry-points (read-only)
    GLOBAL = "global"  # XDG global agents/ (mutable)


@dataclass(frozen=True)
class AgentDir:
    """A directory that may contain agent .md files."""

    path: Path
    source: AgentDirSource
    readonly: bool


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


@lru_cache(maxsize=1)
def get_plugin_agents_paths() -> List[Path]:
    """Get agent directory paths contributed by installed plugins.

    Scans the ``tsugite.plugins`` entry-point group; each plugin's package
    may ship an ``agents/`` directory next to its top-level module. Plugin
    agent dirs are treated as read-only.

    Returns:
        List of plugin-supplied agent directories (deduped, only existing dirs)
    """
    from tsugite.plugins import GROUP_PLUGINS

    paths: List[Path] = []
    seen: set[Path] = set()
    try:
        eps = importlib.metadata.entry_points(group=GROUP_PLUGINS)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Failed to scan plugin entry points: %s", e)
        return paths

    for ep in eps:
        module_name = ep.value.split(":")[0].strip()
        top_level = module_name.split(".")[0]
        try:
            spec = importlib.util.find_spec(top_level)
        except Exception:
            spec = None
        if spec is None or not spec.origin:
            continue
        agents_dir = Path(spec.origin).parent / "agents"
        resolved = agents_dir.resolve()
        if resolved in seen:
            continue
        if not agents_dir.is_dir():
            continue
        seen.add(resolved)
        paths.append(agents_dir)
    return paths


def iter_agent_search_paths(
    *,
    current_agent_dir: Optional[Path] = None,
    workspace: Optional[Any] = None,
    extra_project_dirs: Optional[List[Path]] = None,
    include_local_roots: bool = True,
) -> List[AgentDir]:
    """Yield the canonical agent-search order, deduped by resolved path.

    Order:

      1. workspace.agents_dir (if workspace given)
      2. current_agent_dir / ".tsugite"
      3. current_agent_dir / ".tsugite" / "agents"
      4. current_agent_dir / "agents"
      5. current_agent_dir itself (for `<name>.md` directly in cwd)
      6. caller-supplied extra project dirs (HTTP adapter feeds per-agent dirs here)
      7. builtin agents
      8. XDG global agents (user agents beat plugin agents of the same name)
      9. plugin-supplied agents

    include_local_roots: when False, skip entries 2 and 5 (the bare .tsugite/
    and cwd roots). Name-resolution callers want them so `extends: foo` finds a
    sibling foo.md; discovery callers that glob *.md must skip them, otherwise
    every frontmattered note in a docs/notes workspace lists as an agent.

    Callers that only care about a subset filter on ``AgentDir.source``.
    """
    results: List[AgentDir] = []
    seen: set[Path] = set()

    def _add(path: Path, source: AgentDirSource, readonly: bool) -> None:
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError):
            return
        if resolved in seen:
            return
        seen.add(resolved)
        results.append(AgentDir(path=path, source=source, readonly=readonly))

    if workspace is not None and hasattr(workspace, "agents_dir"):
        _add(workspace.agents_dir, AgentDirSource.WORKSPACE, readonly=False)

    if current_agent_dir is not None:
        if include_local_roots:
            _add(current_agent_dir / ".tsugite", AgentDirSource.PROJECT, readonly=False)
        _add(current_agent_dir / ".tsugite" / "agents", AgentDirSource.PROJECT, readonly=False)
        _add(current_agent_dir / "agents", AgentDirSource.PROJECT, readonly=False)
        if include_local_roots:
            _add(current_agent_dir, AgentDirSource.PROJECT, readonly=False)

    for extra in extra_project_dirs or []:
        _add(extra, AgentDirSource.PROJECT, readonly=False)

    _add(get_builtin_agents_path(), AgentDirSource.BUILTIN, readonly=True)

    for global_dir in get_global_agents_paths():
        _add(global_dir, AgentDirSource.GLOBAL, readonly=False)

    for plugin_path in get_plugin_agents_paths():
        _add(plugin_path, AgentDirSource.PLUGIN, readonly=True)

    return results


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

    Search order matches `iter_agent_search_paths`: workspace agents,
    project-local (.tsugite/, agents/, current dir), builtin, global, plugin.
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

    agent_name = agent_ref if agent_ref.endswith(".md") else f"{agent_ref}.md"

    for entry in iter_agent_search_paths(current_agent_dir=current_agent_dir, workspace=workspace):
        candidate = entry.path / agent_name
        if candidate.exists():
            return candidate.resolve()

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
    - Strings (instructions): concatenate with newline
    """
    from .md_agents import AgentConfig

    # Merge all field types
    merged_data = {}
    merged_data.update(merge_scalar_fields(parent, child))
    merged_data.update(merge_list_fields(parent, child))
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
        "reasoning_effort": child.reasoning_effort if child.reasoning_effort else parent.reasoning_effort,
        "auto_load_agent_list": child.auto_load_agent_list or parent.auto_load_agent_list,
    }


def merge_list_fields(parent, child) -> Dict[str, List]:
    """Merge list fields from parent and child configs.

    Different list types have different merge strategies:
    - tools, attachments: merge and deduplicate
    - prefetch: concatenate (parent first)
    - custom_tools: deduplicate by "name" field

    Args:
        parent: Parent AgentConfig
        child: Child AgentConfig

    Returns:
        Dict of merged list fields
    """

    def _merge_dedup(parent_list, child_list):
        return list(dict.fromkeys((parent_list or []) + (child_list or [])))

    def _merge_attachments(parent_list, child_list):
        """Merge attachments preserving parent order; child entries override by key.

        Strings are keyed by their full text (so `-foo.md` and `foo.md` stay distinct).
        AttachmentSpec entries are keyed by `path`, so a child spec with the same path
        replaces the parent spec in place.
        """
        from tsugite.md_agents import AttachmentSpec

        merged: Dict[Any, Any] = {}
        for item in (parent_list or []) + (child_list or []):
            key = ("spec", item.path) if isinstance(item, AttachmentSpec) else ("str", item)
            merged[key] = item
        return list(merged.values())

    merged_tools = _merge_dedup(parent.tools, child.tools)
    merged_attachments = _merge_attachments(parent.attachments, child.attachments)
    merged_skills = _merge_dedup(parent.auto_load_skills, child.auto_load_skills)
    merged_auto_load_agents = _merge_dedup(parent.auto_load_agents, child.auto_load_agents)

    # Custom tools - deduplicate by "name" field (child overrides parent)
    custom_tool_dict = {}
    for tool in (parent.custom_tools or []) + (child.custom_tools or []):
        custom_tool_dict[tool["name"]] = tool

    return {
        "tools": merged_tools,
        "attachments": merged_attachments,
        "prefetch": (parent.prefetch or []) + (child.prefetch or []),
        "custom_tools": list(custom_tool_dict.values()),
        "auto_load_skills": merged_skills,
        "auto_load_agents": merged_auto_load_agents,
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
    parts = [s for s in (parent.instructions, child.instructions) if s]
    return "\n\n".join(parts)


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
