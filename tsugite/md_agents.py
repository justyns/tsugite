"""Agent markdown parser and template renderer."""

from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import re
from .utils import parse_yaml_frontmatter


@dataclass
class AgentConfig:
    """Agent configuration from YAML frontmatter."""

    name: str
    description: str = ""
    model: str = "ollama:qwen2.5-coder:7b"
    max_steps: int = 5
    tools: List[str] = field(default_factory=list)
    prefetch: List[Dict[str, Any]] = field(default_factory=list)
    permissions_profile: str = "default"
    context_budget: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.tools is None:
            self.tools = []
        if self.prefetch is None:
            self.prefetch = []
        if not self.context_budget:
            self.context_budget = {"tokens": 8000, "priority": ["system", "task"]}


@dataclass
class Agent:
    """Parsed agent with config and content."""

    config: AgentConfig
    content: str
    file_path: Path

    @property
    def name(self) -> str:
        return self.config.name


def parse_agent(text: str, file_path: Optional[Path] = None) -> Agent:
    """Parse agent markdown text with YAML frontmatter."""
    # Parse YAML frontmatter
    frontmatter, markdown_content = parse_yaml_frontmatter(text, "Agent text")

    # Create config
    config = AgentConfig(**frontmatter)

    return Agent(config=config, content=markdown_content, file_path=file_path or Path(""))


def parse_agent_file(file_path: Path) -> Agent:
    """Parse an agent markdown file with YAML frontmatter."""

    if not file_path.exists():
        raise FileNotFoundError(f"Agent file not found: {file_path}")

    content = file_path.read_text(encoding="utf-8")
    return parse_agent(content, file_path)


def extract_directives(content: str) -> List[Dict[str, Any]]:
    """Extract tsugite directives from markdown content."""
    directives = []

    # Pattern to match <!-- tsu:directive ... -->
    pattern = r"<!--\s*tsu:(\w+)\s+([^>]+)\s*-->"

    for match in re.finditer(pattern, content):
        directive_type = match.group(1)
        directive_args = match.group(2).strip()

        # Parse directive arguments (simplified for now)
        directive = {
            "type": directive_type,
            "raw_args": directive_args,
            "position": match.span(),
        }

        # Basic parsing for common patterns
        if "name=" in directive_args and "args=" in directive_args:
            # Extract tool name and args
            name_match = re.search(r'name=(["\']?)(\w+)\1', directive_args)
            if name_match:
                directive["name"] = name_match.group(2)

            # Extract assign parameter
            assign_match = re.search(r'assign=(["\']?)(\w+)\1', directive_args)
            if assign_match:
                directive["assign"] = assign_match.group(2)

        directives.append(directive)

    return directives


def validate_agent(agent: Agent) -> List[str]:
    """Validate agent configuration and content."""
    errors = []

    # Validate required fields
    if not agent.config.name:
        errors.append("Agent name is required")

    if not agent.config.model:
        errors.append("Agent model is required")

    # Validate model format
    if not _is_valid_model_format(agent.config.model):
        errors.append(f"Invalid model format: {agent.config.model}")

    # Validate tools exist
    for tool in agent.config.tools:
        if not isinstance(tool, str):
            errors.append(f"Tool name must be string: {tool}")

    # Validate max_steps is positive
    if agent.config.max_steps <= 0:
        errors.append("max_steps must be positive")

    # Validate directives in content
    directives = extract_directives(agent.content)
    for directive in directives:
        if directive["type"] == "tool" and "name" not in directive:
            errors.append(f"Tool directive missing name: {directive['raw_args']}")

    return errors


def _is_valid_model_format(model: str) -> bool:
    """Check if model string follows expected format."""
    if ":" not in model:
        return False

    provider, model_name = model.split(":", 1)
    return bool(provider.strip() and model_name.strip())


def validate_agent_execution(agent: Agent | Path) -> tuple[bool, str]:
    """Validate that an agent can be executed.

    Args:
        agent: Parsed agent or path to agent file

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Handle both Path objects and Agent objects
    if isinstance(agent, Path):
        try:
            agent_text = agent.read_text()
            agent = parse_agent(agent_text, agent)
        except Exception as e:
            return False, f"Failed to parse agent file: {e}"

    # First do basic validation
    errors = validate_agent(agent)
    if errors:
        return False, "; ".join(errors)

    try:
        # Check if model is supported
        from .models import get_model

        get_model(agent.config.model)
    except Exception as e:
        return False, f"Model validation failed: {e}"

    try:
        # Check if tools exist
        from .tool_adapter import get_smolagents_tools

        get_smolagents_tools(agent.config.tools)
    except Exception as e:
        return False, f"Tool validation failed: {e}"

    # Check template rendering with minimal context
    from .renderer import AgentRenderer

    renderer = AgentRenderer()
    try:
        test_context = {"user_prompt": "test", "task_summary": "## Current Tasks\nNo tasks yet."}

        # If agent has prefetch, create mock variables
        if agent.config.prefetch:
            for prefetch_item in agent.config.prefetch:
                assign_name = prefetch_item.get("assign")
                if assign_name:
                    test_context[assign_name] = "mock_prefetch_data"

        renderer.render(agent.content, test_context)
    except Exception as e:
        return False, f"Template validation failed: {e}"

    return True, "Agent is valid"
