"""Agent markdown parser and template renderer."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import re


@dataclass
class AgentConfig:
    """Agent configuration from YAML frontmatter."""

    name: str
    description: str = ""
    model: str = "ollama:qwen2.5-coder:7b"
    max_steps: int = 5
    tools: List[str] = []
    prefetch: List[Dict[str, Any]] = []
    permissions_profile: str = "default"
    context_budget: Dict[str, Any] = {}

    def __post_init__(self):
        if self.tools is None:
            self.tools = []
        if self.prefetch is None:
            self.prefetch = []
        if self.context_budget is None:
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

    # Check for YAML frontmatter
    if not text.startswith("---"):
        raise ValueError("Agent text must start with YAML frontmatter")

    # Split frontmatter and content
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError("Invalid YAML frontmatter format")

    # Parse YAML frontmatter
    try:
        frontmatter = yaml.safe_load(parts[1])
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML frontmatter: {e}")

    # Create config
    config = AgentConfig(**frontmatter)

    # Get markdown content (everything after second ---)
    markdown_content = parts[2].strip()

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

    # Validate tools exist (TODO: check against registry)
    for tool in agent.config.tools:
        if not isinstance(tool, str):
            errors.append(f"Tool name must be string: {tool}")

    # Validate directives in content
    directives = extract_directives(agent.content)
    for directive in directives:
        if directive["type"] == "tool" and "name" not in directive:
            errors.append(f"Tool directive missing name: {directive['raw_args']}")

    return errors
