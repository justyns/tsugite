"""Agent markdown parser and template renderer."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import parse_yaml_frontmatter


@dataclass
class AgentConfig:
    """Agent configuration from YAML frontmatter."""

    name: str
    description: str = ""
    model: Optional[str] = None
    max_steps: int = 5
    tools: List[str] = field(default_factory=list)
    prefetch: List[Dict[str, Any]] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)
    permissions_profile: str = "default"
    context_budget: Dict[str, Any] = field(default_factory=dict)
    instructions: str = ""
    mcp_servers: Optional[Dict[str, Optional[List[str]]]] = None
    extends: Optional[str] = None
    reasoning_effort: Optional[str] = None  # For reasoning models (low, medium, high)
    custom_tools: List[Dict[str, Any]] = field(default_factory=list)  # Per-agent shell tools

    def __post_init__(self):
        if self.tools is None:
            self.tools = []
        if self.prefetch is None:
            self.prefetch = []
        if self.attachments is None:
            self.attachments = []
        if not self.context_budget:
            self.context_budget = {"tokens": 8000, "priority": ["system", "task"]}
        if self.instructions is None:
            self.instructions = ""
        if self.mcp_servers is None:
            self.mcp_servers = {}
        if self.custom_tools is None:
            self.custom_tools = []


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
    """Parse an agent markdown file with YAML frontmatter.

    This function also resolves agent inheritance if configured.

    Args:
        file_path: Path to agent markdown file

    Returns:
        Parsed Agent with resolved inheritance

    Raises:
        FileNotFoundError: If agent file doesn't exist
        ValueError: If circular inheritance or missing parent agent
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Agent file not found: {file_path}")

    content = file_path.read_text(encoding="utf-8")
    agent = parse_agent(content, file_path)

    # Resolve inheritance if needed
    if agent.config.extends or agent.config.extends != "none":
        from .agent_inheritance import resolve_agent_inheritance

        agent = resolve_agent_inheritance(agent, file_path)

    return agent


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


@dataclass
class ToolDirective:
    """Represents a tool directive in agent content."""

    name: str
    args: Dict[str, Any]
    assign_var: str
    start_pos: int
    end_pos: int
    raw_match: str


def extract_tool_directives(content: str) -> List[ToolDirective]:
    """Extract <!-- tsu:tool --> directives from content.

    Args:
        content: Raw markdown content

    Returns:
        List of ToolDirective objects with parsed name, args, and assignment

    Example:
        >>> content = '''
        ... <!-- tsu:tool name="fetch_json" args={"url": "http://example.com"} assign="data" -->
        ... '''
        >>> directives = extract_tool_directives(content)
        >>> directives[0].name
        'fetch_json'
        >>> directives[0].args
        {'url': 'http://example.com'}
    """
    directives = []

    # Pattern to match <!-- tsu:tool name="..." args={...} assign="..." -->
    # Uses non-greedy match to handle multiple directives
    pattern = r"<!--\s*tsu:tool\s+([^>]+?)\s*-->"

    for match in re.finditer(pattern, content):
        raw_args = match.group(1).strip()
        start_pos = match.start()
        end_pos = match.end()
        raw_match = match.group(0)

        # Extract name (required)
        name_match = re.search(r'name=(["\']?)(\w+)\1', raw_args)
        if not name_match:
            raise ValueError(f"Tool directive missing required 'name' attribute: {raw_args}")

        tool_name = name_match.group(2)

        # Extract args (required, should be JSON)
        # Find args= and extract the JSON object that follows
        args_start = raw_args.find("args=")
        if args_start == -1:
            raise ValueError(f"Tool directive missing required 'args' attribute: {raw_args}")

        # Find the opening brace after args=
        json_start = raw_args.find("{", args_start)
        if json_start == -1:
            raise ValueError(f"Tool directive args must be a JSON object: {raw_args}")

        # Find the matching closing brace (handle nested braces)
        brace_count = 0
        json_end = json_start
        for i in range(json_start, len(raw_args)):
            if raw_args[i] == "{":
                brace_count += 1
            elif raw_args[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break

        if brace_count != 0:
            raise ValueError(f"Unmatched braces in tool directive args: {raw_args}")

        # Parse JSON args
        try:
            args_json_str = raw_args[json_start:json_end]
            args_dict = json.loads(args_json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in tool directive args for '{tool_name}': {e}")

        # Extract assign (required)
        assign_match = re.search(r'assign=(["\']?)(\w+)\1', raw_args)
        if not assign_match:
            raise ValueError(f"Tool directive missing required 'assign' attribute: {raw_args}")

        assign_var = assign_match.group(2)

        directives.append(
            ToolDirective(
                name=tool_name,
                args=args_dict,
                assign_var=assign_var,
                start_pos=start_pos,
                end_pos=end_pos,
                raw_match=raw_match,
            )
        )

    return directives


@dataclass
class StepDirective:
    """Represents a step in multi-step agent execution."""

    name: str
    content: str
    assign_var: Optional[str] = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    max_retries: int = 0
    retry_delay: float = 0.0
    start_pos: int = 0
    end_pos: int = 0


def extract_step_directives(content: str, include_preamble: bool = True) -> tuple[str, List[StepDirective]]:
    """Extract step directives and preamble from markdown content.

    Steps are marked with <!-- tsu:step name="..." assign="..." --> comments.
    Each step's content runs from the directive to the next step (or EOF).
    Content before the first step is the preamble and can be prepended to all steps.

    Args:
        content: Raw markdown content
        include_preamble: If True, prepend preamble to each step's content

    Returns:
        Tuple of (preamble, list of StepDirective objects with parsed attributes and content)

    Example:
        >>> content = '''
        ... Header content
        ... <!-- tsu:step name="research" assign="data" -->
        ... Do research here
        ... <!-- tsu:step name="write" -->
        ... Write content
        ... '''
        >>> preamble, steps = extract_step_directives(content)
        >>> len(steps)
        2
        >>> 'Header content' in preamble
        True
    """
    steps = []

    # Pattern to match <!-- tsu:step name="..." assign="..." -->
    # Both name and assign can use single or double quotes, or no quotes
    pattern = r"<!--\s*tsu:step\s+([^>]+?)\s*-->"

    matches = list(re.finditer(pattern, content))

    # Extract preamble (content before first step)
    preamble = ""
    if matches:
        preamble = content[: matches[0].start()].strip()

    for i, match in enumerate(matches):
        args = match.group(1).strip()
        start_pos = match.end()

        # Determine end position (next step or EOF)
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(content)

        # Extract step content
        step_content = content[start_pos:end_pos].strip()

        # Prepend preamble to each step if requested and preamble exists
        if include_preamble and preamble:
            step_content = f"{preamble}\n\n{step_content}"

        # Parse name attribute (required)
        name_match = re.search(r'name=(["\']?)(\w+)\1', args)
        if not name_match:
            raise ValueError(f"Step directive missing required 'name' attribute: {args}")

        step_name = name_match.group(2)

        # Parse assign attribute (optional)
        assign_var = None
        assign_match = re.search(r'assign=(["\']?)(\w+)\1', args)
        if assign_match:
            assign_var = assign_match.group(2)

        # Parse model_kwargs from remaining attributes
        model_kwargs = {}

        # Check for json shorthand
        json_match = re.search(r'json=(["\']?)(true|false)\1', args)
        if json_match and json_match.group(2) == "true":
            model_kwargs["response_format"] = {"type": "json_object"}

        # Check for response_format (overrides json shorthand)
        rf_match = re.search(r'response_format=(["\'])(.+?)\1', args)
        if rf_match:
            try:
                rf_value = rf_match.group(2)
                model_kwargs["response_format"] = json.loads(rf_value)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in response_format for step '{step_name}': {e}")

        # Parse temperature
        temp_match = re.search(r'temperature=(["\']?)([0-9.]+)\1', args)
        if temp_match:
            model_kwargs["temperature"] = float(temp_match.group(2))

        # Parse max_tokens
        tokens_match = re.search(r'max_tokens=(["\']?)([0-9]+)\1', args)
        if tokens_match:
            model_kwargs["max_tokens"] = int(tokens_match.group(2))

        # Parse top_p
        top_p_match = re.search(r'top_p=(["\']?)([0-9.]+)\1', args)
        if top_p_match:
            model_kwargs["top_p"] = float(top_p_match.group(2))

        # Parse frequency_penalty
        freq_match = re.search(r'frequency_penalty=(["\']?)([0-9.]+)\1', args)
        if freq_match:
            model_kwargs["frequency_penalty"] = float(freq_match.group(2))

        # Parse presence_penalty
        pres_match = re.search(r'presence_penalty=(["\']?)([0-9.]+)\1', args)
        if pres_match:
            model_kwargs["presence_penalty"] = float(pres_match.group(2))

        # Parse reasoning_effort (for o1, o3, Claude extended thinking)
        reasoning_match = re.search(r'reasoning_effort=(["\']?)(low|medium|high)\1', args)
        if reasoning_match:
            model_kwargs["reasoning_effort"] = reasoning_match.group(2)

        # Parse max_retries
        max_retries = 0
        retries_match = re.search(r'max_retries=(["\']?)([0-9]+)\1', args)
        if retries_match:
            max_retries = int(retries_match.group(2))

        # Parse retry_delay
        retry_delay = 0.0
        delay_match = re.search(r'retry_delay=(["\']?)([0-9.]+)\1', args)
        if delay_match:
            retry_delay = float(delay_match.group(2))

        steps.append(
            StepDirective(
                name=step_name,
                content=step_content,
                assign_var=assign_var,
                model_kwargs=model_kwargs,
                max_retries=max_retries,
                retry_delay=retry_delay,
                start_pos=start_pos,
                end_pos=end_pos,
            )
        )

    return preamble, steps


def has_step_directives(content: str) -> bool:
    """Check if markdown content contains step directives.

    Args:
        content: Raw markdown content

    Returns:
        True if content has at least one step directive
    """
    pattern = r"<!--\s*tsu:step\s+"
    return bool(re.search(pattern, content))


def validate_agent(agent: Agent) -> List[str]:
    """Validate agent configuration and content."""
    errors = []

    # Validate required fields
    if not agent.config.name:
        errors.append("Agent name is required")

    # Model is now optional - it will be loaded from config if not specified

    # Validate model format if specified
    if agent.config.model and not _is_valid_model_format(agent.config.model):
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

    # Only validate model if specified in agent config
    if agent.config.model:
        try:
            from .models import parse_model_string

            parse_model_string(agent.config.model)
        except Exception as e:
            return False, f"Model validation failed: {e}"

    # Validate that tools exist (with helpful error messages)
    missing_tools = []
    if agent.config.tools:
        from .tools import _tools as registered_tools

        # Expand tool specs to get actual tool names
        try:
            from .tools import expand_tool_specs

            expanded_tools = expand_tool_specs(agent.config.tools)

            # Check each tool
            for tool_name in expanded_tools:
                if tool_name not in registered_tools:
                    missing_tools.append(tool_name)
        except Exception as e:
            # If expansion fails, that's a validation error
            return False, f"Tool specification error: {e}"

    if missing_tools:
        from .shell_tool_config import get_custom_tools_config_path

        error_msg = f"Tool(s) not found: {', '.join(missing_tools)}. "

        # Check if custom tools config exists
        config_path = get_custom_tools_config_path()
        if config_path.exists():
            error_msg += f"Check {config_path} for custom tool definitions. "
        else:
            error_msg += f"Create {config_path} to define custom tools. "

        error_msg += "Run 'tsugite tools list' to see available tools."

        return False, error_msg

    # Check template rendering with minimal context
    from .renderer import AgentRenderer

    renderer = AgentRenderer()
    try:
        test_context = {
            "user_prompt": "test",
            "task_summary": "## Current Tasks\nNo tasks yet.",
            "is_interactive": False,
        }

        # If agent has prefetch, create mock variables
        if agent.config.prefetch:
            for prefetch_item in agent.config.prefetch:
                assign_name = prefetch_item.get("assign")
                if assign_name:
                    test_context[assign_name] = "mock_prefetch_data"

        # If agent has steps, create mock variables for step assignments
        # This allows validation to pass for multi-step agents where variables
        # are created dynamically by earlier steps
        if has_step_directives(agent.content):
            try:
                preamble, steps = extract_step_directives(agent.content)
                for step in steps:
                    if step.assign_var:
                        test_context[step.assign_var] = "mock_step_result"
            except Exception:
                # If step parsing fails, that's a different validation error
                # Let it be caught by the normal validation flow
                pass

        renderer.render(agent.content, test_context)
    except Exception as e:
        return False, f"Template validation failed: {e}"

    return True, "Agent is valid"
