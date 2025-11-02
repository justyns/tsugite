"""Agent markdown parser and template renderer."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .utils import parse_yaml_frontmatter


def _parse_directive_attribute(
    args: str,
    attr_name: str,
    value_pattern: str = r"([^\"']+)",
    converter: Optional[Callable[[str], Any]] = None,
    default: Any = None,
) -> Any:
    """Parse an attribute from directive arguments using regex.

    Args:
        args: The directive arguments string
        attr_name: Name of the attribute to extract
        value_pattern: Regex pattern for the value (default: any non-quote chars)
        converter: Optional function to convert the string value (e.g., int, float)
        default: Default value if attribute not found

    Returns:
        Parsed and converted value, or default if not found
    """
    pattern = rf'{attr_name}=(["\']?)({value_pattern})\1'
    match = re.search(pattern, args)
    if not match:
        return default

    value = match.group(2)
    if converter:
        return converter(value)
    return value


class AgentConfig(BaseModel):
    """Agent configuration from YAML frontmatter."""

    model_config = ConfigDict(
        extra="forbid",  # Reject unknown fields to catch typos in YAML
        str_strip_whitespace=True,  # Auto-strip whitespace from strings
    )

    name: str
    description: str = ""
    model: Optional[str] = None
    max_turns: int = 5
    tools: List[str] = Field(default_factory=list)
    prefetch: List[Dict[str, Any]] = Field(default_factory=list)
    attachments: List[str] = Field(default_factory=list)
    permissions_profile: str = "default"
    context_budget: Dict[str, Any] = Field(default_factory=lambda: {"tokens": 8000, "priority": ["system", "task"]})
    instructions: str = ""
    mcp_servers: Dict[str, Optional[List[str]]] = Field(default_factory=dict)
    extends: Optional[str] = None
    reasoning_effort: Optional[str] = None  # For reasoning models (low, medium, high)
    custom_tools: List[Dict[str, Any]] = Field(default_factory=list)  # Per-agent shell tools
    text_mode: bool = False  # Allow text-only responses (code blocks optional)
    initial_tasks: List[Union[str, Dict[str, Any]]] = Field(
        default_factory=list
    )  # Tasks to pre-populate (strings or dicts)
    disable_history: bool = False  # Disable conversation history persistence for this agent
    auto_context: Optional[bool] = None  # Auto-load context files (None = use config default)
    visibility: str = "public"  # Agent visibility: public, private, internal
    spawnable: bool = True  # Whether this agent can be spawned by other agents

    @field_validator("visibility", mode="after")
    @classmethod
    def validate_visibility(cls, v: str) -> str:
        """Validate visibility is one of the allowed values."""
        allowed = ["public", "private", "internal"]
        if v not in allowed:
            raise ValueError(f"visibility must be one of {allowed}, got: {v}")
        return v

    @field_validator("initial_tasks", mode="after")
    @classmethod
    def normalize_initial_tasks(cls, v: List[Union[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Normalize initial_tasks: convert strings to dicts with defaults."""
        normalized_tasks = []
        for task in v:
            if isinstance(task, str):
                # Simple string format: convert to dict with defaults
                normalized_tasks.append({"title": task, "status": "pending", "optional": False})
            elif isinstance(task, dict):
                # Dict format: ensure all required fields exist with defaults
                normalized = {
                    "title": task.get("title", ""),
                    "status": task.get("status", "pending"),
                    "optional": task.get("optional", False),
                }
                normalized_tasks.append(normalized)
            else:
                raise ValueError(f"Invalid initial_tasks entry: {task}. Must be string or dict.")

        return normalized_tasks


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
    if agent.config.extends != "none":
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

        # Extract and parse JSON args (required)
        args_dict = extract_and_parse_json_args(raw_args, tool_name)

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
    continue_on_error: bool = False
    timeout: Optional[int] = None  # Timeout in seconds, None = no timeout
    start_pos: int = 0
    end_pos: int = 0

    # Loop control
    repeat_while: Optional[str] = None  # Jinja2 expression or helper name to continue repeating
    repeat_until: Optional[str] = None  # Jinja2 expression or helper name to stop repeating
    max_iterations: int = 10  # Maximum loop iterations (safety valve)


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
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)

        # Extract step content
        step_content = content[start_pos:end_pos].strip()

        # Prepend preamble if requested
        if include_preamble and preamble:
            step_content = f"{preamble}\n\n{step_content}"

        # Parse name attribute (required)
        name_match = re.search(r'name=(["\']?)(\w+)\1', args)
        if not name_match:
            raise ValueError(f"Step directive missing required 'name' attribute: {args}")
        step_name = name_match.group(2)

        # Parse optional attributes
        assign_var = _parse_directive_attribute(args, "assign", r"\w+")
        timeout = _parse_directive_attribute(args, "timeout", r"[0-9]+", int)

        # Use helpers for complex parsing
        model_kwargs = parse_model_kwargs_from_args(args, step_name)
        max_retries, retry_delay, continue_on_error = parse_retry_params_from_args(args)
        repeat_while, repeat_until, max_iterations = parse_loop_params_from_args(args, step_name)

        steps.append(
            StepDirective(
                name=step_name,
                content=step_content,
                assign_var=assign_var,
                model_kwargs=model_kwargs,
                max_retries=max_retries,
                retry_delay=retry_delay,
                continue_on_error=continue_on_error,
                timeout=timeout,
                repeat_while=repeat_while,
                repeat_until=repeat_until,
                max_iterations=max_iterations,
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

    # Validate max_turns is positive
    if agent.config.max_turns <= 0:
        errors.append("max_turns must be positive")

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

    # Basic validation
    errors = validate_agent(agent)
    if errors:
        return False, "; ".join(errors)

    # Validate model if specified
    if agent.config.model:
        is_valid, error = validate_model_string(agent.config.model)
        if not is_valid:
            return False, error

    # Validate tool syntax
    if agent.config.tools:
        is_valid, error = validate_tool_specs(agent.config.tools)
        if not is_valid:
            return False, error

    # Check template rendering with minimal context
    from .renderer import AgentRenderer

    renderer = AgentRenderer()
    try:
        test_context = build_validation_test_context(agent)
        add_mock_step_variables(test_context, agent.content)
        add_mock_tool_variables(test_context, agent.content)
        renderer.render(agent.content, test_context)
    except Exception as e:
        return False, f"Template validation failed: {e}"

    return True, "Agent is valid"


def parse_model_kwargs_from_args(args: str, step_name: str) -> dict[str, Any]:
    """Parse model kwargs from step directive arguments.

    Extracts temperature, max_tokens, top_p, penalties, response_format,
    and reasoning_effort from directive args.

    Args:
        args: Raw directive arguments string
        step_name: Name of step (for error messages)

    Returns:
        Dict of model kwargs to pass to LLM

    Raises:
        ValueError: If JSON parsing fails for response_format
    """
    import json

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
            raise ValueError(f"Invalid JSON in response_format for step '{step_name}': {e}") from e

    # Parse numeric parameters
    temperature = _parse_directive_attribute(args, "temperature", r"[0-9.]+", float)
    if temperature is not None:
        model_kwargs["temperature"] = temperature

    max_tokens = _parse_directive_attribute(args, "max_tokens", r"[0-9]+", int)
    if max_tokens is not None:
        model_kwargs["max_tokens"] = max_tokens

    top_p = _parse_directive_attribute(args, "top_p", r"[0-9.]+", float)
    if top_p is not None:
        model_kwargs["top_p"] = top_p

    freq_penalty = _parse_directive_attribute(args, "frequency_penalty", r"[0-9.]+", float)
    if freq_penalty is not None:
        model_kwargs["frequency_penalty"] = freq_penalty

    pres_penalty = _parse_directive_attribute(args, "presence_penalty", r"[0-9.]+", float)
    if pres_penalty is not None:
        model_kwargs["presence_penalty"] = pres_penalty

    # Parse reasoning_effort (for o1, o3, Claude extended thinking)
    reasoning_effort = _parse_directive_attribute(args, "reasoning_effort", r"low|medium|high")
    if reasoning_effort:
        model_kwargs["reasoning_effort"] = reasoning_effort

    return model_kwargs


def parse_retry_params_from_args(args: str) -> tuple[int, float, bool]:
    """Parse retry parameters from step directive arguments.

    Args:
        args: Raw directive arguments string

    Returns:
        Tuple of (max_retries, retry_delay, continue_on_error)
    """
    max_retries = _parse_directive_attribute(args, "max_retries", r"[0-9]+", int, default=0)
    retry_delay = _parse_directive_attribute(args, "retry_delay", r"[0-9.]+", float, default=0.0)
    continue_str = _parse_directive_attribute(args, "continue_on_error", r"true|false", default="false")
    continue_on_error = continue_str.lower() == "true"

    return max_retries, retry_delay, continue_on_error


def parse_loop_params_from_args(args: str, step_name: str) -> tuple[Optional[str], Optional[str], int]:
    """Parse loop control parameters from step directive arguments.

    Args:
        args: Raw directive arguments string
        step_name: Name of step (for error messages)

    Returns:
        Tuple of (repeat_while, repeat_until, max_iterations)

    Raises:
        ValueError: If both repeat_while and repeat_until are specified
    """
    repeat_while = _parse_directive_attribute(args, "repeat_while", r".+?")
    repeat_until = _parse_directive_attribute(args, "repeat_until", r".+?")
    max_iterations = _parse_directive_attribute(args, "max_iterations", r"[0-9]+", int, default=10)

    # Validate: cannot specify both
    if repeat_while and repeat_until:
        raise ValueError(f"Step '{step_name}' cannot specify both repeat_while and repeat_until. Use one or the other.")

    return repeat_while, repeat_until, max_iterations


def find_json_object_in_string(text: str, start_keyword: str) -> tuple[int, int]:
    """Find a JSON object in a string starting after a keyword.

    Handles nested braces correctly.

    Args:
        text: Text to search in
        start_keyword: Keyword before JSON (e.g., "args=")

    Returns:
        Tuple of (json_start_pos, json_end_pos)

    Raises:
        ValueError: If JSON object not found or has unmatched braces
    """
    keyword_pos = text.find(start_keyword)
    if keyword_pos == -1:
        raise ValueError(f"Keyword '{start_keyword}' not found in text")

    # Find the opening brace after keyword
    json_start = text.find("{", keyword_pos)
    if json_start == -1:
        raise ValueError(f"No JSON object found after '{start_keyword}'")

    # Find matching closing brace (handle nested braces)
    brace_count = 0
    json_end = json_start
    for i in range(json_start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                json_end = i + 1
                break

    if brace_count != 0:
        raise ValueError(f"Unmatched braces in JSON object after '{start_keyword}'")

    return json_start, json_end


def extract_and_parse_json_args(raw_args: str, tool_name: str) -> dict[str, Any]:
    """Extract and parse JSON args from tool directive arguments.

    Args:
        raw_args: Raw directive arguments string
        tool_name: Name of tool (for error messages)

    Returns:
        Parsed args dict

    Raises:
        ValueError: If args missing, invalid JSON, or has syntax errors
    """
    import json

    # Find args= and extract JSON object
    args_start = raw_args.find("args=")
    if args_start == -1:
        raise ValueError(f"Tool directive missing required 'args' attribute: {raw_args}")

    try:
        json_start, json_end = find_json_object_in_string(raw_args, "args=")
    except ValueError as e:
        raise ValueError(f"Tool directive args must be a JSON object: {raw_args}") from e

    # Parse JSON args
    try:
        args_json_str = raw_args[json_start:json_end]
        return json.loads(args_json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in tool directive args for '{tool_name}': {e}") from e


def build_validation_test_context(agent, include_prefetch: bool = True) -> dict[str, Any]:
    """Build minimal test context for agent validation.

    Args:
        agent: Agent object to build context for
        include_prefetch: Whether to include mock prefetch variables

    Returns:
        Dict of context variables for template rendering
    """
    test_context = {
        "user_prompt": "test",
        "task_summary": "## Current Tasks\nNo tasks yet.",
        "is_interactive": False,
        "tools": agent.config.tools or [],
        "text_mode": agent.config.text_mode,
        "is_subagent": False,
        "parent_agent": None,
        "chat_history": [],
    }

    # Add mock prefetch variables
    if include_prefetch and agent.config.prefetch:
        for item in agent.config.prefetch:
            assign_name = item.get("assign")
            if assign_name:
                test_context[assign_name] = "mock_prefetch_data"

    return test_context


def add_mock_step_variables(test_context: dict[str, Any], agent_content: str) -> None:
    """Add mock variables for step assignments to test context.

    Modifies test_context in place.

    Args:
        test_context: Context dict to modify
        agent_content: Agent content to extract steps from
    """
    if has_step_directives(agent_content):
        try:
            preamble, steps = extract_step_directives(agent_content)
            for step in steps:
                if step.assign_var:
                    test_context[step.assign_var] = "mock_step_result"
        except Exception:
            # If step parsing fails, let it be caught by normal validation
            pass


def add_mock_tool_variables(test_context: dict[str, Any], agent_content: str) -> None:
    """Add mock variables for tool directive assignments to test context.

    Modifies test_context in place.

    Args:
        test_context: Context dict to modify
        agent_content: Agent content to extract tool directives from
    """
    try:
        directives = extract_tool_directives(agent_content)
        for directive in directives:
            if directive.assign_var:
                test_context[directive.assign_var] = "mock_tool_result"
    except Exception:
        # If tool directive parsing fails, let it be caught by normal validation
        pass


def validate_model_string(model: str) -> tuple[bool, Optional[str]]:
    """Validate model string format.

    Args:
        model: Model string to validate (e.g., "openai:gpt-4")

    Returns:
        Tuple of (is_valid, error_message). Error is None if valid.
    """
    try:
        from .models import parse_model_string

        parse_model_string(model)
        return True, None
    except Exception as e:
        return False, f"Model validation failed: {e}"


def validate_tool_specs(tool_specs: list[str]) -> tuple[bool, Optional[str]]:
    """Validate tool specifications are syntactically correct.

    Args:
        tool_specs: List of tool specification strings

    Returns:
        Tuple of (is_valid, error_message). Error is None if valid.
    """
    try:
        # Import to ensure module can be loaded
        from .tools import expand_tool_specs  # noqa: F401

        # Check syntax is valid
        for tool_spec in tool_specs:
            if not isinstance(tool_spec, str):
                return False, f"Tool specification must be string: {tool_spec}"

        return True, None
    except Exception as e:
        return False, f"Tool import error: {e}"
