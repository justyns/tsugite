"""Tests for tool system."""

import asyncio

import pytest

from tsugite.core.tools import Tool, create_tool_from_function, create_tool_from_tsugite


def test_tool_creation():
    """Test creating a Tool."""

    def add(a: int, b: int) -> int:
        return a + b

    tool = Tool(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=add,
    )

    assert tool.name == "add"
    assert tool.description == "Add two numbers"
    assert tool.parameters["type"] == "object"
    assert tool.function(5, 3) == 8


@pytest.mark.asyncio
async def test_tool_execute_sync():
    """Test executing a synchronous tool."""

    def multiply(x: int, y: int) -> int:
        return x * y

    tool = Tool(
        name="multiply",
        description="Multiply two numbers",
        parameters={},
        function=multiply,
    )

    result = await tool.execute(x=5, y=3)
    assert result == 15


@pytest.mark.asyncio
async def test_tool_execute_async():
    """Test executing an asynchronous tool."""

    async def async_add(a: int, b: int) -> int:
        await asyncio.sleep(0.01)  # Simulate async work
        return a + b

    tool = Tool(
        name="async_add",
        description="Add two numbers asynchronously",
        parameters={},
        function=async_add,
    )

    result = await tool.execute(a=10, b=20)
    assert result == 30


def test_tool_to_code_prompt():
    """Test generating code prompt from tool."""

    def greet(name: str, age: int = 25) -> str:
        """Greet a person with their name and age."""
        return f"Hello {name}, you are {age} years old"

    tool = Tool(
        name="greet",
        description="Greet a person with their name and age",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's name"},
                "age": {"type": "integer", "description": "Person's age"},
            },
            "required": ["name"],
        },
        function=greet,
    )

    prompt = tool.to_code_prompt()

    # Check that prompt contains function signature
    assert "def greet(" in prompt
    assert "name: str" in prompt
    assert "age: int" in prompt

    # Check that prompt contains description
    assert "Greet a person with their name and age" in prompt

    # Check that prompt contains parameter docs
    assert "Person's name" in prompt
    assert "Person's age" in prompt


def test_create_tool_from_function_basic():
    """Test creating tool from a simple function."""

    def subtract(a: int, b: int) -> int:
        """Subtract b from a"""
        return a - b

    tool = create_tool_from_function(subtract)

    assert tool.name == "subtract"
    assert tool.description == "Subtract b from a"
    assert "a" in tool.parameters["properties"]
    assert "b" in tool.parameters["properties"]
    assert tool.parameters["required"] == ["a", "b"]


def test_create_tool_from_function_with_defaults():
    """Test creating tool from function with default arguments."""

    def power(base: float, exponent: float = 2.0) -> float:
        """Raise base to exponent"""
        return base**exponent

    tool = create_tool_from_function(power)

    assert tool.name == "power"
    assert tool.description == "Raise base to exponent"
    assert "base" in tool.parameters["required"]
    assert "exponent" not in tool.parameters["required"]  # Has default


def test_create_tool_from_function_custom_name():
    """Test creating tool with custom name."""

    def my_func(x: int) -> int:
        """Do something"""
        return x * 2

    tool = create_tool_from_function(my_func, name="double", description="Double a number")

    assert tool.name == "double"
    assert tool.description == "Double a number"


@pytest.mark.asyncio
async def test_create_tool_from_function_async():
    """Test creating tool from async function."""

    async def fetch_data(url: str) -> str:
        """Fetch data from URL"""
        await asyncio.sleep(0.01)
        return f"Data from {url}"

    tool = create_tool_from_function(fetch_data)

    assert tool.name == "fetch_data"
    result = await tool.execute(url="http://example.com")
    assert "Data from http://example.com" in result


def test_create_tool_from_tsugite():
    """Test creating tool from tsugite registry."""
    from tsugite.tools import tool as tool_decorator

    # Register a mock tool for testing
    @tool_decorator
    def test_mock_tool(text: str) -> str:
        """Mock tool for testing"""
        return f"Processed: {text}"

    # Create Tool from tsugite registry
    tool = create_tool_from_tsugite("test_mock_tool")

    assert tool.name == "test_mock_tool"
    assert tool.description == "Mock tool for testing"
    assert "text" in tool.parameters["properties"]


@pytest.mark.asyncio
async def test_create_tool_from_tsugite_execution():
    """Test executing a tool created from tsugite registry."""
    from tsugite.tools import tool as tool_decorator

    # Register a mock tool for testing
    @tool_decorator
    def test_echo(message: str) -> str:
        """Echo the message"""
        return f"Echo: {message}"

    # Create and execute tool
    tool = create_tool_from_tsugite("test_echo")
    result = await tool.execute(message="Hello World")

    assert "Echo: Hello World" in result


def test_tool_parameters_json_schema():
    """Test that tool parameters are valid JSON schema."""

    def complex_function(name: str, items: list, config: dict, count: int = 10) -> str:
        """Complex function with various types"""
        return "result"

    tool = create_tool_from_function(complex_function)

    # Check JSON schema structure
    assert tool.parameters["type"] == "object"
    assert "properties" in tool.parameters
    assert "required" in tool.parameters

    # Check type mapping
    assert tool.parameters["properties"]["name"]["type"] == "string"
    assert tool.parameters["properties"]["items"]["type"] == "array"
    assert tool.parameters["properties"]["config"]["type"] == "object"
    assert tool.parameters["properties"]["count"]["type"] == "integer"


def test_tool_to_code_prompt_formatting():
    """Test that code prompt is properly formatted for LLM."""

    def search_web(query: str, max_results: int = 10) -> list:
        """Search the web for information

        Args:
            query: Search query string
            max_results: Maximum number of results to return
        """
        return []

    tool = Tool(
        name="search_web",
        description="Search the web for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string"},
                "max_results": {"type": "integer", "description": "Maximum number of results to return"},
            },
            "required": ["query"],
        },
        function=search_web,
    )

    prompt = tool.to_code_prompt()

    # Check Python syntax
    assert prompt.startswith("def search_web(")
    assert "-> Any:" in prompt or "-> list:" in prompt
    assert '"""' in prompt  # Has docstring

    # Check that it's valid Python (can be parsed)
    compile(prompt, "<string>", "exec")


def test_create_tool_from_function_unmapped_types():
    """Test creating tool from function with unmapped type annotations.

    When a type annotation can't be mapped to JSON Schema (e.g., datetime, custom classes),
    the schema should omit the type constraint rather than using an invalid "Any" type.
    """
    from datetime import datetime

    def schedule_task(task_name: str, when: datetime, priority: int = 5) -> str:
        """Schedule a task for a specific time"""
        return f"Scheduled {task_name}"

    tool = create_tool_from_function(schedule_task)

    # Check basic structure
    assert tool.name == "schedule_task"
    assert tool.description == "Schedule a task for a specific time"

    # Check that known types are mapped correctly
    assert tool.parameters["properties"]["task_name"]["type"] == "string"
    assert tool.parameters["properties"]["priority"]["type"] == "integer"

    # Check that unmapped type (datetime) has no type constraint
    # This is valid JSON Schema - it just doesn't constrain the type
    assert "when" in tool.parameters["properties"]
    when_schema = tool.parameters["properties"]["when"]
    assert "type" not in when_schema or when_schema == {}

    # Check required fields
    assert "task_name" in tool.parameters["required"]
    assert "when" in tool.parameters["required"]
    assert "priority" not in tool.parameters["required"]  # Has default


def test_tool_signature_keyword_only_parameters():
    """Test that tool signatures use keyword-only parameters.

    This ensures LLMs see the correct calling convention (keyword arguments only).
    """

    def search_files(pattern: str, path: str = ".") -> str:
        """Search for pattern in files"""
        return "results"

    tool = Tool(
        name="search_files",
        description="Search for pattern in files",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Search pattern"},
                "path": {"type": "string", "description": "Directory to search"},
            },
            "required": ["pattern"],
        },
        function=search_files,
    )

    prompt = tool.to_code_prompt()

    # Verify signature uses keyword-only parameters (has * marker)
    assert "def search_files(*, " in prompt, "Signature should include * for keyword-only params"

    # Verify parameters are after the *
    assert "*, pattern: str, path: str)" in prompt

    # Verify it's valid Python
    compile(prompt, "<string>", "exec")


def test_tool_usage_examples_in_docstring():
    """Test that tool docstrings include usage examples with keyword arguments.

    This helps LLMs learn the correct calling convention by example.
    """

    def calculate(x: int, y: int, operation: str = "add") -> int:
        """Perform calculation"""
        return 0

    tool = Tool(
        name="calculate",
        description="Perform calculation on two numbers",
        parameters={
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "First number"},
                "y": {"type": "integer", "description": "Second number"},
                "operation": {"type": "string", "description": "Operation to perform"},
            },
            "required": ["x", "y"],
        },
        function=calculate,
    )

    prompt = tool.to_code_prompt()

    # Verify usage example section exists
    assert "Usage:" in prompt, "Docstring should include Usage section"

    # Verify example shows keyword argument syntax
    assert "calculate(" in prompt
    assert "x=" in prompt, "Example should use keyword argument syntax (x=...)"
    assert "y=" in prompt, "Example should use keyword argument syntax (y=...)"
    assert "operation=" in prompt, "Example should use keyword argument syntax (operation=...)"

    # Verify example is in the docstring (between triple quotes)
    docstring_start = prompt.find('"""')
    docstring_end = prompt.rfind('"""')
    docstring = prompt[docstring_start:docstring_end]

    assert "Usage:" in docstring, "Usage should be inside the docstring"
    assert "result = calculate(" in docstring, "Example should show result assignment"
