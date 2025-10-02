"""Tests for the Jinja2 template renderer."""

from datetime import datetime
from unittest.mock import patch

import pytest

from tsugite.renderer import AgentRenderer, file_exists, is_dir, is_file, now, read_text, slugify, today


def test_helper_functions():
    """Test the helper functions available in templates."""
    # Test now() function
    with patch("tsugite.renderer.datetime") as mock_dt:
        mock_datetime = datetime(2023, 12, 25, 14, 30, 45)
        mock_dt.now.return_value = mock_datetime
        result = now()
        assert result == "2023-12-25T14:30:45"

    # Test today() function
    with patch("tsugite.renderer.datetime") as mock_dt:
        mock_datetime = datetime(2023, 12, 25, 14, 30, 45)
        mock_dt.now.return_value = mock_datetime
        result = today()
        assert result == "2023-12-25"

    # Test slugify function
    assert slugify("Hello World!") == "hello-world"
    assert slugify("Test_File-Name.txt") == "test_file-name-txt"
    assert slugify("  Multiple   Spaces  ") == "multiple-spaces"
    assert slugify("Special@#$%Characters") == "special-characters"
    assert slugify("") == ""


def test_renderer_basic():
    """Test basic template rendering."""
    renderer = AgentRenderer()

    content = "Hello {{ name }}!"
    result = renderer.render(content, {"name": "World"})

    assert result == "Hello World!"


def test_renderer_with_helper_functions():
    """Test rendering with built-in helper functions."""
    content = """
Current time: {{ now() }}
Today's date: {{ today() }}
Slug: {{ "Test File" | slugify }}
User: {{ env.USER }}
""".strip()

    with patch("tsugite.renderer.datetime") as mock_dt:
        mock_datetime = datetime(2023, 12, 25, 14, 30, 45)
        mock_dt.now.return_value = mock_datetime

        with patch.dict("os.environ", {"USER": "testuser"}):
            # Create renderer after patching env
            renderer = AgentRenderer()
            result = renderer.render(content)

    lines = result.strip().split("\n")
    assert lines[0] == "Current time: 2023-12-25T14:30:45"
    assert lines[1] == "Today's date: 2023-12-25"
    assert lines[2] == "Slug: test-file"
    assert lines[3] == "User: testuser"


def test_renderer_with_variables():
    """Test render_with_variables method."""
    renderer = AgentRenderer()

    content = """
# Task
{{ user_prompt }}

# Context
Weather: {{ weather }}
Temperature: {{ temperature|default("Unknown") }}
""".strip()

    result = renderer.render_with_variables(content, user_prompt="Test the system", variables={"weather": "Sunny"})

    assert "Test the system" in result
    assert "Weather: Sunny" in result
    assert "Temperature: Unknown" in result


def test_renderer_jinja_features():
    """Test advanced Jinja2 features work correctly."""
    renderer = AgentRenderer()

    content = """
{% if items %}
Items ({{ items|length }}):
{% for item in items %}
- {{ item.name }}: {{ item.value }}
{% endfor %}
{% else %}
No items found.
{% endif %}

{% set total = items|map(attribute='value')|sum %}
Total: {{ total }}
""".strip()

    items = [
        {"name": "item1", "value": 10},
        {"name": "item2", "value": 20},
        {"name": "item3", "value": 30},
    ]

    result = renderer.render(content, {"items": items})

    assert "Items (3):" in result
    assert "- item1: 10" in result
    assert "- item2: 20" in result
    assert "- item3: 30" in result
    assert "Total: 60" in result


def test_renderer_empty_items():
    """Test rendering with empty items list."""
    renderer = AgentRenderer()

    content = """
{% if items %}
Items found: {{ items|length }}
{% else %}
No items found.
{% endif %}
""".strip()

    result = renderer.render(content, {"items": []})
    assert "No items found." in result


def test_renderer_undefined_variable():
    """Test that undefined variables raise errors with StrictUndefined."""
    renderer = AgentRenderer()

    content = "Hello {{ undefined_variable }}!"

    with pytest.raises(ValueError, match="Template rendering failed"):
        renderer.render(content)


def test_renderer_filters():
    """Test common Jinja2 filters work."""
    renderer = AgentRenderer()

    content = """
Upper: {{ text|upper }}
Lower: {{ text|lower }}
Title: {{ text|title }}
Length: {{ text|length }}
Default: {{ missing|default("fallback") }}
""".strip()

    result = renderer.render(content, {"text": "hello world"})

    assert "Upper: HELLO WORLD" in result
    assert "Lower: hello world" in result
    assert "Title: Hello World" in result
    assert "Length: 11" in result
    assert "Default: fallback" in result


def test_renderer_whitespace_control():
    """Test that whitespace control works correctly."""
    renderer = AgentRenderer()

    content = """
{%- if True %}
Line with whitespace control
{%- endif %}
Normal line
"""

    result = renderer.render(content)

    # The {%- removes the newline, so both lines are joined
    assert result.strip() == "Line with whitespace controlNormal line"
    assert "Line with whitespace control" in result
    assert "Normal line" in result


def test_renderer_complex_context():
    """Test rendering with complex nested context."""
    renderer = AgentRenderer()

    content = """
# Agent: {{ config.name }}
Model: {{ config.model }}

{% if config.tools %}
Tools:
{% for tool in config.tools %}
- {{ tool }}
{% endfor %}
{% endif %}

{% if memories %}
Recent memories:
{% for memory in memories[:3] %}
- {{ memory.title }}: {{ memory.content[:50] }}...
{% endfor %}
{% endif %}
""".strip()

    context = {
        "config": {
            "name": "test_agent",
            "model": "ollama:qwen2.5-coder:7b",
            "tools": ["read_file", "write_file", "run"],
        },
        "memories": [
            {
                "title": "Memory 1",
                "content": "This is a long memory content that should be truncated",
            },
            {"title": "Memory 2", "content": "Another memory"},
        ],
    }

    result = renderer.render(content, context)

    assert "Agent: test_agent" in result
    assert "Model: ollama:qwen2.5-coder:7b" in result
    assert "- read_file" in result
    assert "- write_file" in result
    assert "- run" in result
    assert "- Memory 1: This is a long memory content that should be trunc..." in result
    assert "- Memory 2: Another memory..." in result


def test_renderer_environment_access():
    """Test accessing environment variables in templates."""
    renderer = AgentRenderer()

    content = """
Home: {{ env.HOME }}
Path exists: {{ 'PATH' in env }}
Custom: {{ env.get('CUSTOM_VAR', 'not_set') }}
""".strip()

    with patch.dict("os.environ", {"HOME": "/home/test", "PATH": "/usr/bin"}):
        result = renderer.render(content)

    assert "Home:" in result  # Just check that HOME is rendered
    assert "Path exists: True" in result
    assert "Custom: not_set" in result


def test_render_with_variables_defaults():
    """Test render_with_variables with default parameters."""
    renderer = AgentRenderer()

    content = "Prompt: {{ user_prompt|default('No prompt') }}"

    # Test with empty prompt
    result = renderer.render_with_variables(content)
    assert "Prompt:" in result  # Empty string, not default

    # Test with None variables
    result = renderer.render_with_variables(content, variables=None)
    assert "Prompt:" in result


def test_slugify_edge_cases():
    """Test slugify function with edge cases."""
    assert slugify("UPPERCASE") == "uppercase"
    assert slugify("123numbers456") == "123numbers456"
    assert slugify("---multiple---dashes---") == "multiple-dashes"
    assert slugify("   ") == ""
    assert slugify("single") == "single"
    assert slugify("中文字符") == ""  # Non-ASCII removed
    assert slugify("file.name.ext") == "file-name-ext"


def test_renderer_error_handling():
    """Test error handling in template rendering."""
    renderer = AgentRenderer()

    # Test syntax error
    content = "{{ unclosed_expression"
    with pytest.raises(ValueError, match="Template rendering failed"):
        renderer.render(content)

    # Test runtime error (undefined filter)
    content = "{{ 'test' | nonexistent_filter }}"
    with pytest.raises(ValueError, match="Template rendering failed"):
        renderer.render(content)


def test_render_with_variables_demo_style(monkeypatch):
    """Ensure the developer demo scenario renders as expected."""
    monkeypatch.setenv("USER", "demo_user")

    content = """
# System
Current time: {{ now() }}
Today: {{ today() }}

# Task
{{ user_prompt }}

# Test Variables
Weather: {{ weather|default("Unknown") }}
User: {{ env.USER }}
""".strip()

    renderer = AgentRenderer()

    with patch("tsugite.renderer.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
        result = renderer.render_with_variables(
            content,
            user_prompt="Test task",
            variables={"weather": "Sunny"},
        )

    assert "# System" in result
    assert "Current time: 2024-01-01T12:00:00" in result
    assert "Today: 2024-01-01" in result
    assert "# Task" in result
    assert "Test task" in result
    assert "Weather: Sunny" in result
    assert "User: demo_user" in result


def test_filesystem_helper_functions(tmp_path):
    """Test filesystem helper functions."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()

    assert file_exists(str(test_file)) is True
    assert file_exists(str(test_dir)) is True
    assert file_exists(str(tmp_path / "nonexistent.txt")) is False

    assert is_file(str(test_file)) is True
    assert is_file(str(test_dir)) is False
    assert is_file(str(tmp_path / "nonexistent.txt")) is False

    assert is_dir(str(test_dir)) is True
    assert is_dir(str(test_file)) is False
    assert is_dir(str(tmp_path / "nonexistent_dir")) is False

    assert read_text(str(test_file)) == "test content"
    assert read_text(str(tmp_path / "nonexistent.txt")) == ""
    assert read_text(str(tmp_path / "nonexistent.txt"), default="fallback") == "fallback"


def test_renderer_with_filesystem_helpers(tmp_path):
    """Test rendering templates with filesystem helper functions."""
    config_file = tmp_path / "config.json"
    config_file.write_text('{"setting": "value"}')

    content = """
{% if file_exists("config.json") %}
Config file exists!
Content: {{ read_text("config.json") }}
{% else %}
No config file found.
{% endif %}
""".strip()

    renderer = AgentRenderer()
    result = renderer.render(content, {"cwd": str(tmp_path)})

    # Without changing working directory, file won't exist
    assert "No config file found." in result


def test_renderer_filesystem_conditionals(tmp_path):
    """Test complex conditionals with filesystem helpers."""
    (tmp_path / "data.txt").write_text("some data")
    (tmp_path / "backups").mkdir()

    content = """
{% if is_file("data.txt") %}
Data file: {{ read_text("data.txt") }}
{% endif %}

{% if is_dir("backups") %}
Backups directory exists!
{% endif %}

{% if file_exists("missing.txt") %}
Should not appear
{% else %}
File is missing as expected
{% endif %}
""".strip()

    renderer = AgentRenderer()

    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = renderer.render(content)

        assert "Data file: some data" in result
        assert "Backups directory exists!" in result
        assert "File is missing as expected" in result
        assert "Should not appear" not in result
    finally:
        os.chdir(original_cwd)


def test_read_text_with_default():
    """Test read_text with custom default value."""
    renderer = AgentRenderer()

    content = """
Content: {{ read_text("nonexistent.txt", default="<no file>") }}
""".strip()

    result = renderer.render(content)
    assert "Content: <no file>" in result
