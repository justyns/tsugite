"""Tests for the Jinja2 template renderer."""

from datetime import datetime
from unittest.mock import patch

import pytest

from tsugite.renderer import (
    AgentRenderer,
    file_exists,
    is_dir,
    is_file,
    now,
    read_text,
    slugify,
    strip_ignored_sections,
    today,
)


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


class TestIgnoreSyntax:
    """Tests for the <!-- tsu:ignore --> directive."""

    def test_strip_single_ignore_block(self):
        """Test stripping a single ignore block."""
        content = """
Normal content
<!-- tsu:ignore -->
This is ignored
<!-- /tsu:ignore -->
More content
""".strip()

        result = strip_ignored_sections(content)

        assert "Normal content" in result
        assert "More content" in result
        assert "This is ignored" not in result
        assert "tsu:ignore" not in result

    def test_strip_multiple_ignore_blocks(self):
        """Test stripping multiple ignore blocks."""
        content = """
Start
<!-- tsu:ignore -->
First ignored block
<!-- /tsu:ignore -->
Middle
<!-- tsu:ignore -->
Second ignored block
<!-- /tsu:ignore -->
End
""".strip()

        result = strip_ignored_sections(content)

        assert "Start" in result
        assert "Middle" in result
        assert "End" in result
        assert "First ignored block" not in result
        assert "Second ignored block" not in result

    def test_strip_inline_ignore(self):
        """Test stripping inline ignore blocks."""
        content = "Before <!-- tsu:ignore -->ignored content<!-- /tsu:ignore --> after"

        result = strip_ignored_sections(content)

        assert "Before" in result
        assert "after" in result
        assert "ignored content" not in result

    def test_strip_empty_ignore_block(self):
        """Test stripping empty ignore blocks."""
        content = """
Normal content
<!-- tsu:ignore -->
<!-- /tsu:ignore -->
More content
""".strip()

        result = strip_ignored_sections(content)

        assert "Normal content" in result
        assert "More content" in result
        assert "tsu:ignore" not in result

    def test_strip_ignore_with_no_spaces(self):
        """Test stripping ignore blocks without spaces in tags."""
        content = """
Normal
<!--tsu:ignore-->
Ignored
<!--/tsu:ignore-->
After
""".strip()

        result = strip_ignored_sections(content)

        assert "Normal" in result
        assert "After" in result
        assert "Ignored" not in result

    def test_strip_ignore_with_extra_spaces(self):
        """Test stripping ignore blocks with extra spaces."""
        content = """
Normal
<!--   tsu:ignore   -->
Ignored
<!--   /tsu:ignore   -->
After
""".strip()

        result = strip_ignored_sections(content)

        assert "Normal" in result
        assert "After" in result
        assert "Ignored" not in result

    def test_ignore_block_with_jinja_variables(self):
        """Test that Jinja variables in ignored blocks are stripped."""
        content = """
Normal {{ variable }}
<!-- tsu:ignore -->
This has {{ jinja_var }} that should be ignored
{% if condition %}
This too
{% endif %}
<!-- /tsu:ignore -->
After
""".strip()

        result = strip_ignored_sections(content)

        assert "Normal {{ variable }}" in result
        assert "After" in result
        assert "jinja_var" not in result
        assert "condition" not in result

    def test_ignore_block_with_multiline_content(self):
        """Test stripping ignore blocks with multiple lines."""
        content = """
# Agent Documentation

<!-- tsu:ignore -->
## How to Use This Agent

This agent performs the following steps:
1. First step
2. Second step
3. Third step

Example usage:
```bash
tsugite run agent.md "task"
```
<!-- /tsu:ignore -->

# Actual Agent Instructions

Do the task: {{ user_prompt }}
""".strip()

        result = strip_ignored_sections(content)

        assert "# Agent Documentation" in result
        assert "# Actual Agent Instructions" in result
        assert "Do the task: {{ user_prompt }}" in result
        assert "How to Use This Agent" not in result
        assert "Example usage" not in result
        assert "tsugite run" not in result

    def test_renderer_with_ignore_blocks(self):
        """Test that AgentRenderer properly strips ignore blocks during rendering."""
        renderer = AgentRenderer()

        content = """
# Agent Task

<!-- tsu:ignore -->
DOCUMENTATION SECTION:
This agent is for testing.
Variables available: {{ user_prompt }}
<!-- /tsu:ignore -->

Task: {{ user_prompt }}

{% if mode == "verbose" %}
Verbose mode enabled
{% endif %}
""".strip()

        result = renderer.render(content, {"user_prompt": "test task", "mode": "verbose"})

        assert "# Agent Task" in result
        assert "Task: test task" in result
        assert "Verbose mode enabled" in result
        assert "DOCUMENTATION SECTION" not in result
        assert "This agent is for testing" not in result

    def test_no_ignore_blocks(self):
        """Test that content without ignore blocks passes through unchanged."""
        content = """
Normal content
With multiple lines
No ignore blocks here
""".strip()

        result = strip_ignored_sections(content)

        assert result == content

    def test_ignore_blocks_dont_affect_regular_comments(self):
        """Test that regular HTML comments are not affected."""
        content = """
Normal content
<!-- Regular HTML comment -->
<!-- Another comment -->
Still normal
""".strip()

        result = strip_ignored_sections(content)

        # Regular HTML comments should remain
        assert "<!-- Regular HTML comment -->" in result
        assert "<!-- Another comment -->" in result
        assert "Normal content" in result
        assert "Still normal" in result
