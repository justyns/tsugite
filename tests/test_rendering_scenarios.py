"""Test rendering scenarios for Tsugite agents."""

from unittest.mock import patch

import pytest

from tsugite.agent_runner import execute_prefetch, validate_agent_execution
from tsugite.renderer import AgentRenderer


class TestBasicRendering:
    """Test basic template rendering functionality."""

    def test_simple_template_rendering(self):
        """Test basic template rendering with user_prompt."""
        renderer = AgentRenderer()
        template = "Hello {{ user_prompt }}!"
        result = renderer.render(template, {"user_prompt": "world"})
        assert result == "Hello world!"

    def test_empty_prompt_rendering(self):
        """Test rendering with empty prompt."""
        renderer = AgentRenderer()
        template = "Task: {{ user_prompt }}"
        result = renderer.render(template, {"user_prompt": ""})
        assert result == "Task: "

    def test_no_user_prompt_template(self):
        """Test template that doesn't use user_prompt variable."""
        renderer = AgentRenderer()
        template = "System message without user input"
        result = renderer.render(template, {"user_prompt": "ignored"})
        assert result == "System message without user input"


class TestHelperFunctions:
    """Test Jinja2 helper functions."""

    def test_now_function(self):
        """Test now() helper function."""
        renderer = AgentRenderer()
        template = "Current time: {{ now() }}"
        result = renderer.render(template)
        assert "Current time:" in result
        assert len(result) > len("Current time: ")

    def test_today_function(self):
        """Test today() helper function."""
        renderer = AgentRenderer()
        template = "Today is: {{ today() }}"
        result = renderer.render(template)
        assert "Today is:" in result
        assert len(result.split("-")) >= 3  # Should be YYYY-MM-DD format

    def test_slugify_function(self):
        """Test slugify() helper function."""
        renderer = AgentRenderer()
        template = "Slug: {{ slugify('Hello World!') }}"
        result = renderer.render(template)
        assert result == "Slug: hello-world"

    def test_slugify_filter(self):
        """Test slugify as Jinja2 filter."""
        renderer = AgentRenderer()
        template = "Slug: {{ 'Hello World!' | slugify }}"
        result = renderer.render(template)
        assert result == "Slug: hello-world"

    def test_env_access(self):
        """Test environment variable access."""
        renderer = AgentRenderer()
        template = "Home: {{ env.HOME }}"
        with patch.dict("os.environ", {"HOME": "/test/home"}):
            # Need to create a new renderer to pick up the patched env
            renderer = AgentRenderer()
            result = renderer.render(template)
            assert result == "Home: /test/home"


class TestPrefetchRendering:
    """Test rendering with prefetch variables."""

    def test_simple_prefetch_variable(self):
        """Test template using prefetch variable."""
        renderer = AgentRenderer()
        template = "Content: {{ prefetch_data }}"
        context = {"prefetch_data": "test content", "user_prompt": "task"}
        result = renderer.render(template, context)
        assert result == "Content: test content"

    def test_prefetch_with_slicing(self):
        """Test prefetch variable with slicing operation."""
        renderer = AgentRenderer()
        template = "Preview: {{ readme_content[:50] }}..."
        content = "This is a long README file content that should be truncated"
        context = {"readme_content": content, "user_prompt": "task"}
        result = renderer.render(template, context)
        # Test that slicing works and content is properly truncated
        assert result.startswith("Preview: This is a long README")
        assert result.endswith("...")
        # Check that exactly 50 characters were taken from content
        slice_part = result[9:-3]  # Remove "Preview: " and "..."
        assert len(slice_part) == 50
        assert slice_part == content[:50]

    def test_multiple_prefetch_variables(self):
        """Test template with multiple prefetch variables."""
        renderer = AgentRenderer()
        template = """
Summary:
- Config: {{ config_data }}
- Notes: {{ notes_data }}
- Task: {{ user_prompt }}
        """.strip()
        context = {
            "config_data": "app config",
            "notes_data": "user notes",
            "user_prompt": "analyze data",
        }
        result = renderer.render(template, context)
        expected = """Summary:
- Config: app config
- Notes: user notes
- Task: analyze data"""
        assert result == expected


class TestAgentValidation:
    """Test agent validation with various scenarios."""

    def test_validate_simple_agent(self, temp_dir):
        """Test validation of simple agent without prefetch."""
        agent_content = """---
name: simple_test
model: openai:gpt-4o-mini
tools: []
---
# Task
{{ user_prompt }}
"""
        agent_file = temp_dir / "simple_agent.md"
        agent_file.write_text(agent_content)

        is_valid, message = validate_agent_execution(agent_file)
        assert is_valid, f"Validation failed: {message}"

    def test_validate_agent_with_prefetch(self, temp_dir):
        """Test validation of agent with prefetch variables."""
        agent_content = """---
name: prefetch_test
model: openai:gpt-4o-mini
tools: [read_file]
prefetch:
  - tool: read_file
    args: { path: "test.txt" }
    assign: file_content
---
# Task
Content: {{ file_content[:100] }}
Task: {{ user_prompt }}
"""
        agent_file = temp_dir / "prefetch_agent.md"
        agent_file.write_text(agent_content)

        is_valid, message = validate_agent_execution(agent_file)
        assert is_valid, f"Validation failed: {message}"

    def test_validate_agent_with_multiple_prefetch(self, temp_dir):
        """Test validation with multiple prefetch variables."""
        agent_content = """---
name: multi_prefetch_test
model: openai:gpt-4o-mini
tools: [read_file]
prefetch:
  - tool: read_file
    args: { path: "config.json" }
    assign: config
  - tool: read_file
    args: { path: "notes.md" }
    assign: notes
---
# Task
Config: {{ config }}
Notes: {{ notes }}
Task: {{ user_prompt }}
"""
        agent_file = temp_dir / "multi_prefetch_agent.md"
        agent_file.write_text(agent_content)

        is_valid, message = validate_agent_execution(agent_file)
        assert is_valid, f"Validation failed: {message}"

    def test_validate_agent_with_invalid_template(self, temp_dir):
        """Test validation fails for invalid template syntax."""
        agent_content = """---
name: invalid_test
model: openai:gpt-4o-mini
tools: []
---
# Task
{{ undefined_variable }}
"""
        agent_file = temp_dir / "invalid_agent.md"
        agent_file.write_text(agent_content)

        is_valid, message = validate_agent_execution(agent_file)
        assert not is_valid
        assert "undefined_variable" in message.lower()


class TestPrefetchExecution:
    """Test prefetch execution functionality."""

    @patch("tsugite.agent_runner.call_tool")
    def test_execute_simple_prefetch(self, mock_call_tool):
        """Test executing simple prefetch configuration."""
        mock_call_tool.return_value = "test file content"

        prefetch_config = [
            {
                "tool": "read_file",
                "args": {"path": "test.txt"},
                "assign": "file_content",
            }
        ]

        result = execute_prefetch(prefetch_config)

        assert result == {"file_content": "test file content"}
        mock_call_tool.assert_called_once_with("read_file", path="test.txt")

    @patch("tsugite.agent_runner.call_tool")
    def test_execute_multiple_prefetch(self, mock_call_tool):
        """Test executing multiple prefetch tools."""
        mock_call_tool.side_effect = ["config data", "notes data"]

        prefetch_config = [
            {"tool": "read_file", "args": {"path": "config.json"}, "assign": "config"},
            {"tool": "read_file", "args": {"path": "notes.md"}, "assign": "notes"},
        ]

        result = execute_prefetch(prefetch_config)

        expected = {"config": "config data", "notes": "notes data"}
        assert result == expected
        assert mock_call_tool.call_count == 2

    @patch("tsugite.agent_runner.call_tool")
    def test_prefetch_with_tool_failure(self, mock_call_tool):
        """Test prefetch handles tool failures gracefully."""
        mock_call_tool.side_effect = Exception("Tool failed")

        prefetch_config = [
            {
                "tool": "read_file",
                "args": {"path": "missing.txt"},
                "assign": "file_content",
            }
        ]

        result = execute_prefetch(prefetch_config)

        assert result == {"file_content": None}

    def test_prefetch_with_missing_assign(self):
        """Test prefetch ignores items without assign field."""
        prefetch_config = [
            {
                "tool": "read_file",
                "args": {"path": "test.txt"},
                # Missing "assign" field
            }
        ]

        result = execute_prefetch(prefetch_config)
        assert result == {}


class TestComplexRenderingScenarios:
    """Test complex rendering scenarios."""

    def test_conditional_rendering(self):
        """Test conditional rendering with Jinja2."""
        renderer = AgentRenderer()
        template = """
{% if user_prompt %}
Task: {{ user_prompt }}
{% else %}
No task specified
{% endif %}
        """.strip()

        # With prompt
        result1 = renderer.render(template, {"user_prompt": "test task"})
        assert "Task: test task" in result1

        # Without prompt
        result2 = renderer.render(template, {"user_prompt": ""})
        assert "No task specified" in result2

    def test_loop_rendering(self):
        """Test loop rendering with prefetch data."""
        renderer = AgentRenderer()
        template = """
Files:
{% for file in file_list %}
- {{ file }}
{% endfor %}
        """.strip()

        context = {
            "file_list": ["file1.txt", "file2.txt", "file3.txt"],
            "user_prompt": "list files",
        }
        result = renderer.render(template, context)

        assert "- file1.txt" in result
        assert "- file2.txt" in result
        assert "- file3.txt" in result

    def test_nested_data_access(self):
        """Test accessing nested data structures."""
        renderer = AgentRenderer()
        template = """
User: {{ user_data.name }}
Email: {{ user_data.email }}
Settings: {{ user_data.settings.theme }}
        """.strip()

        context = {
            "user_data": {
                "name": "John Doe",
                "email": "john@example.com",
                "settings": {"theme": "dark"},
            },
            "user_prompt": "show user info",
        }
        result = renderer.render(template, context)

        assert "User: John Doe" in result
        assert "Email: john@example.com" in result
        assert "Settings: dark" in result

    def test_filter_combinations(self):
        """Test combining multiple Jinja2 filters."""
        renderer = AgentRenderer()
        template = "Title: {{ title | upper | replace('_', ' ') }}"

        context = {"title": "hello_world_test", "user_prompt": "format title"}
        result = renderer.render(template, context)

        assert result == "Title: HELLO WORLD TEST"


class TestErrorHandling:
    """Test error handling in rendering scenarios."""

    def test_undefined_variable_error(self):
        """Test that undefined variables raise proper errors."""
        renderer = AgentRenderer()
        template = "Value: {{ undefined_var }}"

        with pytest.raises(ValueError, match="Template rendering failed"):
            renderer.render(template, {"user_prompt": "test"})

    def test_invalid_filter_error(self):
        """Test that invalid filters raise proper errors."""
        renderer = AgentRenderer()
        template = "Value: {{ 'test' | nonexistent_filter }}"

        with pytest.raises(ValueError, match="Template rendering failed"):
            renderer.render(template, {"user_prompt": "test"})

    def test_syntax_error_handling(self):
        """Test handling of template syntax errors."""
        renderer = AgentRenderer()
        template = "Value: {{ invalid syntax"  # Missing closing braces

        with pytest.raises(ValueError, match="Template rendering failed"):
            renderer.render(template, {"user_prompt": "test"})


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path
