"""Test CLI rendering commands and features."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from tsugite.tsugite import app


class TestCliRenderCommand:
    """Test the CLI render command."""

    def setup_method(self):
        """Set up test runner."""
        self.runner = CliRunner()

    def test_render_simple_agent(self, temp_dir):
        """Test rendering a simple agent."""
        agent_content = """---
name: simple_test
model: openai:gpt-4o-mini
tools: []
---
# Task
Hello {{ user_prompt }}!
"""
        agent_file = temp_dir / "simple.md"
        agent_file.write_text(agent_content)

        result = self.runner.invoke(app, ["render", str(agent_file), "world"])

        assert result.exit_code == 0
        assert "Hello world!" in result.stdout

    def test_render_with_empty_prompt(self, temp_dir):
        """Test rendering with empty prompt (optional)."""
        agent_content = """---
name: no_prompt_test
model: openai:gpt-4o-mini
tools: []
---
# System Message
This agent doesn't need user input.
"""
        agent_file = temp_dir / "no_prompt.md"
        agent_file.write_text(agent_content)

        result = self.runner.invoke(app, ["render", str(agent_file)])

        assert result.exit_code == 0
        assert "This agent doesn't need user input." in result.stdout

    def test_render_with_helper_functions(self, temp_dir):
        """Test rendering with helper functions."""
        agent_content = """---
name: helpers_test
model: openai:gpt-4o-mini
tools: []
---
# Context
- Time: {{ now() }}
- Date: {{ today() }}
- Slug: {{ "Hello World!" | slugify }}

# Task
{{ user_prompt }}
"""
        agent_file = temp_dir / "helpers.md"
        agent_file.write_text(agent_content)

        result = self.runner.invoke(app, ["render", str(agent_file), "test task"])

        assert result.exit_code == 0
        assert "Slug: hello-world" in result.stdout
        assert "test task" in result.stdout

    @patch("tsugite.agent_runner.call_tool")
    def test_render_with_prefetch(self, mock_call_tool, temp_dir):
        """Test rendering agent with prefetch tools."""
        mock_call_tool.return_value = "mock file content"

        agent_content = """---
name: prefetch_test
model: openai:gpt-4o-mini
tools: [read_file]
prefetch:
  - tool: read_file
    args: { path: "test.txt" }
    assign: file_content
---
# Content Preview
{{ file_content[:20] }}...

# Task
{{ user_prompt }}
"""
        agent_file = temp_dir / "prefetch.md"
        agent_file.write_text(agent_content)

        result = self.runner.invoke(app, ["render", str(agent_file), "analyze content"])

        assert result.exit_code == 0
        assert "mock file content" in result.stdout
        assert "analyze content" in result.stdout
        mock_call_tool.assert_called_once_with("read_file", path="test.txt")

    def test_render_nonexistent_file(self):
        """Test render command with nonexistent file."""
        result = self.runner.invoke(app, ["render", "/nonexistent/agent.md", "test"])

        assert result.exit_code == 1
        assert "Agent file not found" in result.stdout

    def test_render_non_markdown_file(self, temp_dir):
        """Test render command with non-markdown file."""
        text_file = temp_dir / "not_agent.txt"
        text_file.write_text("Not an agent file")

        result = self.runner.invoke(app, ["render", str(text_file), "test"])

        assert result.exit_code == 1
        assert "must be a .md file" in result.stdout

    def test_render_invalid_template(self, temp_dir):
        """Test render command with invalid template."""
        agent_content = """---
name: invalid_test
model: openai:gpt-4o-mini
tools: []
---
# Task
{{ undefined_variable }}
"""
        agent_file = temp_dir / "invalid.md"
        agent_file.write_text(agent_content)

        result = self.runner.invoke(app, ["render", str(agent_file), "test"])

        assert result.exit_code == 1
        assert "Render error" in result.stdout


class TestCliRunCommand:
    """Test the CLI run command with new features."""

    def setup_method(self):
        """Set up test runner."""
        self.runner = CliRunner()

    def test_run_with_optional_prompt(self, temp_dir):
        """Test run command with optional prompt."""
        agent_content = """---
name: optional_prompt_test
model: openai:gpt-4o-mini
tools: []
---
# System Message
Static content without user prompt.
"""
        agent_file = temp_dir / "optional_prompt.md"
        agent_file.write_text(agent_content)

        # Mock the agent execution to avoid model calls
        with patch("tsugite.tsugite.run_agent") as mock_run:
            mock_run.return_value = "Agent completed successfully"

            result = self.runner.invoke(app, ["run", str(agent_file)])

            assert result.exit_code == 0
            # Verify empty prompt was passed
            mock_run.assert_called_once()
            call_args = mock_run.call_args[1]  # keyword arguments
            assert call_args["prompt"] == ""

    def test_run_with_debug_flag(self, temp_dir):
        """Test run command with debug flag."""
        agent_content = """---
name: debug_test
model: openai:gpt-4o-mini
tools: []
---
# Task
Debug test: {{ user_prompt }}
"""
        agent_file = temp_dir / "debug.md"
        agent_file.write_text(agent_content)

        with patch("tsugite.tsugite.run_agent") as mock_run:
            mock_run.return_value = "Debug test completed"

            result = self.runner.invoke(app, ["run", str(agent_file), "test task", "--debug"])

            assert result.exit_code == 0
            # Verify debug flag was passed
            mock_run.assert_called_once()
            call_args = mock_run.call_args[1]
            assert call_args["debug"] is True

    def test_run_validation_with_prefetch(self, temp_dir):
        """Test run command validation with prefetch variables."""
        agent_content = """---
name: validation_test
model: openai:gpt-4o-mini
tools: [read_file]
prefetch:
  - tool: read_file
    args: { path: "config.json" }
    assign: config_data
---
# Task
Config: {{ config_data }}
Task: {{ user_prompt }}
"""
        agent_file = temp_dir / "validation.md"
        agent_file.write_text(agent_content)

        with patch("tsugite.tsugite.run_agent") as mock_run:
            mock_run.side_effect = Exception("Model execution failed")

            result = self.runner.invoke(app, ["run", str(agent_file), "test task"])

            # Should pass validation (not exit due to template error)
            # but fail at execution (expected with mock)
            assert "Starting agent execution" in result.stdout


class TestDebugOutput:
    """Test debug output functionality."""

    def test_debug_output_in_agent_runner(self, capsys):
        """Test debug output shows rendered prompt."""
        from tsugite.agent_runner import run_agent
        from unittest.mock import patch

        # Create a minimal agent content
        agent_content = """---
name: debug_output_test
model: openai:gpt-4o-mini
tools: []
---

# Task
Debug test: {{ user_prompt }}"""

        with patch("pathlib.Path.read_text", return_value=agent_content):
            with patch("tsugite.agent_runner.get_smolagents_tools", return_value=[]):
                with patch("tsugite.agent_runner.get_model") as mock_get_model:
                    # Mock the smolagents CodeAgent to avoid actual execution
                    mock_agent = MagicMock()
                    mock_agent.run.return_value = "Test completed"

                    with patch("tsugite.agent_runner.CodeAgent", return_value=mock_agent):
                        result = run_agent(agent_path=Path("test.md"), prompt="test input", debug=True)

                        # Capture the debug output
                        captured = capsys.readouterr()

                        assert "DEBUG: Rendered Prompt" in captured.out
                        assert "Debug test: test input" in captured.out
                        assert "=" * 60 in captured.out  # Debug separator


class TestComplexScenarios:
    """Test complex rendering scenarios through CLI."""

    def setup_method(self):
        """Set up test runner."""
        self.runner = CliRunner()

    @patch("tsugite.agent_runner.call_tool")
    def test_multi_prefetch_rendering(self, mock_call_tool, temp_dir):
        """Test rendering with multiple prefetch tools."""
        mock_call_tool.side_effect = [
            '{"theme": "dark", "lang": "en"}',  # config.json
            "# Notes\nImportant notes here",  # notes.md
        ]

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
# Configuration
{{ config }}

# Notes Preview
{{ notes[:30] }}...

# Task
{{ user_prompt }}
"""
        agent_file = temp_dir / "multi_prefetch.md"
        agent_file.write_text(agent_content)

        result = self.runner.invoke(app, ["render", str(agent_file), "process data"])

        assert result.exit_code == 0
        assert '{"theme": "dark", "lang": "en"}' in result.stdout
        assert "Important notes here" in result.stdout
        assert "process data" in result.stdout

    def test_conditional_template_rendering(self, temp_dir):
        """Test conditional template rendering."""
        agent_content = """---
name: conditional_test
model: openai:gpt-4o-mini
tools: []
---
# System
{% if user_prompt %}
Task specified: {{ user_prompt }}
{% else %}
No specific task
{% endif %}

# Status
Ready to proceed.
"""
        agent_file = temp_dir / "conditional.md"
        agent_file.write_text(agent_content)

        # Test with prompt
        result1 = self.runner.invoke(app, ["render", str(agent_file), "analyze data"])
        assert result1.exit_code == 0
        assert "Task specified: analyze data" in result1.stdout

        # Test without prompt
        result2 = self.runner.invoke(app, ["render", str(agent_file)])
        assert result2.exit_code == 0
        assert "No specific task" in result2.stdout


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path
