"""Tests for <!-- tsu:tool --> directive functionality."""

from unittest.mock import patch

import pytest

from tsugite.agent_runner import execute_tool_directives
from tsugite.md_agents import extract_tool_directives


class TestToolDirectiveParsing:
    """Tests for parsing tool directives from content."""

    def test_extract_single_tool_directive(self):
        """Test extracting a single tool directive."""
        content = """
<!-- tsu:tool name="read_file" args={"path": "test.txt"} assign="data" -->
"""
        directives = extract_tool_directives(content)

        assert len(directives) == 1
        assert directives[0].name == "read_file"
        assert directives[0].args == {"path": "test.txt"}
        assert directives[0].assign_var == "data"

    def test_extract_multiple_tool_directives(self):
        """Test extracting multiple tool directives."""
        content = """
<!-- tsu:tool name="read_file" args={"path": "file1.txt"} assign="data1" -->
Some content
<!-- tsu:tool name="read_file" args={"path": "file2.txt"} assign="data2" -->
More content
"""
        directives = extract_tool_directives(content)

        assert len(directives) == 2
        assert directives[0].name == "read_file"
        assert directives[0].args == {"path": "file1.txt"}
        assert directives[0].assign_var == "data1"
        assert directives[1].name == "read_file"
        assert directives[1].args == {"path": "file2.txt"}
        assert directives[1].assign_var == "data2"

    def test_extract_tool_directive_with_complex_args(self):
        """Test extracting tool directive with complex JSON args."""
        content = """
<!-- tsu:tool name="fetch_json" args={"url": "http://example.com", "headers": {"auth": "token"}} assign="api_data" -->
"""
        directives = extract_tool_directives(content)

        assert len(directives) == 1
        assert directives[0].name == "fetch_json"
        assert directives[0].args == {"url": "http://example.com", "headers": {"auth": "token"}}
        assert directives[0].assign_var == "api_data"

    def test_tool_directive_missing_name_raises_error(self):
        """Test that directive without name raises ValueError."""
        content = """
<!-- tsu:tool args={"path": "test.txt"} assign="data" -->
"""
        with pytest.raises(ValueError, match="missing required 'name' attribute"):
            extract_tool_directives(content)

    def test_tool_directive_missing_args_raises_error(self):
        """Test that directive without args raises ValueError."""
        content = """
<!-- tsu:tool name="read_file" assign="data" -->
"""
        with pytest.raises(ValueError, match="missing required 'args' attribute"):
            extract_tool_directives(content)

    def test_tool_directive_missing_assign_raises_error(self):
        """Test that directive without assign raises ValueError."""
        content = """
<!-- tsu:tool name="read_file" args={"path": "test.txt"} -->
"""
        with pytest.raises(ValueError, match="missing required 'assign' attribute"):
            extract_tool_directives(content)

    def test_tool_directive_invalid_json_raises_error(self):
        """Test that directive with invalid JSON raises ValueError."""
        content = """
<!-- tsu:tool name="read_file" args={invalid json} assign="data" -->
"""
        with pytest.raises(ValueError, match="Invalid JSON"):
            extract_tool_directives(content)

    def test_empty_content_returns_empty_list(self):
        """Test that content without directives returns empty list."""
        content = """
Normal markdown content
No directives here
"""
        directives = extract_tool_directives(content)
        assert directives == []

    def test_tool_directive_with_no_spaces(self):
        """Test parsing directive without spaces in HTML comment."""
        content = """
<!--tsu:tool name="read_file" args={"path":"test.txt"} assign="data"-->
"""
        directives = extract_tool_directives(content)

        assert len(directives) == 1
        assert directives[0].name == "read_file"

    def test_tool_directive_with_extra_spaces(self):
        """Test parsing directive with extra spaces."""
        content = """
<!--   tsu:tool   name="read_file"   args={"path": "test.txt"}   assign="data"   -->
"""
        directives = extract_tool_directives(content)

        assert len(directives) == 1
        assert directives[0].name == "read_file"


class TestToolDirectiveExecution:
    """Tests for executing tool directives."""

    def test_execute_simple_tool_directive(self):
        """Test executing a single tool directive."""
        content = """
<!-- tsu:tool name="test_tool" args={"param": "value"} assign="result" -->
"""
        # Mock the call_tool function
        with patch("tsugite.agent_runner.runner.call_tool") as mock_call_tool:
            mock_call_tool.return_value = "Tool executed successfully"

            modified_content, context = execute_tool_directives(content)

            # Check that tool was called with correct args
            mock_call_tool.assert_called_once_with("test_tool", param="value")

            # Check that directive was replaced
            assert "tsu:tool" not in modified_content
            assert "Tool 'test_tool' executed" in modified_content

            # Check that context has the result
            assert "result" in context
            assert context["result"] == "Tool executed successfully"

    def test_execute_multiple_tool_directives(self):
        """Test executing multiple tool directives in sequence."""
        content = """
<!-- tsu:tool name="tool1" args={"key": "value1"} assign="data1" -->
<!-- tsu:tool name="tool2" args={"key": "value2"} assign="data2" -->
"""
        # Mock the call_tool function to return different results
        with patch("tsugite.agent_runner.runner.call_tool") as mock_call_tool:
            mock_call_tool.side_effect = ["Result 1", "Result 2"]

            modified_content, context = execute_tool_directives(content)

            # Check both directives were executed
            assert "data1" in context
            assert "data2" in context
            assert context["data1"] == "Result 1"
            assert context["data2"] == "Result 2"

    def test_tool_directive_failure_assigns_none(self):
        """Test that failed tool directive assigns None."""
        content = """
<!-- tsu:tool name="read_file" args={"path": "/nonexistent/file.txt"} assign="missing_data" -->
"""
        modified_content, context = execute_tool_directives(content)

        # Check that context has None for failed tool
        assert "missing_data" in context
        assert context["missing_data"] is None

        # Check that failure note was added
        assert "Tool 'read_file' failed" in modified_content

    def test_execute_with_existing_context(self):
        """Test executing tool directives with existing context."""
        content = """
<!-- tsu:tool name="test_tool" args={"key": "value"} assign="new_data" -->
"""
        existing_context = {"existing_var": "existing_value"}

        with patch("tsugite.agent_runner.runner.call_tool") as mock_call_tool:
            mock_call_tool.return_value = "New result"

            modified_content, tool_context = execute_tool_directives(content, existing_context)

            # New tool context should only have new variables
            assert "new_data" in tool_context
            assert "existing_var" not in tool_context

    def test_no_directives_returns_unchanged(self):
        """Test that content without directives is returned unchanged."""
        content = """
Normal content
No directives here
"""
        modified_content, context = execute_tool_directives(content)

        assert modified_content == content
        assert context == {}

    def test_malformed_directive_logs_warning(self):
        """Test that malformed directives log warnings."""

        from tsugite.events import EventBus, WarningEvent

        content = """
<!-- tsu:tool name="test" args={malformed} assign="data" -->
"""
        # Create event bus and track emitted events
        event_bus = EventBus()
        events = []

        def track_event(event):
            events.append(event)

        event_bus.subscribe(track_event)

        # Execute with event bus
        modified_content, context = execute_tool_directives(content, event_bus=event_bus)

        # Verify warning event was emitted
        assert len(events) > 0
        warning_events = [e for e in events if isinstance(e, WarningEvent)]
        assert len(warning_events) == 1
        assert "Failed to parse" in warning_events[0].message

        # Should return content unchanged with empty context
        assert context == {}


class TestCustomShellToolInjection:
    """Tests for custom shell tool injection into executor namespace."""

    def test_shell_tool_injected_into_executor(self):
        """Test that custom shell tools are accessible in executor namespace."""
        from tsugite.core.agent import TsugiteAgent
        from tsugite.core.tools import create_tool_from_tsugite
        from tsugite.tools.shell_tools import ShellToolDefinition, ShellToolParameter, register_shell_tool

        # Create a simple echo tool
        tool_def = ShellToolDefinition(
            name="test_echo",
            description="Test echo command",
            command="echo {message}",
            parameters={
                "message": ShellToolParameter(
                    name="message",
                    type="str",
                    description="Message to echo",
                    required=True,
                )
            },
            timeout=5,
        )

        # Register the shell tool
        register_shell_tool(tool_def)

        # Create a Tool object from the registered tool
        tool_obj = create_tool_from_tsugite("test_echo")

        # Create an agent with the tool
        agent = TsugiteAgent(
            model_string="ollama:qwen2.5-coder:7b",
            tools=[tool_obj],
            instructions="",
            max_turns=1,
        )

        # Verify the tool was injected into executor namespace
        assert hasattr(agent.executor, "namespace")
        assert "test_echo" in agent.executor.namespace
        assert callable(agent.executor.namespace["test_echo"])

    def test_multiple_shell_tools_injected(self):
        """Test that multiple custom shell tools are all injected."""
        from tsugite.core.agent import TsugiteAgent
        from tsugite.core.tools import create_tool_from_tsugite
        from tsugite.tools.shell_tools import ShellToolDefinition, ShellToolParameter, register_shell_tools

        # Create multiple tool definitions
        tool_defs = [
            ShellToolDefinition(
                name="test_ls",
                description="List files",
                command="ls {path}",
                parameters={"path": ShellToolParameter(name="path", type="str", default=".", required=False)},
            ),
            ShellToolDefinition(
                name="test_pwd",
                description="Print working directory",
                command="pwd",
                parameters={},
            ),
        ]

        # Register all tools
        register_shell_tools(tool_defs)

        # Create Tool objects from registered tools
        tool_objs = [create_tool_from_tsugite(td.name) for td in tool_defs]

        # Create agent with both tools
        agent = TsugiteAgent(
            model_string="ollama:qwen2.5-coder:7b",
            tools=tool_objs,
            instructions="",
            max_turns=1,
        )

        # Verify both tools were injected
        assert "test_ls" in agent.executor.namespace
        assert "test_pwd" in agent.executor.namespace
        assert callable(agent.executor.namespace["test_ls"])
        assert callable(agent.executor.namespace["test_pwd"])

    def test_shell_tool_callable_from_exec(self):
        """Test that injected shell tools can actually be called from exec()."""
        from tsugite.core.agent import TsugiteAgent
        from tsugite.core.tools import create_tool_from_tsugite
        from tsugite.tools.shell_tools import ShellToolDefinition, register_shell_tool

        # Create a simple tool
        tool_def = ShellToolDefinition(
            name="test_date",
            description="Get current date",
            command="date +%Y-%m-%d",
            parameters={},
        )

        register_shell_tool(tool_def)

        # Create Tool object from registered tool
        tool_obj = create_tool_from_tsugite("test_date")

        # Create agent
        agent = TsugiteAgent(
            model_string="ollama:qwen2.5-coder:7b",
            tools=[tool_obj],
            instructions="",
            max_turns=1,
        )

        # Try calling the tool from the executor namespace
        test_code = """
result = test_date()
"""
        # This should not raise NameError
        exec(test_code, agent.executor.namespace)
        assert "result" in agent.executor.namespace


class TestToolDirectivesIntegration:
    """Integration tests for tool directives with agents."""

    def test_tool_directive_in_regular_agent(self, tmp_path):
        """Test tool directive in a regular (non-multistep) agent."""
        # This would require mocking the entire agent execution
        # For now, we'll test the parsing and execution separately
        # Full integration tests can be added later
        pass

    def test_tool_directive_in_multistep_agent(self, tmp_path):
        """Test tool directive in a multi-step agent."""
        # Similar to above - would require full agent execution
        pass

    def test_tool_directive_with_jinja_in_content(self):
        """Test that Jinja variables in content still work with tool directives."""
        content = """
User asked: {{ user_prompt }}

<!-- tsu:tool name="read_file" args={"path": "test.txt"} assign="data" -->

Result: {{ data }}
"""
        # Tool directive extraction should work regardless of Jinja syntax
        directives = extract_tool_directives(content)
        assert len(directives) == 1

    def test_tool_directive_result_available_in_template(self):
        """Test that tool directive results are available in Jinja templates."""
        from tsugite.renderer import AgentRenderer

        content = """
<!-- tsu:tool name="get_data" args={"source": "api"} assign="file_data" -->

The file contains: {{ file_data }}
"""
        # Execute tool directives with mock
        with patch("tsugite.agent_runner.runner.call_tool") as mock_call_tool:
            mock_call_tool.return_value = "Tool result content"

            modified_content, context = execute_tool_directives(content)

            # Render with context
            renderer = AgentRenderer()
            rendered = renderer.render(modified_content, context)

            # Check that tool result was rendered
            assert "The file contains: Tool result content" in rendered
            assert "tsu:tool" not in rendered

    def test_combined_ignore_and_tool_directives(self):
        """Test using both ignore and tool directives in same content."""
        content = """
<!-- tsu:ignore -->
This is documentation that should be ignored.
The tool directive below will execute.
<!-- /tsu:ignore -->

<!-- tsu:tool name="get_file" args={"path": "test.txt"} assign="data" -->

Result: {{ data }}
"""
        # Execute tool directives first with mock
        with patch("tsugite.agent_runner.runner.call_tool") as mock_call_tool:
            mock_call_tool.return_value = "Data from file"

            modified_content, context = execute_tool_directives(content)

            # Then render (which strips ignore blocks)
            from tsugite.renderer import AgentRenderer

            renderer = AgentRenderer()
            rendered = renderer.render(modified_content, context)

            # Documentation should be gone
            assert "This is documentation" not in rendered

            # Tool result should be present
            assert "Result: Data from file" in rendered
