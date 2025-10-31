"""Tests for CLI argument parsing."""

import pytest

from tsugite.cli.helpers import parse_cli_arguments


class TestParseCliArguments:
    def test_single_agent_with_quoted_prompt(self):
        """Test single agent with a quoted prompt."""
        agents, prompt, _ = parse_cli_arguments(["+assistant", "create a report"], check_stdin=False)
        assert agents == ["+assistant"]
        assert prompt == "create a report"

    def test_single_agent_with_empty_prompt(self):
        """Test single agent with no prompt."""
        agents, prompt, _ = parse_cli_arguments(["+assistant"], check_stdin=False)
        assert agents == ["+assistant"]
        assert prompt == ""

    def test_multiple_agents_positional(self):
        """Test multiple agents with quoted prompt."""
        agents, prompt, _ = parse_cli_arguments(["+assistant", "+jira", "+coder", "fix bug"], check_stdin=False)
        assert agents == ["+assistant", "+jira", "+coder"]
        assert prompt == "fix bug"

    def test_unquoted_multi_word_prompt(self):
        """Test agent with unquoted multi-word prompt."""
        agents, prompt, _ = parse_cli_arguments(
            ["+assistant", "create", "a", "ticket", "for", "bug", "123"], check_stdin=False
        )
        assert agents == ["+assistant"]
        assert prompt == "create a ticket for bug 123"

    def test_multiple_agents_unquoted_prompt(self):
        """Test multiple agents with unquoted prompt."""
        agents, prompt, _ = parse_cli_arguments(["+a", "+b", "do", "this", "task"], check_stdin=False)
        assert agents == ["+a", "+b"]
        assert prompt == "do this task"

    def test_agent_paths_with_md(self):
        """Test agent file paths ending in .md."""
        agents, prompt, _ = parse_cli_arguments(["agent.md", "helper.md", "run task"], check_stdin=False)
        assert agents == ["agent.md", "helper.md"]
        assert prompt == "run task"

    def test_agent_paths_with_slash(self):
        """Test agent paths with slashes."""
        agents, prompt, _ = parse_cli_arguments(["agents/main.md", "helpers/test", "do work"], check_stdin=False)
        assert agents == ["agents/main.md", "helpers/test"]
        assert prompt == "do work"

    def test_mixed_agent_formats(self):
        """Test mix of shorthand and file paths."""
        agents, prompt, _ = parse_cli_arguments(["+assistant", "helper.md", "agents/coder", "task"], check_stdin=False)
        assert agents == ["+assistant", "helper.md", "agents/coder"]
        assert prompt == "task"

    def test_no_arguments_error(self):
        """Test error when no arguments provided."""
        with pytest.raises(ValueError, match="No arguments provided"):
            parse_cli_arguments([])

    def test_no_agents_defaults_to_builtin(self):
        """Test that no agents defaults to default."""
        agents, prompt, _ = parse_cli_arguments(["just", "a", "prompt"], check_stdin=False)
        assert agents == ["+default"]
        assert prompt == "just a prompt"

    def test_single_word_prompt(self):
        """Test single word prompt."""
        agents, prompt, _ = parse_cli_arguments(["+assistant", "hello"], check_stdin=False)
        assert agents == ["+assistant"]
        assert prompt == "hello"

    def test_prompt_with_special_characters(self):
        """Test prompt with special characters."""
        agents, prompt, _ = parse_cli_arguments(
            ["+assistant", "fix", "bug", "#123", "in", "auth.py"], check_stdin=False
        )
        assert agents == ["+assistant"]
        assert prompt == "fix bug #123 in auth.py"

    def test_only_agents_no_prompt(self):
        """Test multiple agents with no prompt."""
        agents, prompt, _ = parse_cli_arguments(["+a", "+b", "+c"], check_stdin=False)
        assert agents == ["+a", "+b", "+c"]
        assert prompt == ""

    def test_agent_followed_by_non_agent(self):
        """Test that once we see non-agent, rest is prompt."""
        agents, prompt, _ = parse_cli_arguments(["+a", "task.md", "+b"], check_stdin=False)
        # "task.md" ends with .md so it's an agent
        # "+b" comes after, also an agent
        assert agents == ["+a", "task.md", "+b"]
        assert prompt == ""

    def test_stop_agent_detection_after_first_non_agent(self):
        """Test that +x after non-agent word is part of prompt."""
        agents, prompt, _ = parse_cli_arguments(["+assistant", "create", "+urgent", "ticket"], check_stdin=False)
        assert agents == ["+assistant"]
        assert prompt == "create +urgent ticket"

    def test_file_reference_not_treated_as_agent(self):
        """Test that @filename.md is treated as prompt, not agent."""
        agents, prompt, _ = parse_cli_arguments(["+assistant", "Summarize", "@README.md"], check_stdin=False)
        assert agents == ["+assistant"]
        assert prompt == "Summarize @README.md"

    def test_file_reference_with_path_not_treated_as_agent(self):
        """Test that @path/file.md is treated as prompt, not agent."""
        agents, prompt, _ = parse_cli_arguments(["+assistant", "Review", "@src/main.py"], check_stdin=False)
        assert agents == ["+assistant"]
        assert prompt == "Review @src/main.py"

    def test_quoted_file_reference_not_treated_as_agent(self):
        """Test that quoted @filename is treated as prompt."""
        agents, prompt, _ = parse_cli_arguments(["+assistant", 'Check @"my file.md"'], check_stdin=False)
        assert agents == ["+assistant"]
        assert prompt == 'Check @"my file.md"'

    def test_prompt_ending_with_md_not_treated_as_agent(self):
        """Test that prompts ending with .md are not treated as agent files."""
        agents, prompt, _ = parse_cli_arguments(["+assistant", "Summarize @README.md"], check_stdin=False)
        assert agents == ["+assistant"]
        assert prompt == "Summarize @README.md"

    def test_prompt_with_spaces_not_treated_as_agent(self):
        """Test that arguments with spaces are treated as prompts."""
        agents, prompt, _ = parse_cli_arguments(["+assistant", "Review the code in agent.md"], check_stdin=False)
        assert agents == ["+assistant"]
        assert prompt == "Review the code in agent.md"
