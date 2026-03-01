"""Tests for workspace hooks."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from tsugite.events.events import ToolCallEvent, ToolResultEvent
from tsugite.hooks import HookHandler, HooksConfig, HookRule, load_hooks_config


@pytest.fixture
def tmp_workspace(tmp_path):
    (tmp_path / ".tsugite").mkdir()
    return tmp_path


class TestLoadHooksConfig:
    def test_missing_file_returns_none(self, tmp_path):
        assert load_hooks_config(tmp_path) is None

    def test_valid_yaml(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text(
            "hooks:\n  post_tool:\n    - tools: [write_file]\n      run: echo done\n"
        )
        config = load_hooks_config(tmp_workspace)
        assert config is not None
        assert len(config.post_tool) == 1
        assert config.post_tool[0].tools == ["write_file"]
        assert config.post_tool[0].run == "echo done"

    def test_empty_yaml_returns_none(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text("")
        assert load_hooks_config(tmp_workspace) is None

    def test_yaml_without_hooks_key_returns_none(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text("other: value\n")
        assert load_hooks_config(tmp_workspace) is None

    def test_match_field_parsed(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text(
            "hooks:\n  post_tool:\n    - tools: [write_file]\n"
            "      match: \"{{ path.startswith('memory/') }}\"\n"
            "      run: echo {{ path }}\n"
        )
        config = load_hooks_config(tmp_workspace)
        assert config.post_tool[0].match == "{{ path.startswith('memory/') }}"


class TestHookHandler:
    def _make_handler(self, rules, workspace_dir=None):
        config = HooksConfig(post_tool=rules)
        return HookHandler(config, workspace_dir or Path("/tmp/test"))

    def _simulate_tool_call(self, handler, tool_name, arguments=None, success=True):
        handler.handle_event(ToolCallEvent(tool_name=tool_name, arguments=arguments or {}))
        handler.handle_event(ToolResultEvent(tool_name=tool_name, success=success))

    def test_fires_on_successful_tool_call(self):
        handler = self._make_handler([HookRule(tools=["write_file"], run="echo {{ path }}")])
        with patch("tsugite.hooks.subprocess.Popen") as mock_popen:
            self._simulate_tool_call(handler, "write_file", {"path": "/tmp/test/foo.md"})
            mock_popen.assert_called_once()
            assert "foo.md" in mock_popen.call_args[0][0]

    def test_skips_on_failed_tool_call(self):
        handler = self._make_handler([HookRule(tools=["write_file"], run="echo done")])
        with patch("tsugite.hooks.subprocess.Popen") as mock_popen:
            self._simulate_tool_call(handler, "write_file", {"path": "x"}, success=False)
            mock_popen.assert_not_called()

    def test_skips_non_matching_tool(self):
        handler = self._make_handler([HookRule(tools=["write_file"], run="echo done")])
        with patch("tsugite.hooks.subprocess.Popen") as mock_popen:
            self._simulate_tool_call(handler, "read_file")
            mock_popen.assert_not_called()

    def test_wildcard_matches_any_tool(self):
        handler = self._make_handler([HookRule(tools=["*"], run="echo {{ tool }}")])
        with patch("tsugite.hooks.subprocess.Popen") as mock_popen:
            self._simulate_tool_call(handler, "anything")
            mock_popen.assert_called_once()
            assert "anything" in mock_popen.call_args[0][0]

    def test_match_truthy_fires(self):
        handler = self._make_handler([
            HookRule(tools=["write_file"], match="{{ path.startswith('memory/') }}", run="echo yes")
        ])
        with patch("tsugite.hooks.subprocess.Popen") as mock_popen:
            self._simulate_tool_call(handler, "write_file", {"path": "memory/foo.md"})
            mock_popen.assert_called_once()

    def test_match_falsy_skips(self):
        handler = self._make_handler([
            HookRule(tools=["write_file"], match="{{ path.startswith('memory/') }}", run="echo yes")
        ])
        with patch("tsugite.hooks.subprocess.Popen") as mock_popen:
            self._simulate_tool_call(handler, "write_file", {"path": "other/foo.md"})
            mock_popen.assert_not_called()

    def test_match_error_skips_with_warning(self):
        handler = self._make_handler([
            HookRule(tools=["write_file"], match="{{ undefined_func() }}", run="echo yes")
        ])
        with patch("tsugite.hooks.subprocess.Popen") as mock_popen:
            self._simulate_tool_call(handler, "write_file")
            mock_popen.assert_not_called()

    def test_run_renders_kwargs(self):
        handler = self._make_handler([
            HookRule(tools=["run"], run="echo {{ command }}")
        ])
        with patch("tsugite.hooks.subprocess.Popen") as mock_popen:
            self._simulate_tool_call(handler, "run", {"command": "ls -la"})
            assert mock_popen.call_args[0][0] == "echo ls -la"

    def test_path_resolved_to_relative(self, tmp_path):
        handler = self._make_handler(
            [HookRule(tools=["write_file"], match="{{ path.startswith('memory/') }}", run="echo {{ path }}")],
            workspace_dir=tmp_path,
        )
        abs_path = str(tmp_path / "memory" / "note.md")
        with patch("tsugite.hooks.subprocess.Popen") as mock_popen:
            self._simulate_tool_call(handler, "write_file", {"path": abs_path})
            mock_popen.assert_called_once()
            assert mock_popen.call_args[0][0] == "echo memory/note.md"

    def test_popen_error_logged_not_raised(self):
        handler = self._make_handler([HookRule(tools=["write_file"], run="echo done")])
        with patch("tsugite.hooks.subprocess.Popen", side_effect=OSError("boom")):
            self._simulate_tool_call(handler, "write_file")

    def test_no_match_field_always_fires(self):
        handler = self._make_handler([HookRule(tools=["write_file"], run="echo always")])
        with patch("tsugite.hooks.subprocess.Popen") as mock_popen:
            self._simulate_tool_call(handler, "write_file")
            mock_popen.assert_called_once()

    def test_pending_cleared_after_result(self):
        handler = self._make_handler([HookRule(tools=["write_file"], run="echo done")])
        with patch("tsugite.hooks.subprocess.Popen") as mock_popen:
            self._simulate_tool_call(handler, "write_file")
            # Second result without a preceding call should not fire
            handler.handle_event(ToolResultEvent(tool_name="write_file", success=True))
            assert mock_popen.call_count == 1

    def test_popen_called_with_correct_kwargs(self, tmp_path):
        handler = self._make_handler([HookRule(tools=["write_file"], run="echo done")], workspace_dir=tmp_path)
        with patch("tsugite.hooks.subprocess.Popen") as mock_popen:
            self._simulate_tool_call(handler, "write_file")
            mock_popen.assert_called_once_with(
                "echo done",
                shell=True,
                cwd=str(tmp_path),
                stdout=subprocess.DEVNULL,
            )

    def test_multiple_rules_all_evaluated(self):
        handler = self._make_handler([
            HookRule(tools=["write_file"], run="echo first"),
            HookRule(tools=["write_file"], run="echo second"),
        ])
        with patch("tsugite.hooks.subprocess.Popen") as mock_popen:
            self._simulate_tool_call(handler, "write_file")
            assert mock_popen.call_count == 2
