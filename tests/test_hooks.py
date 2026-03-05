"""Tests for workspace hooks."""

import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import jinja2
import pytest

from tsugite.events.events import ToolCallEvent, ToolResultEvent
from tsugite.hooks import (
    HookHandler,
    HooksConfig,
    HookRule,
    _execute_hook,
    _render_and_execute,
    fire_compact_hooks,
    fire_pre_message_hooks,
    load_hooks_config,
)


def _ok_result(stdout="", stderr=""):
    return MagicMock(returncode=0, stdout=stdout, stderr=stderr)


def _fail_result(code=1, stdout="", stderr=""):
    return MagicMock(returncode=code, stdout=stdout, stderr=stderr)


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
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            self._simulate_tool_call(handler, "write_file", {"path": "/tmp/test/foo.md"})
            mock_run.assert_called_once()
            assert "foo.md" in mock_run.call_args[0][0]

    def test_skips_on_failed_tool_call(self):
        handler = self._make_handler([HookRule(tools=["write_file"], run="echo done")])
        with patch("tsugite.hooks.subprocess.run") as mock_run:
            self._simulate_tool_call(handler, "write_file", {"path": "x"}, success=False)
            mock_run.assert_not_called()

    def test_skips_non_matching_tool(self):
        handler = self._make_handler([HookRule(tools=["write_file"], run="echo done")])
        with patch("tsugite.hooks.subprocess.run") as mock_run:
            self._simulate_tool_call(handler, "read_file")
            mock_run.assert_not_called()

    def test_wildcard_matches_any_tool(self):
        handler = self._make_handler([HookRule(tools=["*"], run="echo {{ tool }}")])
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            self._simulate_tool_call(handler, "anything")
            mock_run.assert_called_once()
            assert "anything" in mock_run.call_args[0][0]

    def test_match_truthy_fires(self):
        handler = self._make_handler([
            HookRule(tools=["write_file"], match="{{ path.startswith('memory/') }}", run="echo yes")
        ])
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            self._simulate_tool_call(handler, "write_file", {"path": "memory/foo.md"})
            mock_run.assert_called_once()

    def test_match_falsy_skips(self):
        handler = self._make_handler([
            HookRule(tools=["write_file"], match="{{ path.startswith('memory/') }}", run="echo yes")
        ])
        with patch("tsugite.hooks.subprocess.run") as mock_run:
            self._simulate_tool_call(handler, "write_file", {"path": "other/foo.md"})
            mock_run.assert_not_called()

    def test_match_error_skips_with_warning(self):
        handler = self._make_handler([
            HookRule(tools=["write_file"], match="{{ undefined_func() }}", run="echo yes")
        ])
        with patch("tsugite.hooks.subprocess.run") as mock_run:
            self._simulate_tool_call(handler, "write_file")
            mock_run.assert_not_called()

    def test_run_renders_kwargs(self):
        handler = self._make_handler([
            HookRule(tools=["run"], run="echo {{ command }}")
        ])
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            self._simulate_tool_call(handler, "run", {"command": "ls -la"})
            assert mock_run.call_args[0][0] == "echo ls -la"

    def test_path_resolved_to_relative(self, tmp_path):
        handler = self._make_handler(
            [HookRule(tools=["write_file"], match="{{ path.startswith('memory/') }}", run="echo {{ path }}")],
            workspace_dir=tmp_path,
        )
        abs_path = str(tmp_path / "memory" / "note.md")
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            self._simulate_tool_call(handler, "write_file", {"path": abs_path})
            mock_run.assert_called_once()
            assert mock_run.call_args[0][0] == "echo memory/note.md"

    def test_error_logged_not_raised(self):
        handler = self._make_handler([HookRule(tools=["write_file"], run="echo done")])
        with patch("tsugite.hooks.subprocess.run", side_effect=OSError("boom")):
            self._simulate_tool_call(handler, "write_file")

    def test_no_match_field_always_fires(self):
        handler = self._make_handler([HookRule(tools=["write_file"], run="echo always")])
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            self._simulate_tool_call(handler, "write_file")
            mock_run.assert_called_once()

    def test_pending_cleared_after_result(self):
        handler = self._make_handler([HookRule(tools=["write_file"], run="echo done")])
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            self._simulate_tool_call(handler, "write_file")
            handler.handle_event(ToolResultEvent(tool_name="write_file", success=True))
            assert mock_run.call_count == 1

    def test_subprocess_run_called_with_correct_kwargs(self, tmp_path):
        handler = self._make_handler([HookRule(tools=["write_file"], run="echo done")], workspace_dir=tmp_path)
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            self._simulate_tool_call(handler, "write_file")
            mock_run.assert_called_once_with(
                "echo done",
                shell=True,
                cwd=str(tmp_path),
                capture_output=True,
                text=True,
                errors="replace",
                timeout=300,
            )

    def test_multiple_rules_all_evaluated(self):
        handler = self._make_handler([
            HookRule(tools=["write_file"], run="echo first"),
            HookRule(tools=["write_file"], run="echo second"),
        ])
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            self._simulate_tool_call(handler, "write_file")
            assert mock_run.call_count == 2


class TestExecuteHook:
    def test_calls_subprocess_run(self, tmp_path):
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            _execute_hook("echo hi", tmp_path)
            mock_run.assert_called_once_with(
                "echo hi", shell=True, cwd=str(tmp_path),
                capture_output=True, text=True, errors="replace", timeout=300,
            )

    def test_nonzero_exit_returns_none(self, tmp_path):
        with patch("tsugite.hooks.subprocess.run", return_value=_fail_result()):
            assert _execute_hook("exit 1", tmp_path) is None

    def test_failure_logs_stdout_and_stderr(self, tmp_path, caplog):
        with patch("tsugite.hooks.subprocess.run", return_value=_fail_result(stdout="out msg", stderr="err msg")):
            with caplog.at_level(logging.WARNING, logger="tsugite.hooks"):
                _execute_hook("bad cmd", tmp_path)
            assert "Hook stdout: out msg" in caplog.text
            assert "Hook stderr: err msg" in caplog.text

    def test_failure_skips_empty_stdout_stderr(self, tmp_path, caplog):
        with patch("tsugite.hooks.subprocess.run", return_value=_fail_result()):
            with caplog.at_level(logging.WARNING, logger="tsugite.hooks"):
                _execute_hook("bad cmd", tmp_path)
            assert "Hook stdout" not in caplog.text
            assert "Hook stderr" not in caplog.text

    def test_timeout_logged_not_raised(self, tmp_path):
        with patch("tsugite.hooks.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 300)):
            _execute_hook("sleep 999", tmp_path)

    def test_oserror_logged_not_raised(self, tmp_path):
        with patch("tsugite.hooks.subprocess.run", side_effect=OSError("boom")):
            _execute_hook("echo hi", tmp_path)


class TestHookRuleDefaults:
    def test_tools_defaults_to_empty(self):
        rule = HookRule(run="echo hi")
        assert rule.tools == []

    def test_wait_defaults_to_false(self):
        rule = HookRule(tools=["write_file"], run="echo hi")
        assert rule.wait is False

    def test_only_interactive_defaults_to_false(self):
        rule = HookRule(run="echo hi")
        assert rule.only_interactive is False


class TestHooksConfigCompact:
    def test_pre_compact_parsed(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text(
            "hooks:\n  pre_compact:\n    - run: echo extracting\n      wait: true\n"
        )
        config = load_hooks_config(tmp_workspace)
        assert len(config.pre_compact) == 1
        assert config.pre_compact[0].run == "echo extracting"
        assert config.pre_compact[0].wait is True

    def test_post_compact_parsed(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text(
            "hooks:\n  post_compact:\n    - run: echo done\n"
        )
        config = load_hooks_config(tmp_workspace)
        assert len(config.post_compact) == 1

    def test_all_hook_types_together(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text(
            "hooks:\n"
            "  post_tool:\n    - tools: [write_file]\n      run: echo tool\n"
            "  pre_compact:\n    - run: echo pre\n      wait: true\n"
            "  post_compact:\n    - run: echo post\n"
        )
        config = load_hooks_config(tmp_workspace)
        assert len(config.post_tool) == 1
        assert len(config.pre_compact) == 1
        assert len(config.post_compact) == 1


@pytest.mark.asyncio
class TestFireCompactHooks:
    async def test_no_config_is_noop(self, tmp_path):
        await fire_compact_hooks(tmp_path, "pre_compact", {})

    async def test_no_matching_phase_is_noop(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text(
            "hooks:\n  pre_compact:\n    - run: echo pre\n"
        )
        with patch("tsugite.hooks.subprocess.run") as mock_run:
            await fire_compact_hooks(tmp_workspace, "post_compact", {})
            mock_run.assert_not_called()

    async def test_pre_compact_fires_with_context(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text(
            "hooks:\n  pre_compact:\n    - run: echo {{ agent_name }}\n      wait: true\n"
        )
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            await fire_compact_hooks(tmp_workspace, "pre_compact", {"agent_name": "test-agent"})
            mock_run.assert_called_once()
            assert "test-agent" in mock_run.call_args[0][0]

    async def test_post_compact_fires(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text(
            "hooks:\n  post_compact:\n    - run: echo {{ turns_compacted }}\n"
        )
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            await fire_compact_hooks(tmp_workspace, "post_compact", {"turns_compacted": 5})
            mock_run.assert_called_once()
            assert "5" in mock_run.call_args[0][0]

    async def test_match_filters_hooks(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text(
            "hooks:\n  pre_compact:\n"
            "    - run: echo yes\n"
            "      match: \"{{ turn_count > 10 }}\"\n"
            "      wait: true\n"
        )
        with patch("tsugite.hooks.subprocess.run") as mock_run:
            await fire_compact_hooks(tmp_workspace, "pre_compact", {"turn_count": 5})
            mock_run.assert_not_called()

    async def test_multiple_hooks_fire_in_order(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text(
            "hooks:\n  pre_compact:\n"
            "    - run: echo first\n      wait: true\n"
            "    - run: echo second\n      wait: true\n"
        )
        calls = []
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            mock_run.side_effect = lambda *a, **kw: (calls.append(a[0]), _ok_result())[1]
            await fire_compact_hooks(tmp_workspace, "pre_compact", {})
            assert calls == ["echo first", "echo second"]

    async def test_render_error_skips_hook(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text(
            "hooks:\n  pre_compact:\n    - run: echo {{ undefined_var() }}\n      wait: true\n"
        )
        with patch("tsugite.hooks.subprocess.run") as mock_run:
            await fire_compact_hooks(tmp_workspace, "pre_compact", {})
            mock_run.assert_not_called()


class TestCaptureAs:
    def test_capture_as_field_parsed(self):
        rule = HookRule(run="echo hi", capture_as="my_var")
        assert rule.capture_as == "my_var"

    def test_capture_as_defaults_to_none(self):
        rule = HookRule(run="echo hi")
        assert rule.capture_as is None

    def test_capture_as_parsed_from_yaml(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text(
            "hooks:\n  pre_message:\n    - run: echo hello\n      capture_as: rag_context\n"
        )
        config = load_hooks_config(tmp_workspace)
        assert config.pre_message[0].capture_as == "rag_context"


class TestExecuteHookCapture:
    def test_capture_returns_stdout(self, tmp_path):
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result(stdout="captured output\n")):
            output = _execute_hook("echo captured output", tmp_path, capture=True)
            assert output == "captured output"

    def test_capture_returns_none_on_failure(self, tmp_path):
        with patch("tsugite.hooks.subprocess.run", return_value=_fail_result()):
            output = _execute_hook("exit 1", tmp_path, capture=True)
            assert output is None

    def test_no_capture_returns_none(self, tmp_path):
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()):
            result = _execute_hook("echo hi", tmp_path, capture=False)
            assert result is None

    def test_capture_timeout_returns_none(self, tmp_path):
        with patch("tsugite.hooks.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 300)):
            result = _execute_hook("sleep 999", tmp_path, capture=True)
            assert result is None


class TestRenderAndExecuteCapture:
    _env = jinja2.Environment()

    def test_captures_returned(self, tmp_path):
        rules = [HookRule(run="echo hello", capture_as="greeting")]
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result(stdout="hello")):
            captured = _render_and_execute(self._env, rules, {}, tmp_path)
            assert captured == {"greeting": "hello"}

    def test_no_capture_returns_empty_dict(self, tmp_path):
        rules = [HookRule(run="echo hello", wait=True)]
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result(stdout="hello")):
            captured = _render_and_execute(self._env, rules, {}, tmp_path)
            assert captured == {}

    def test_multiple_captures(self, tmp_path):
        rules = [
            HookRule(run="echo one", capture_as="var_a"),
            HookRule(run="echo two", capture_as="var_b"),
        ]
        call_count = [0]
        def mock_run_fn(*a, **kw):
            call_count[0] += 1
            return _ok_result(stdout=f"output_{call_count[0]}")

        with patch("tsugite.hooks.subprocess.run", side_effect=mock_run_fn):
            captured = _render_and_execute(self._env, rules, {}, tmp_path)
            assert captured == {"var_a": "output_1", "var_b": "output_2"}

    def test_failed_capture_not_included(self, tmp_path):
        rules = [HookRule(run="exit 1", capture_as="should_not_appear")]
        with patch("tsugite.hooks.subprocess.run", return_value=_fail_result()):
            captured = _render_and_execute(self._env, rules, {}, tmp_path)
            assert captured == {}

    def test_non_capture_returns_empty_dict(self, tmp_path):
        rules = [HookRule(run="echo fire", tools=["*"])]
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            captured = _render_and_execute(self._env, rules, {}, tmp_path)
            assert captured == {}
            mock_run.assert_called_once()


@pytest.mark.asyncio
class TestFirePreMessageHooks:
    async def test_no_config_returns_empty(self, tmp_path):
        result = await fire_pre_message_hooks(tmp_path, {})
        assert result == {}

    async def test_no_pre_message_rules_returns_empty(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text(
            "hooks:\n  post_tool:\n    - tools: [write_file]\n      run: echo done\n"
        )
        result = await fire_pre_message_hooks(tmp_workspace, {})
        assert result == {}

    async def test_captures_returned(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text(
            "hooks:\n  pre_message:\n    - run: echo rag results\n      capture_as: rag_context\n"
        )
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result(stdout="rag results")):
            result = await fire_pre_message_hooks(tmp_workspace, {"message": "test"})
            assert result == {"rag_context": "rag results"}

    async def test_context_rendered_in_command(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text(
            "hooks:\n  pre_message:\n"
            "    - run: echo {{ message }}\n      capture_as: echo_msg\n"
        )
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result(stdout="hello world")) as mock_run:
            await fire_pre_message_hooks(tmp_workspace, {"message": "hello world"})
            assert "hello world" in mock_run.call_args[0][0]

    async def test_match_filters_pre_message(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text(
            "hooks:\n  pre_message:\n"
            "    - run: echo hi\n      capture_as: x\n"
            '      match: "{{ message | length > 100 }}"\n'
        )
        with patch("tsugite.hooks.subprocess.run") as mock_run:
            result = await fire_pre_message_hooks(tmp_workspace, {"message": "short"})
            mock_run.assert_not_called()
            assert result == {}

    async def test_pre_message_parsed_in_config(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text(
            "hooks:\n  pre_message:\n    - run: echo hi\n      capture_as: var\n"
        )
        config = load_hooks_config(tmp_workspace)
        assert len(config.pre_message) == 1
        assert config.pre_message[0].capture_as == "var"


class TestOnlyInteractive:
    def test_only_interactive_parsed_from_yaml(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text(
            "hooks:\n  pre_message:\n    - run: echo hi\n      only_interactive: true\n"
        )
        config = load_hooks_config(tmp_workspace)
        assert config.pre_message[0].only_interactive is True

    def test_skipped_when_not_interactive(self):
        rules = [HookRule(run="echo hi", only_interactive=True)]
        with patch("tsugite.hooks.subprocess.run") as mock_run:
            _render_and_execute(jinja2.Environment(), rules, {}, Path("."), interactive=False)
            mock_run.assert_not_called()

    def test_fires_when_interactive(self):
        rules = [HookRule(run="echo hi", only_interactive=True)]
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            _render_and_execute(jinja2.Environment(), rules, {}, Path("."), interactive=True)
            mock_run.assert_called_once()

    def test_default_fires_regardless_of_interactive(self):
        rules = [HookRule(run="echo hi")]
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            _render_and_execute(jinja2.Environment(), rules, {}, Path("."), interactive=False)
            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_pre_message_respects_only_interactive(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text(
            "hooks:\n  pre_message:\n"
            "    - run: echo hi\n      capture_as: x\n      only_interactive: true\n"
        )
        with patch("tsugite.hooks.subprocess.run") as mock_run:
            result = await fire_pre_message_hooks(tmp_workspace, {"message": "test"}, interactive=False)
            mock_run.assert_not_called()
            assert result == {}
