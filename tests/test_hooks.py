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
    HookRule,
    HooksConfig,
    _build_hook_env,
    _execute_hook,
    _jinja_env,
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
        handler = self._make_handler(
            [HookRule(tools=["write_file"], match="{{ path.startswith('memory/') }}", run="echo yes")]
        )
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            self._simulate_tool_call(handler, "write_file", {"path": "memory/foo.md"})
            mock_run.assert_called_once()

    def test_match_falsy_skips(self):
        handler = self._make_handler(
            [HookRule(tools=["write_file"], match="{{ path.startswith('memory/') }}", run="echo yes")]
        )
        with patch("tsugite.hooks.subprocess.run") as mock_run:
            self._simulate_tool_call(handler, "write_file", {"path": "other/foo.md"})
            mock_run.assert_not_called()

    def test_match_error_skips_with_warning(self):
        handler = self._make_handler([HookRule(tools=["write_file"], match="{{ undefined_func() }}", run="echo yes")])
        with patch("tsugite.hooks.subprocess.run") as mock_run:
            self._simulate_tool_call(handler, "write_file")
            mock_run.assert_not_called()

    def test_run_renders_kwargs(self):
        handler = self._make_handler([HookRule(tools=["run"], run="echo {{ command }}")])
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
            mock_run.assert_called_once()
            _, kwargs = mock_run.call_args
            assert kwargs["shell"] is True
            assert kwargs["cwd"] == str(tmp_path)
            assert kwargs["capture_output"] is True
            assert kwargs["text"] is True
            assert kwargs["timeout"] == 300
            assert kwargs["env"] is not None
            assert kwargs["env"]["TSUGITE_HOOK_PHASE"] == "post_tool"
            assert kwargs["env"]["TSUGITE_WORKSPACE_DIR"] == str(tmp_path)

    def test_multiple_rules_all_evaluated(self):
        handler = self._make_handler(
            [
                HookRule(tools=["write_file"], run="echo first"),
                HookRule(tools=["write_file"], run="echo second"),
            ]
        )
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            self._simulate_tool_call(handler, "write_file")
            assert mock_run.call_count == 2


class TestExecuteHook:
    def test_calls_subprocess_run(self, tmp_path):
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            _execute_hook("echo hi", tmp_path)
            mock_run.assert_called_once_with(
                "echo hi",
                shell=True,
                cwd=str(tmp_path),
                capture_output=True,
                text=True,
                errors="replace",
                timeout=300,
                env=None,
            )

    def test_nonzero_exit_returns_nonzero_code(self, tmp_path):
        with patch("tsugite.hooks.subprocess.run", return_value=_fail_result()):
            result = _execute_hook("exit 1", tmp_path)
            assert result.exit_code == 1

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
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text("hooks:\n  post_compact:\n    - run: echo done\n")
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
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text("hooks:\n  pre_compact:\n    - run: echo pre\n")
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
            'hooks:\n  pre_compact:\n    - run: echo yes\n      match: "{{ turn_count > 10 }}"\n      wait: true\n'
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
    def test_returns_stdout(self, tmp_path):
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result(stdout="captured output\n")):
            result = _execute_hook("echo captured output", tmp_path)
            assert result.stdout == "captured output"
            assert result.exit_code == 0

    def test_failure_returns_nonzero(self, tmp_path):
        with patch("tsugite.hooks.subprocess.run", return_value=_fail_result()):
            result = _execute_hook("exit 1", tmp_path)
            assert result.exit_code == 1

    def test_success_returns_zero(self, tmp_path):
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()):
            result = _execute_hook("echo hi", tmp_path)
            assert result.exit_code == 0

    def test_timeout_returns_negative_exit(self, tmp_path):
        with patch("tsugite.hooks.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 300)):
            result = _execute_hook("sleep 999", tmp_path)
            assert result.exit_code == -1
            assert "Timed out" in result.stderr


class TestRenderAndExecuteCapture:
    _env = jinja2.Environment()

    def test_captures_returned(self, tmp_path):
        rules = [HookRule(run="echo hello", capture_as="greeting")]
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result(stdout="hello")):
            results = _render_and_execute(self._env, rules, {}, tmp_path)
            assert results.captured == {"greeting": "hello"}
            assert len(results.executions) == 1
            assert results.executions[0].exit_code == 0

    def test_no_capture_returns_empty_dict(self, tmp_path):
        rules = [HookRule(run="echo hello", wait=True)]
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result(stdout="hello")):
            results = _render_and_execute(self._env, rules, {}, tmp_path)
            assert results.captured == {}

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
            results = _render_and_execute(self._env, rules, {}, tmp_path)
            assert results.captured == {"var_a": "output_1", "var_b": "output_2"}

    def test_failed_capture_not_included(self, tmp_path):
        rules = [HookRule(run="exit 1", capture_as="should_not_appear")]
        with patch("tsugite.hooks.subprocess.run", return_value=_fail_result()):
            results = _render_and_execute(self._env, rules, {}, tmp_path)
            assert results.captured == {}
            assert len(results.executions) == 1
            assert results.executions[0].exit_code == 1

    def test_non_capture_returns_empty_dict(self, tmp_path):
        rules = [HookRule(run="echo fire", tools=["*"])]
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            results = _render_and_execute(self._env, rules, {}, tmp_path)
            assert results.captured == {}
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
            "hooks:\n  pre_message:\n    - run: echo {{ message }}\n      capture_as: echo_msg\n"
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
            "hooks:\n  pre_message:\n    - run: echo hi\n      capture_as: x\n      only_interactive: true\n"
        )
        with patch("tsugite.hooks.subprocess.run") as mock_run:
            result = await fire_pre_message_hooks(tmp_workspace, {"message": "test"}, interactive=False)
            mock_run.assert_not_called()
            assert result == {}


class TestShellQuoteFilter:
    def test_apostrophe_escaped(self):
        result = _jinja_env.from_string("{{ msg | shell_quote }}").render(msg="it's a test")
        assert result == "'it'\"'\"'s a test'"

    def test_backtick_escaped(self):
        result = _jinja_env.from_string("{{ msg | shell_quote }}").render(msg="hello `world`")
        assert result == "'hello `world`'"

    def test_dollar_escaped(self):
        result = _jinja_env.from_string("{{ msg | shell_quote }}").render(msg="cost is $100")
        assert result == "'cost is $100'"

    def test_safe_string_unchanged(self):
        result = _jinja_env.from_string("{{ msg | shell_quote }}").render(msg="hello")
        assert result == "hello"

    def test_in_command_context(self):
        result = _jinja_env.from_string("search {{ msg | shell_quote }} --json").render(msg="it's here")
        assert result.startswith("search ")
        assert result.endswith(" --json")


class TestRunAsList:
    def test_list_run_parsed(self):
        rule = HookRule(run=["echo", "{{ message }}"])
        assert isinstance(rule.run, list)

    def test_list_run_from_yaml(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text(
            "hooks:\n  pre_message:\n    - run:\n        - echo\n        - '{{ message }}'\n      capture_as: out\n"
        )
        config = load_hooks_config(tmp_workspace)
        assert isinstance(config.pre_message[0].run, list)
        assert config.pre_message[0].run == ["echo", "{{ message }}"]

    def test_list_execute_hook_uses_shell_false(self, tmp_path):
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            _execute_hook(["echo", "hello"], tmp_path)
            mock_run.assert_called_once_with(
                ["echo", "hello"],
                shell=False,
                cwd=str(tmp_path),
                capture_output=True,
                text=True,
                errors="replace",
                timeout=300,
                env=None,
            )

    def test_list_render_and_execute(self, tmp_path):
        rules = [HookRule(run=["echo", "{{ name }}"], capture_as="out")]
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result(stdout="alice")) as mock_run:
            results = _render_and_execute(_jinja_env, rules, {"name": "alice"}, tmp_path)
            assert mock_run.call_args[0][0] == ["echo", "alice"]
            assert results.captured == {"out": "alice"}

    def test_list_with_shell_metacharacters(self, tmp_path):
        rules = [HookRule(run=["echo", "{{ msg }}"], capture_as="out")]
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result(stdout="it's $fine")) as mock_run:
            _render_and_execute(_jinja_env, rules, {"msg": "it's $fine"}, tmp_path)
            assert mock_run.call_args[0][0] == ["echo", "it's $fine"]
            assert mock_run.call_args[1]["shell"] is False

    @pytest.mark.asyncio
    async def test_pre_message_list_run(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text(
            "hooks:\n  pre_message:\n    - run:\n        - echo\n        - '{{ message }}'\n      capture_as: out\n"
        )
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result(stdout="hello")) as mock_run:
            result = await fire_pre_message_hooks(tmp_workspace, {"message": "it's a test"})
            assert mock_run.call_args[0][0] == ["echo", "it's a test"]
            assert mock_run.call_args[1]["shell"] is False
            assert result == {"out": "hello"}


class TestHookName:
    def test_name_field_parsed(self):
        rule = HookRule(run="echo hi", name="My Hook")
        assert rule.name == "My Hook"

    def test_name_defaults_to_none(self):
        rule = HookRule(run="echo hi")
        assert rule.name is None

    def test_name_parsed_from_yaml(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text(
            "hooks:\n  pre_message:\n    - run: echo hi\n      name: RAG Search\n      capture_as: rag\n"
        )
        config = load_hooks_config(tmp_workspace)
        assert config.pre_message[0].name == "RAG Search"


class TestOnStatusCallback:
    _env = jinja2.Environment()

    def test_on_status_called_with_name(self, tmp_path):
        rules = [HookRule(run="echo hi", name="RAG Search")]
        status_calls = []
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()):
            _render_and_execute(self._env, rules, {}, tmp_path, on_status=status_calls.append)
        assert status_calls == ["Running RAG Search..."]

    def test_on_status_falls_back_to_capture_as(self, tmp_path):
        rules = [HookRule(run="echo hi", capture_as="rag_context")]
        status_calls = []
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result(stdout="out")):
            _render_and_execute(self._env, rules, {}, tmp_path, on_status=status_calls.append)
        assert status_calls == ["Running rag_context..."]

    def test_on_status_falls_back_to_hook(self, tmp_path):
        rules = [HookRule(run="echo hi")]
        status_calls = []
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()):
            _render_and_execute(self._env, rules, {}, tmp_path, on_status=status_calls.append)
        assert status_calls == ["Running hook..."]

    def test_on_status_none_does_not_break(self, tmp_path):
        rules = [HookRule(run="echo hi", name="Test")]
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()):
            _render_and_execute(self._env, rules, {}, tmp_path, on_status=None)

    def test_on_status_called_per_rule(self, tmp_path):
        rules = [
            HookRule(run="echo a", name="First"),
            HookRule(run="echo b", name="Second"),
        ]
        status_calls = []
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()):
            _render_and_execute(self._env, rules, {}, tmp_path, on_status=status_calls.append)
        assert status_calls == ["Running First...", "Running Second..."]

    def test_on_status_not_called_for_skipped_rules(self, tmp_path):
        rules = [HookRule(run="echo hi", name="Skipped", only_interactive=True)]
        status_calls = []
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()):
            _render_and_execute(self._env, rules, {}, tmp_path, interactive=False, on_status=status_calls.append)
        assert status_calls == []

    @pytest.mark.asyncio
    async def test_fire_pre_message_hooks_passes_on_status(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text(
            "hooks:\n  pre_message:\n    - run: echo hi\n      name: Search\n      capture_as: x\n"
        )
        status_calls = []
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result(stdout="result")):
            await fire_pre_message_hooks(tmp_workspace, {"message": "test"}, on_status=status_calls.append)
        assert status_calls == ["Running Search..."]


class TestBuildHookEnv:
    def test_inherits_parent_env(self):
        env = _build_hook_env({})
        assert "PATH" in env

    def test_context_vars_prefixed(self):
        env = _build_hook_env({"tool": "write_file", "path": "foo.md"})
        assert env["TSUGITE_TOOL"] == "write_file"
        assert env["TSUGITE_PATH"] == "foo.md"

    def test_int_and_bool_converted_to_str(self):
        env = _build_hook_env({"count": 42, "verbose": True})
        assert env["TSUGITE_COUNT"] == "42"
        assert env["TSUGITE_VERBOSE"] == "True"

    def test_none_and_complex_skipped(self):
        env = _build_hook_env({"empty": None, "nested": {"a": 1}, "items": [1, 2]})
        assert "TSUGITE_EMPTY" not in env
        assert "TSUGITE_NESTED" not in env
        assert "TSUGITE_ITEMS" not in env

    def test_phase_and_workspace_metadata(self):
        env = _build_hook_env({}, phase="pre_compact", workspace_dir="/home/user/project")
        assert env["TSUGITE_HOOK_PHASE"] == "pre_compact"
        assert env["TSUGITE_WORKSPACE_DIR"] == "/home/user/project"

    def test_empty_phase_and_workspace_not_set(self):
        env = _build_hook_env({})
        assert "TSUGITE_HOOK_PHASE" not in env
        assert "TSUGITE_WORKSPACE_DIR" not in env

    def test_custom_env_applied(self):
        env = _build_hook_env({}, custom_env={"MY_VAR": "hello"})
        assert env["MY_VAR"] == "hello"

    def test_custom_env_overrides_tsugite_vars(self):
        env = _build_hook_env({"tool": "write_file"}, custom_env={"TSUGITE_TOOL": "overridden"})
        assert env["TSUGITE_TOOL"] == "overridden"


class TestExecuteHookEnv:
    def test_env_kwarg_passed_to_subprocess(self, tmp_path):
        fake_env = {"MY_VAR": "hello"}
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            _execute_hook("echo hi", tmp_path, env=fake_env)
            assert mock_run.call_args[1]["env"] == {"MY_VAR": "hello"}

    def test_env_none_by_default(self, tmp_path):
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            _execute_hook("echo hi", tmp_path)
            assert mock_run.call_args[1]["env"] is None


class TestRenderAndExecuteEnv:
    _env = jinja2.Environment()

    def test_env_flows_through(self, tmp_path):
        rules = [HookRule(run="echo hi")]
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            _render_and_execute(self._env, rules, {"tool": "write_file"}, tmp_path, phase="post_tool")
            env = mock_run.call_args[1]["env"]
            assert env["TSUGITE_TOOL"] == "write_file"
            assert env["TSUGITE_HOOK_PHASE"] == "post_tool"
            assert env["TSUGITE_WORKSPACE_DIR"] == str(tmp_path)

    def test_custom_env_rendered_with_jinja(self, tmp_path):
        rules = [HookRule(run="echo hi", env={"FILE": "{{ path }}", "STATIC": "fixed"})]
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            _render_and_execute(self._env, rules, {"path": "foo.md"}, tmp_path)
            env = mock_run.call_args[1]["env"]
            assert env["FILE"] == "foo.md"
            assert env["STATIC"] == "fixed"

    def test_custom_env_render_failure_skips_key(self, tmp_path):
        rules = [HookRule(run="echo hi", env={"BAD": "{{ undefined_func() }}", "GOOD": "ok"})]
        with patch("tsugite.hooks.subprocess.run", return_value=_ok_result()) as mock_run:
            _render_and_execute(self._env, rules, {}, tmp_path)
            env = mock_run.call_args[1]["env"]
            assert "BAD" not in env
            assert env["GOOD"] == "ok"


class TestHookRuleEnv:
    def test_env_defaults_to_empty_dict(self):
        rule = HookRule(run="echo hi")
        assert rule.env == {}

    def test_env_parsed_from_yaml(self, tmp_workspace):
        (tmp_workspace / ".tsugite" / "hooks.yaml").write_text(
            "hooks:\n  post_tool:\n    - tools: [write_file]\n      run: echo done\n"
            "      env:\n        FILE_PATH: '{{ path }}'\n        LOG_LEVEL: debug\n"
        )
        config = load_hooks_config(tmp_workspace)
        assert config.post_tool[0].env == {"FILE_PATH": "{{ path }}", "LOG_LEVEL": "debug"}
