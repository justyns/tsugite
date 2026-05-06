"""Tests for `<!-- tsu:exec -->` directive parsing and execution."""

from dataclasses import dataclass, field
from typing import List

import pytest

from tsugite.agent_runner import ExecBlockResult, execute_exec_directives, run_python_block
from tsugite.md_agents import ExecDirective, extract_exec_directives


class TestExecDirectiveParsing:
    """Parser-level tests for extract_exec_directives()."""

    def test_extract_single_exec_directive(self):
        content = """
<!-- tsu:exec name="dispatch" assign="targets" -->
targets = ["a", "b"]
targets
<!-- /tsu:exec -->
"""
        directives = extract_exec_directives(content)

        assert len(directives) == 1
        d = directives[0]
        assert isinstance(d, ExecDirective)
        assert d.name == "dispatch"
        assert d.assign_var == "targets"
        assert "targets = " in d.code
        assert d.timeout == 30
        assert d.continue_on_error is False
        assert d.stdout_assign is None

    def test_extract_exec_directive_with_optional_attrs(self):
        content = """
<!-- tsu:exec name="job" assign="result" stdout_assign="logs" timeout="5" continue_on_error="true" -->
print("hello")
result = 1
result
<!-- /tsu:exec -->
"""
        directives = extract_exec_directives(content)

        assert len(directives) == 1
        d = directives[0]
        assert d.name == "job"
        assert d.assign_var == "result"
        assert d.stdout_assign == "logs"
        assert d.timeout == 5
        assert d.continue_on_error is True

    def test_extract_multiple_exec_directives_preserves_order(self):
        content = """
<!-- tsu:exec name="first" assign="a" -->
a = 1
<!-- /tsu:exec -->

intervening text

<!-- tsu:exec name="second" assign="b" -->
b = 2
<!-- /tsu:exec -->
"""
        directives = extract_exec_directives(content)

        assert [d.name for d in directives] == ["first", "second"]
        assert [d.assign_var for d in directives] == ["a", "b"]

    def test_extract_exec_directive_missing_name_errors(self):
        content = """
<!-- tsu:exec assign="x" -->
x = 1
<!-- /tsu:exec -->
"""
        with pytest.raises(ValueError, match="missing required 'name'"):
            extract_exec_directives(content)

    def test_extract_exec_directive_unmatched_open_close(self):
        content = """
<!-- tsu:exec name="unclosed" assign="x" -->
x = 1
"""
        with pytest.raises(ValueError, match="unmatched"):
            extract_exec_directives(content)

    def test_extract_exec_directive_empty_body(self):
        content = """
<!-- tsu:exec name="noop" -->
<!-- /tsu:exec -->
"""
        directives = extract_exec_directives(content)
        assert len(directives) == 1
        assert directives[0].code.strip() == ""

    def test_extract_exec_directive_crlf_line_endings(self):
        content = '<!-- tsu:exec name="crlf" assign="x" -->\r\n' "x = 7\r\n" "x\r\n" "<!-- /tsu:exec -->\r\n"
        directives = extract_exec_directives(content)
        assert len(directives) == 1
        assert "x = 7" in directives[0].code

    def test_extract_exec_directive_no_open_tag_returns_empty(self):
        assert extract_exec_directives("nothing here") == []

    def test_extract_exec_directive_close_tag_in_body_truncates(self):
        # Documented v1 limitation: non-greedy regex matches the first close tag
        # even inside a Python string literal. We test the failure mode: parse
        # succeeds (matching to the first close), and the trailing text becomes
        # content outside the directive.
        content = """
<!-- tsu:exec name="early" assign="x" -->
s = "literal <!-- /tsu:exec -->"
<!-- /tsu:exec -->
"""
        directives = extract_exec_directives(content)
        assert len(directives) == 1
        assert "literal" in directives[0].code
        # The trailing real close tag gets parsed as text outside the block; the
        # body cut off at the first occurrence. Users hitting this should split
        # the literal across string concatenation.
        assert "<!-- /tsu:exec -->" not in directives[0].code

    def test_extract_exec_directive_with_jinja_in_body_is_passed_through(self):
        # Per design decision 1: body is NOT Jinja-rendered. Jinja syntax
        # inside the block survives untouched and is handed to Python verbatim.
        content = """
<!-- tsu:exec name="jinja_in_body" assign="x" -->
x = "{{ this is literal }}"
x
<!-- /tsu:exec -->
"""
        directives = extract_exec_directives(content)
        assert len(directives) == 1
        assert "{{ this is literal }}" in directives[0].code


class TestRunPythonBlock:
    """Runner-level tests for run_python_block()."""

    def test_explicit_return_value_captured(self):
        code = "return_value([1, 2, 3])"
        result = run_python_block(code, locals_dict={})
        assert result.error is None
        assert result.return_value == [1, 2, 3]

    def test_last_expression_captured(self):
        code = "x = 6 * 7\nx"
        result = run_python_block(code, locals_dict={})
        assert result.error is None
        assert result.return_value == 42

    def test_explicit_overrides_last_expression(self):
        code = """
return_value("explicit")
"trailing-expr"
"""
        result = run_python_block(code, locals_dict={})
        assert result.error is None
        assert result.return_value == "explicit"

    def test_locals_injected_as_python_locals(self):
        # Per design decision 1: prior-directive vars arrive as Python locals,
        # NOT via Jinja substitution.
        code = "result = prior_var + 10\nresult"
        result = run_python_block(code, locals_dict={"prior_var": 5})
        assert result.error is None
        assert result.return_value == 15

    def test_jinja_in_body_passes_through_to_python(self):
        # Verifies design decision 1: Jinja `{{ }}` inside the body is treated
        # as literal Python source. The user can build template strings safely.
        code = 'tmpl = "{{ name }}"\ntmpl'
        result = run_python_block(code, locals_dict={})
        assert result.error is None
        assert result.return_value == "{{ name }}"

    def test_stdout_capture(self):
        code = 'print("from-stdout")\nreturn_value("rv")'
        result = run_python_block(code, locals_dict={})
        assert result.error is None
        assert "from-stdout" in result.stdout
        assert result.return_value == "rv"

    def test_timeout_aborts(self):
        code = "import time\ntime.sleep(5)"
        with pytest.raises(RuntimeError, match="timed out"):
            run_python_block(code, locals_dict={}, timeout=1)

    def test_continue_on_error_swallows_exception(self):
        code = "raise RuntimeError('boom')"
        result = run_python_block(code, locals_dict={}, continue_on_error=True)
        assert result.error is not None
        assert "boom" in result.error
        assert result.return_value is None

    def test_fail_loud_default_raises(self):
        code = "raise RuntimeError('boom')"
        with pytest.raises(RuntimeError, match="boom"):
            run_python_block(code, locals_dict={})

    def test_path_coerced_to_str(self):
        code = "from pathlib import Path\nPath('/tmp/x')"
        result = run_python_block(code, locals_dict={})
        assert result.error is None
        assert result.return_value == "/tmp/x"

    def test_datetime_coerced_to_isoformat(self):
        code = "from datetime import datetime\ndatetime(2026, 1, 2, 3, 4, 5)"
        result = run_python_block(code, locals_dict={})
        assert result.error is None
        assert result.return_value == "2026-01-02T03:04:05"

    def test_set_coerced_to_sorted_list(self):
        code = "{'b', 'a', 'c'}"
        result = run_python_block(code, locals_dict={})
        assert result.error is None
        assert result.return_value == ["a", "b", "c"]

    def test_unsupported_type_fails_loud(self):
        # An open file handle is not JSON-serializable and has no coercion.
        code = "import io\nio.StringIO()"
        with pytest.raises(RuntimeError, match="not JSON-serializable"):
            run_python_block(code, locals_dict={})

    def test_open_is_allowed(self, tmp_path):
        # Intentional divergence from SubprocessExecutor: plain Python `open()`
        # works in tsu:exec because the directive is for deterministic
        # preprocessing, not LLM-emitted code.
        target = tmp_path / "data.txt"
        target.write_text("hello-from-disk")
        code = f"open({str(target)!r}).read()"
        result = run_python_block(code, locals_dict={})
        assert result.error is None
        assert result.return_value == "hello-from-disk"

    def test_returns_exec_block_result(self):
        result = run_python_block("1 + 1", locals_dict={})
        assert isinstance(result, ExecBlockResult)


@dataclass
class _RecordingBus:
    """Minimal EventBus stand-in for orchestrator tests."""

    emitted: List[object] = field(default_factory=list)

    def emit(self, event):
        self.emitted.append(event)


class TestExecuteExecDirectivesOrchestration:
    """Orchestrator-level tests for execute_exec_directives()."""

    def test_later_block_sees_earlier_assigned_var(self):
        content = """
<!-- tsu:exec name="first" assign="a" -->
return_value(7)
<!-- /tsu:exec -->

<!-- tsu:exec name="second" assign="b" -->
return_value(a + 5)
<!-- /tsu:exec -->
"""
        modified, ctx = execute_exec_directives(content)
        assert ctx == {"a": 7, "b": 12}
        assert "tsu:exec" not in modified

    def test_stdout_assign_populates_context(self):
        content = """
<!-- tsu:exec name="logger" stdout_assign="log" assign="rv" -->
print("line one")
print("line two")
return_value("done")
<!-- /tsu:exec -->
"""
        _, ctx = execute_exec_directives(content)
        assert ctx["rv"] == "done"
        assert "line one" in ctx["log"]
        assert "line two" in ctx["log"]

    def test_continue_on_error_emits_warning_and_nulls_assigns(self):
        content = """
<!-- tsu:exec name="boom" assign="x" stdout_assign="log" continue_on_error="true" -->
raise RuntimeError("kaboom")
<!-- /tsu:exec -->
"""
        bus = _RecordingBus()
        modified, ctx = execute_exec_directives(content, event_bus=bus)
        assert ctx == {"x": None, "log": ""}
        assert any("boom" in getattr(e, "message", "") for e in bus.emitted)
        assert "Exec 'boom' failed" in modified

    def test_no_directives_returns_passthrough(self):
        content = "no exec blocks here\n"
        modified, ctx = execute_exec_directives(content)
        assert modified == content
        assert ctx == {}

    def test_duplicate_raw_match_blocks_are_replaced_independently(self):
        content = """
<!-- tsu:exec name="dup" -->
1 + 1
<!-- /tsu:exec -->

<!-- tsu:exec name="dup" -->
1 + 1
<!-- /tsu:exec -->
"""
        modified, _ = execute_exec_directives(content)
        assert modified.count("<!-- Exec 'dup' executed -->") == 2
