"""Positional tool calls must work in the subprocess executor, matching LocalExecutor.

Regression guard for the executor-default flip (Option D): non-sandboxed `tsu run` now uses
SubprocessExecutor, and the LLM frequently calls tools positionally (read_file("x")). The
old `def {name}(**kwargs)` stubs rejected that.
"""

import asyncio

from tsugite.core.subprocess_executor import SubprocessExecutor
from tsugite.core.tools import create_tool_from_tsugite


def test_positional_args_for_child_tool(reset_tool_registry, tmp_path):
    from tsugite.tools import _register_tool
    from tsugite.tools.fs import read_file

    _register_tool(read_file)  # importable -> runs in the child
    (tmp_path / "f.txt").write_text("hello-positional")

    ex = SubprocessExecutor(workspace_dir=tmp_path)
    try:
        ex.set_tools([create_tool_from_tsugite("read_file")])
        result = asyncio.run(ex.execute('print(read_file("f.txt"))'))
        assert result.error is None, result.error
        assert "hello-positional" in (result.output + result.stdout)
    finally:
        ex.cleanup()


def test_positional_keyword_collision_raises(reset_tool_registry, tmp_path):
    """A positional arg that also has an explicit keyword must raise, not silently overwrite
    (matches Python and LocalExecutor)."""
    from tsugite.tools import _register_tool
    from tsugite.tools.fs import read_file

    _register_tool(read_file)
    ex = SubprocessExecutor(workspace_dir=tmp_path)
    try:
        ex.set_tools([create_tool_from_tsugite("read_file")])
        result = asyncio.run(ex.execute('read_file("a.txt", path="b.txt")'))
        assert result.error is not None
        assert "multiple values" in result.error.lower(), result.error
    finally:
        ex.cleanup()


def test_positional_args_for_parent_only_tool(reset_tool_registry, tmp_path):
    from tsugite.tools import tool

    @tool(parent_only=True)
    def echo_arg(value: str) -> str:
        return f"echoed:{value}"

    ex = SubprocessExecutor(workspace_dir=tmp_path)
    try:
        ex.set_tools([create_tool_from_tsugite("echo_arg")])
        result = asyncio.run(ex.execute('print(echo_arg("hi"))'))
        assert result.error is None, result.error
        assert "echoed:hi" in (result.output + result.stdout)
    finally:
        ex.cleanup()
