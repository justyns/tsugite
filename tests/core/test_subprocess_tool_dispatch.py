"""Tool dispatch: importable plugin tools run in the (sandboxed) child; parent-only stay in the parent."""

from tsugite.core.subprocess_executor import SubprocessExecutor, _importable_in_child


def test_importable_in_child_true_for_plugin_tool():
    import tsugite_web

    assert _importable_in_child("tsugite_web", "web_search", tsugite_web.web_search) is True


def test_importable_in_child_true_for_builtin():
    from tsugite.tools.fs import read_file

    assert _importable_in_child("tsugite.tools.fs", "read_file", read_file) is True


def test_importable_in_child_false_for_local_closure():
    def local_tool():
        return None

    # A function defined inside another function isn't importable by name from its module.
    assert _importable_in_child(local_tool.__module__, "local_tool", local_tool) is False


def test_importable_in_child_false_for_main():
    def f():
        return None

    assert _importable_in_child("__main__", "f", f) is False


def test_set_tools_routes_plugin_to_child_and_parent_only_to_parent(reset_tool_registry):
    from tsugite_web import web_search

    from tsugite.core.tools import create_tool_from_tsugite
    from tsugite.tools import _register_tool, tool
    from tsugite.tools.fs import read_file

    _register_tool(read_file)  # built-in (tsugite.tools.fs)
    _register_tool(web_search)  # plugin (tsugite_web), importable -> child

    @tool(parent_only=True)
    def needs_parent() -> str:
        return "x"

    ex = SubprocessExecutor()
    try:
        ex.set_tools([create_tool_from_tsugite(n) for n in ["read_file", "web_search", "needs_parent"]])
        assert "read_file" in ex._local_tools
        assert "web_search" in ex._local_tools  # plugin tool now runs in the child
        assert "needs_parent" in ex._parent_only_tools
    finally:
        ex.cleanup()
