import contextvars
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tsugite
from tests.interaction_doubles import FakeBackend
from tsugite.agent_preparation import AgentPreparer
from tsugite.agent_runner.runner import run_agent_async
from tsugite.core.agent import TsugiteAgent
from tsugite.interaction import set_interaction_backend
from tsugite.md_agents import parse_agent, parse_agent_file
from tsugite.permissions import Permissions, set_permissions
from tsugite.providers.base import CompletionResponse, Usage
from tsugite.tools import tool
from tsugite.tools import tool_access as ta
from tsugite.tools.fs import read_file

DEFAULT_MD = Path(tsugite.__file__).parent / "builtin_agents" / "default.md"


@pytest.fixture(autouse=True)
def demo_tools(tmp_path):
    @tool
    def demo_extra() -> str:
        """Plugin tool outside the core set.

        Returns an empty string unless the plugin is configured.
        """

    @tool(category="mcp_demo")
    def mcp_demo_send() -> str:
        """Send through the demo MCP server."""

    tool(read_file)
    for fn in (ta.search_tools, ta.load_tool, ta.request_tool_access):
        tool(parent_only=True)(fn)
    ta._remembered.clear()
    token = ta.current_access.set(None)
    set_permissions(Permissions(tmp_path / "permissions.yaml"))
    yield
    set_interaction_backend(None)
    set_permissions(None)
    ta.current_access.reset(token)
    ta._remembered.clear()


def _prepare(md=None):
    agent = parse_agent(md) if md else parse_agent_file(DEFAULT_MD)
    return AgentPreparer().prepare(agent=agent, prompt="hi", context={})


@pytest.mark.parametrize(
    "name, described, allowed",
    [
        ("read_file", True, True),
        ("demo_extra", False, True),
        ("mcp_demo_send", False, False),
    ],
)
def test_default_agent_allows_all_but_describes_core(name, described, allowed):
    prepared = _prepare()
    assert (f"def {name}(" in prepared.system_message) is described
    assert (name in ta.get_access().allowed) is allowed


def test_explicit_tools_agent_keeps_its_list():
    prepared = _prepare("---\nname: narrow\nextends: none\ntools: [read_file]\n---\nhi")
    assert [t.name for t in prepared.tools if t.name != "ask_user"] == ["read_file"]


@pytest.mark.parametrize(
    "query, status",
    [("plugin tool", "available"), ("demo mcp", "not allowed")],
)
def test_search_tools_marks_availability(query, status):
    _prepare()
    assert [m["status"] for m in ta.search_tools(query)] == [status]


def _loaded_agent():
    agent = SimpleNamespace(tools=[], loads=0)
    agent._inject_tools_into_executor = lambda: setattr(agent, "loads", agent.loads + 1)
    _prepare()
    ta.get_access().agent = agent
    return agent


def test_load_tool_adds_available_tool_to_running_agent():
    agent = _loaded_agent()
    assert "demo_extra" in ta.load_tool("demo_extra")
    assert [t.name for t in agent.tools] == ["demo_extra"]
    assert agent.loads == 1
    assert "demo_extra" in [t.name for t in _prepare().tools]


def test_load_tool_refuses_unallowed_tool():
    _loaded_agent()
    with pytest.raises(PermissionError):
        ta.load_tool("mcp_demo_send")


@pytest.mark.parametrize(
    "answer, loaded, denied, always",
    [
        ("Approve", ["mcp_demo_send"], set(), []),
        ("Always allow", ["mcp_demo_send"], set(), ["mcp_demo_send"]),
        ("Deny", [], {"mcp_demo_send"}, []),
    ],
)
def test_request_tool_access_decisions(answer, loaded, denied, always):
    agent = _loaded_agent()
    backend = FakeBackend(answer)
    set_interaction_backend(backend)
    ta.request_tool_access(["mcp_demo_send"], "send the report")
    access = ta.get_access()
    assert [t.name for t in agent.tools] == loaded
    assert access.denied == denied
    assert ta.get_permissions().listed("agents.default.tools") == always
    assert "MCP server demo" in backend.calls[0][0]
    assert "send the report" in backend.calls[0][0]


def test_denied_tool_is_not_asked_again():
    _loaded_agent()
    backend = FakeBackend("Deny")
    set_interaction_backend(backend)
    ta.request_tool_access(["mcp_demo_send"], "first")
    ta.request_tool_access(["mcp_demo_send"], "second")
    assert len(backend.calls) == 1


def test_tools_refuse_agent_without_deferred_tools():
    _prepare("---\nname: narrow\nextends: none\ntools: [read_file]\n---\nhi")
    with pytest.raises(RuntimeError, match="deferred_tools"):
        ta.search_tools("demo")


@pytest.mark.parametrize("name", ["demo_extra", "read_file"])
def test_load_tool_returns_full_docstring(name):
    _loaded_agent()
    info = ta.load_tool(name)
    assert f"def {name}(" in info
    assert inspect.getdoc(ta.get_tool(name).func).splitlines()[-1] in info


CHILD_MD = "---\nname: child\nextends: none\ntools: [read_file]\ndeferred_tools: ['*', '-mcp_*']\n---\nhi"


def _child_turn():
    _prepare(CHILD_MD)
    ta.get_access().agent = SimpleNamespace(tools=[])
    return ta.get_access()


def test_child_agent_in_same_session_gets_its_own_access():
    parent = _loaded_agent()
    set_interaction_backend(FakeBackend("Approve"))
    ta.request_tool_access(["mcp_demo_send"], "send the report")
    child = contextvars.copy_context().run(_child_turn)
    assert "mcp_demo_send" not in child.allowed
    assert ta.get_access().agent is parent
    assert "mcp_demo_send" in ta.get_access().allowed


def test_remembered_access_is_bounded(monkeypatch):
    monkeypatch.setattr(ta, "MAX_REMEMBERED", 1)
    _loaded_agent()
    ta.load_tool("demo_extra")
    _prepare(CHILD_MD)
    _prepare()
    assert ta.get_access().loaded == []


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["local", "subprocess"])
async def test_run_loads_tool_through_executor(tmp_path, mock_provider, monkeypatch, backend):
    monkeypatch.setenv("TSUGITE_EXECUTOR_BACKEND", backend)
    agent_file = tmp_path / "agent.md"
    agent_file.write_text(
        "---\nname: t\nmodel: openai:gpt-4o-mini\nextends: none\ntools: [read_file]\n"
        "deferred_tools: ['*', '-mcp_*']\n---\ndo the thing"
    )
    steps = [
        'load_tool(name="demo_extra")\nprint(search_tools(query="plugin tool"))',
        'demo_extra()\nfinal_answer("done")',
    ]
    mock_provider.acompletion.side_effect = [
        CompletionResponse(content=f"```python-exec\n{code}\n```", usage=Usage(total_tokens=10)) for code in steps
    ]
    agents = []

    def make_agent(**kwargs):
        agents.append(TsugiteAgent(**kwargs))
        return agents[-1]

    with patch("tsugite.agent_runner.runner.TsugiteAgent", side_effect=make_agent):
        assert await run_agent_async(agent_file, "go") == "done"
    assert "demo_extra" in [t.name for t in agents[0].tools]
    assert "'status': 'available'" in mock_provider.acompletion.call_args_list[1].kwargs["messages"][-1]["content"]
