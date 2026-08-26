"""`conversation_id` in the agent template context.

`session_id` is populated only for source="session" runs, so a chat template had
no way to key a path on the conversation it belongs to. `conversation_id` carries
the conv id for every source, and defaults to "" so a template using it still
renders under StrictUndefined outside the daemon.
"""

import pytest
from tsugite_daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite_daemon.config import RuntimeDefaults
from tsugite_daemon.session_store import SessionStore

from tsugite.agent_preparation import AgentPreparer
from tsugite.md_agents import parse_agent


class _StubAdapter(BaseAdapter):
    def get_platform_name(self) -> str:
        return "test"

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


@pytest.fixture
def adapter(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "agent.md").write_text("---\nname: test-agent\n---\n\nHi.\n")
    store = SessionStore(tmp_path / "store.json")
    config = RuntimeDefaults(workspace_dir=ws, agent_file=str(ws / "agent.md"))
    return _StubAdapter(config, store)


def test_chat_source_gets_conversation_id_even_though_session_id_is_blank(adapter):
    ctx = adapter._build_agent_context(
        ChannelContext(source="http", channel_id=None, user_id="u1", reply_to="http:u1"),
        conv_id="20260826_120000_u1_abc123",
    )

    assert ctx["conversation_id"] == "20260826_120000_u1_abc123"
    assert ctx["session_id"] == "", "session_id stays session-source-only"


def test_conversation_id_is_blank_when_there_is_no_conversation(adapter):
    ctx = adapter._build_agent_context(
        ChannelContext(source="http", channel_id=None, user_id="u1", reply_to="http:u1"),
    )

    assert ctx["conversation_id"] == ""


def test_template_using_conversation_id_renders_outside_the_daemon():
    body = "---\nname: t\n---\n\nnotes/{{ conversation_id }}.md\n"

    prepared = AgentPreparer().prepare(agent=parse_agent(body), prompt="go")

    assert "notes/.md" in prepared.rendered_prompt
