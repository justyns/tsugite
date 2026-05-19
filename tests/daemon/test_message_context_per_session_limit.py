"""`_build_message_context` should render the SESSION's tracked context_limit
in `<context_limit>...</context_limit>` so the LLM sees the same value the UI
shows. Pre-fix it read `self.agent_config.context_limit` (agent-wide), which
drifts from the session's actual window once `update_session_context_limit`
fires for that session.
"""

import pytest

from tsugite.daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite.daemon.config import AgentConfig
from tsugite.daemon.session_store import Session, SessionSource, SessionStore


class _StubAdapter(BaseAdapter):
    async def start(self):
        pass

    async def stop(self):
        pass


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "store.json", context_limits={"test-agent": 128_000})


@pytest.fixture
def adapter(tmp_path, store):
    agent_config = AgentConfig(workspace_dir=tmp_path / "ws", agent_file="default")
    agent_config.context_limit = 128_000
    return _StubAdapter("test-agent", agent_config, store)


def test_message_context_uses_session_specific_limit(store, adapter):
    """A session with a 1M-window opus turn under its belt has
    `session.context_limit = 1_000_000`. The next turn's <context_limit> tag
    must reflect 1M, not the 128k agent default.
    """
    session = store.get_or_create_interactive("u1", "test-agent")
    store.update_session_context_limit(session.id, 1_000_000)

    ctx = adapter._build_message_context(
        "hello", ChannelContext(source="web", channel_id="c1", user_id="u1", reply_to="web:c1"), user_id="u1"
    )

    assert "<context_limit>1000000</context_limit>" in ctx, (
        f"expected session's 1M limit in <context_limit>, got: {ctx!r}"
    )


def test_message_context_falls_back_to_agent_default_when_session_has_no_limit(store, adapter):
    """A fresh session that hasn't reported a window yet still gets a meaningful
    value — the agent-wide default — instead of `None`.
    """
    store.get_or_create_interactive("u1", "test-agent")

    ctx = adapter._build_message_context(
        "hello", ChannelContext(source="web", channel_id="c2", user_id="u1", reply_to="web:c2"), user_id="u1"
    )

    assert "<context_limit>128000</context_limit>" in ctx, (
        f"expected agent-default 128k fallback in <context_limit>, got: {ctx!r}"
    )
    assert "None" not in ctx.split("<context_limit>")[1].split("</context_limit>")[0], (
        "fallback must not render the literal string 'None'"
    )
