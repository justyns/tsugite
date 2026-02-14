"""Tests for unified sessions across daemon adapters."""

from pathlib import Path

import pytest

from tsugite.daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite.daemon.config import AgentConfig, DaemonConfig
from tsugite.daemon.session import SessionManager


class _StubAdapter(BaseAdapter):
    async def start(self):
        pass

    async def stop(self):
        pass


@pytest.fixture
def workspace_dir(tmp_path):
    return tmp_path / "workspace"


@pytest.fixture
def session_manager(workspace_dir):
    return SessionManager("test-agent", workspace_dir, context_limit=128000)


@pytest.fixture
def identity_map():
    return {
        "discord:123456789": "justyn",
        "http:justyn": "justyn",
        "http:web-anonymous": "justyn",
    }


def _make_adapter(workspace_dir, session_manager, identity_map=None):
    agent_config = AgentConfig(workspace_dir=workspace_dir, agent_file="default")
    return _StubAdapter("test-agent", agent_config, session_manager, identity_map=identity_map)


class TestIdentityResolution:
    """Tests for identity map lookup in BaseAdapter.resolve_user."""

    def test_linked_user_resolves_to_canonical(self, workspace_dir, session_manager, identity_map):
        adapter = _make_adapter(workspace_dir, session_manager, identity_map)

        ctx = ChannelContext(source="discord", channel_id="dm-chan", user_id="123456789", reply_to="discord:dm-chan")
        assert adapter.resolve_user("123456789", ctx) == "justyn"

        ctx_http = ChannelContext(source="http", channel_id=None, user_id="justyn", reply_to="http:justyn")
        assert adapter.resolve_user("justyn", ctx_http) == "justyn"

    def test_unlinked_user_gets_bare_id(self, workspace_dir, session_manager, identity_map):
        adapter = _make_adapter(workspace_dir, session_manager, identity_map)

        ctx = ChannelContext(source="discord", channel_id="dm-chan", user_id="999999", reply_to="discord:dm-chan")
        assert adapter.resolve_user("999999", ctx) == "999999"

    def test_empty_identity_map_passthrough(self, workspace_dir, session_manager):
        adapter = _make_adapter(workspace_dir, session_manager)

        ctx = ChannelContext(source="discord", channel_id="dm-chan", user_id="123456789", reply_to="discord:dm-chan")
        assert adapter.resolve_user("123456789", ctx) == "123456789"


class TestGroupIsolation:
    """Group chats produce composite keys, not unified identities."""

    def test_group_chat_isolated(self, workspace_dir, session_manager, identity_map):
        adapter = _make_adapter(workspace_dir, session_manager, identity_map)

        ctx = ChannelContext(
            source="discord",
            channel_id="guild-channel-42",
            user_id="123456789",
            reply_to="discord:guild-channel-42",
            metadata={"guild_id": "777"},
        )
        resolved = adapter.resolve_user("123456789", ctx)
        assert resolved == "discord:guild-channel-42:123456789"
        assert resolved != "justyn"

    def test_same_user_dm_vs_group_different(self, workspace_dir, session_manager, identity_map):
        adapter = _make_adapter(workspace_dir, session_manager, identity_map)

        dm_ctx = ChannelContext(source="discord", channel_id="dm-chan", user_id="123456789", reply_to="discord:dm-chan")
        group_ctx = ChannelContext(
            source="discord",
            channel_id="guild-chan",
            user_id="123456789",
            reply_to="discord:guild-chan",
            metadata={"guild_id": "777"},
        )
        assert adapter.resolve_user("123456789", dm_ctx) != adapter.resolve_user("123456789", group_ctx)


class TestSharedSessionManager:
    """Two adapters sharing a SessionManager get the same conversation_id for linked users."""

    def test_shared_session_same_conv_id(self, workspace_dir, identity_map):
        shared_sm = SessionManager("test-agent", workspace_dir, context_limit=128000)

        discord_adapter = _make_adapter(workspace_dir, shared_sm, identity_map)
        http_adapter = _make_adapter(workspace_dir, shared_sm, identity_map)

        discord_ctx = ChannelContext(
            source="discord", channel_id="dm-chan", user_id="123456789", reply_to="discord:dm-chan"
        )
        http_ctx = ChannelContext(source="http", channel_id=None, user_id="justyn", reply_to="http:justyn")

        discord_user = discord_adapter.resolve_user("123456789", discord_ctx)
        http_user = http_adapter.resolve_user("justyn", http_ctx)

        assert discord_user == http_user == "justyn"

        conv1 = shared_sm.get_or_create_session(discord_user)
        conv2 = shared_sm.get_or_create_session(http_user)
        assert conv1 == conv2


class TestSessionFilenameSanitization:
    """Colons in user IDs produce valid filenames."""

    def test_colons_replaced(self, workspace_dir):
        sm = SessionManager("test-agent", workspace_dir)
        path = sm._get_session_file("discord:guild-chan:123456789")
        assert ":" not in path.name
        assert path.name == "discord_guild-chan_123456789.json"

    def test_simple_id_unchanged(self, workspace_dir):
        sm = SessionManager("test-agent", workspace_dir)
        path = sm._get_session_file("justyn")
        assert path.name == "justyn.json"


class TestNoIdentityLinksBackwardCompat:
    """Empty identity_links = current behavior."""

    def test_default_empty(self):
        config = DaemonConfig(
            agents={"test": AgentConfig(workspace_dir=Path("/tmp/ws"), agent_file="default")},
        )
        assert config.identity_links == {}

    def test_identity_links_parsed(self):
        config = DaemonConfig(
            agents={"test": AgentConfig(workspace_dir=Path("/tmp/ws"), agent_file="default")},
            identity_links={"justyn": ["discord:123", "http:justyn"]},
        )
        assert config.identity_links["justyn"] == ["discord:123", "http:justyn"]


class TestHTTPAdapterResolveUser:
    """HTTPAgentAdapter.resolve_http_user resolves identity for direct session manager calls."""

    def test_resolve_linked(self, workspace_dir, identity_map):
        from tsugite.daemon.adapters.http import HTTPAgentAdapter

        agent_config = AgentConfig(workspace_dir=workspace_dir, agent_file="default")
        sm = SessionManager("test-agent", workspace_dir)
        adapter = HTTPAgentAdapter("test-agent", agent_config, sm, identity_map=identity_map)

        assert adapter.resolve_http_user("justyn") == "justyn"
        assert adapter.resolve_http_user("web-anonymous") == "justyn"

    def test_resolve_unlinked(self, workspace_dir, identity_map):
        from tsugite.daemon.adapters.http import HTTPAgentAdapter

        agent_config = AgentConfig(workspace_dir=workspace_dir, agent_file="default")
        sm = SessionManager("test-agent", workspace_dir)
        adapter = HTTPAgentAdapter("test-agent", agent_config, sm, identity_map=identity_map)

        assert adapter.resolve_http_user("someone-else") == "someone-else"

    def test_resolve_no_map(self, workspace_dir):
        from tsugite.daemon.adapters.http import HTTPAgentAdapter

        agent_config = AgentConfig(workspace_dir=workspace_dir, agent_file="default")
        sm = SessionManager("test-agent", workspace_dir)
        adapter = HTTPAgentAdapter("test-agent", agent_config, sm)

        assert adapter.resolve_http_user("web-anonymous") == "web-anonymous"


class TestSessionManagerThreadSafety:
    """SessionManager operations are thread-safe."""

    def test_concurrent_get_or_create(self, workspace_dir):
        import threading

        sm = SessionManager("test-agent", workspace_dir)
        results = {}

        def get_session(user_id):
            results[user_id] = sm.get_or_create_session(user_id)

        threads = [threading.Thread(target=get_session, args=(f"user-{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert len(set(results.values())) == 10  # All unique conv IDs
