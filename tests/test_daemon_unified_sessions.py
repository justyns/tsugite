"""Tests for unified sessions across daemon adapters."""

import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from tsugite.daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite.daemon.config import AgentConfig, DaemonConfig
from tsugite.daemon.session_store import SessionStore


class _StubAdapter(BaseAdapter):
    async def start(self):
        pass

    async def stop(self):
        pass


@pytest.fixture
def workspace_dir(tmp_path):
    return tmp_path / "workspace"


@pytest.fixture
def session_store(tmp_path):
    return SessionStore(tmp_path / "session_store.json", context_limits={"test-agent": 128000})


@pytest.fixture
def identity_map():
    return {
        "discord:123456789": "justyn",
        "http:justyn": "justyn",
        "http:web-anonymous": "justyn",
    }


def _make_adapter(workspace_dir, session_store, identity_map=None):
    agent_config = AgentConfig(workspace_dir=workspace_dir, agent_file="default")
    return _StubAdapter("test-agent", agent_config, session_store, identity_map=identity_map)


class TestIdentityResolution:
    """Tests for identity map lookup in BaseAdapter.resolve_user."""

    def test_linked_user_resolves_to_canonical(self, workspace_dir, session_store, identity_map):
        adapter = _make_adapter(workspace_dir, session_store, identity_map)

        ctx = ChannelContext(source="discord", channel_id="dm-chan", user_id="123456789", reply_to="discord:dm-chan")
        assert adapter.resolve_user("123456789", ctx) == "justyn"

        ctx_http = ChannelContext(source="http", channel_id=None, user_id="justyn", reply_to="http:justyn")
        assert adapter.resolve_user("justyn", ctx_http) == "justyn"

    def test_unlinked_user_gets_bare_id(self, workspace_dir, session_store, identity_map):
        adapter = _make_adapter(workspace_dir, session_store, identity_map)

        ctx = ChannelContext(source="discord", channel_id="dm-chan", user_id="999999", reply_to="discord:dm-chan")
        assert adapter.resolve_user("999999", ctx) == "999999"

    def test_empty_identity_map_passthrough(self, workspace_dir, session_store):
        adapter = _make_adapter(workspace_dir, session_store)

        ctx = ChannelContext(source="discord", channel_id="dm-chan", user_id="123456789", reply_to="discord:dm-chan")
        assert adapter.resolve_user("123456789", ctx) == "123456789"


class TestGroupIsolation:
    """Group chats produce composite keys, not unified identities."""

    def test_group_chat_isolated(self, workspace_dir, session_store, identity_map):
        adapter = _make_adapter(workspace_dir, session_store, identity_map)

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

    def test_same_user_dm_vs_group_different(self, workspace_dir, session_store, identity_map):
        adapter = _make_adapter(workspace_dir, session_store, identity_map)

        dm_ctx = ChannelContext(source="discord", channel_id="dm-chan", user_id="123456789", reply_to="discord:dm-chan")
        group_ctx = ChannelContext(
            source="discord",
            channel_id="guild-chan",
            user_id="123456789",
            reply_to="discord:guild-chan",
            metadata={"guild_id": "777"},
        )
        assert adapter.resolve_user("123456789", dm_ctx) != adapter.resolve_user("123456789", group_ctx)


class TestSharedSessionStore:
    """Two adapters sharing a SessionStore get the same conversation_id for linked users."""

    def test_shared_session_same_conv_id(self, workspace_dir, tmp_path, identity_map):
        shared_store = SessionStore(tmp_path / "session_store.json", context_limits={"test-agent": 128000})

        discord_adapter = _make_adapter(workspace_dir, shared_store, identity_map)
        http_adapter = _make_adapter(workspace_dir, shared_store, identity_map)

        discord_ctx = ChannelContext(
            source="discord", channel_id="dm-chan", user_id="123456789", reply_to="discord:dm-chan"
        )
        http_ctx = ChannelContext(source="http", channel_id=None, user_id="justyn", reply_to="http:justyn")

        discord_user = discord_adapter.resolve_user("123456789", discord_ctx)
        http_user = http_adapter.resolve_user("justyn", http_ctx)

        assert discord_user == http_user == "justyn"

        session1 = shared_store.get_or_create_interactive(discord_user, "test-agent")
        session2 = shared_store.get_or_create_interactive(http_user, "test-agent")
        assert session1.id == session2.id


class TestSessionStoreInteractive:
    """Tests for interactive session management in SessionStore."""

    def test_get_or_create_interactive(self, tmp_path):
        store = SessionStore(tmp_path / "session_store.json")
        session = store.get_or_create_interactive("user1", "agent1")
        assert session.id == "daemon_agent1_user1"
        assert session.source == "interactive"
        assert session.user_id == "user1"

    def test_get_or_create_interactive_idempotent(self, tmp_path):
        store = SessionStore(tmp_path / "session_store.json")
        s1 = store.get_or_create_interactive("user1", "agent1")
        s2 = store.get_or_create_interactive("user1", "agent1")
        assert s1.id == s2.id

    def test_different_agents_different_sessions(self, tmp_path):
        store = SessionStore(tmp_path / "session_store.json")
        s1 = store.get_or_create_interactive("user1", "agent1")
        s2 = store.get_or_create_interactive("user1", "agent2")
        assert s1.id != s2.id


class TestSessionStoreSkillSuppression:
    """Per-session skill suppression tracked on SessionStore."""

    def test_default_empty(self, tmp_path):
        store = SessionStore(tmp_path / "session_store.json")
        assert store.get_suppressed_skills("any-session") == set()

    def test_suppress_adds_to_set(self, tmp_path):
        store = SessionStore(tmp_path / "session_store.json")
        store.suppress_skill("session-1", "skill-a")
        store.suppress_skill("session-1", "skill-b")
        assert store.get_suppressed_skills("session-1") == {"skill-a", "skill-b"}

    def test_suppression_isolated_per_session(self, tmp_path):
        store = SessionStore(tmp_path / "session_store.json")
        store.suppress_skill("session-1", "skill-a")
        assert store.get_suppressed_skills("session-2") == set()

    def test_unsuppress_removes(self, tmp_path):
        store = SessionStore(tmp_path / "session_store.json")
        store.suppress_skill("session-1", "skill-a")
        store.unsuppress_skill("session-1", "skill-a")
        assert store.get_suppressed_skills("session-1") == set()

    def test_get_returns_copy(self, tmp_path):
        store = SessionStore(tmp_path / "session_store.json")
        store.suppress_skill("session-1", "skill-a")
        snapshot = store.get_suppressed_skills("session-1")
        snapshot.add("skill-b")
        assert store.get_suppressed_skills("session-1") == {"skill-a"}

    def test_suppress_same_skill_is_idempotent(self, tmp_path):
        store = SessionStore(tmp_path / "session_store.json")
        store.suppress_skill("session-1", "skill-a")
        store.suppress_skill("session-1", "skill-a")
        assert store.get_suppressed_skills("session-1") == {"skill-a"}

    def test_unsuppress_unknown_is_noop(self, tmp_path):
        store = SessionStore(tmp_path / "session_store.json")
        # unsuppress on a session with no entry at all
        store.unsuppress_skill("ghost-session", "skill-a")
        assert store.get_suppressed_skills("ghost-session") == set()
        # unsuppress on a session with a different entry
        store.suppress_skill("session-1", "skill-a")
        store.unsuppress_skill("session-1", "skill-b")
        assert store.get_suppressed_skills("session-1") == {"skill-a"}

    def test_unsuppress_cleans_empty_session_entry(self, tmp_path):
        """Empty-set entries shouldn't accumulate across many sessions."""
        store = SessionStore(tmp_path / "session_store.json")
        store.suppress_skill("session-1", "skill-a")
        store.unsuppress_skill("session-1", "skill-a")
        assert "session-1" not in store._suppressed_skills

    def test_suppression_is_not_persisted_across_reload(self, tmp_path):
        """In-memory only: a fresh SessionStore loses prior suppression."""
        path = tmp_path / "session_store.json"
        first = SessionStore(path)
        first.suppress_skill("session-1", "skill-a")
        assert first.get_suppressed_skills("session-1") == {"skill-a"}

        second = SessionStore(path)
        assert second.get_suppressed_skills("session-1") == set()

    def test_suppression_carries_through_compaction(self, tmp_path):
        """Compacting a session migrates suppressions to the new session_id and
        drops the old entry so the dict doesn't leak orphaned keys."""
        store = SessionStore(tmp_path / "session_store.json")
        session = store.get_or_create_interactive("alice", "test-agent")
        store.suppress_skill(session.id, "skill-a")
        store.suppress_skill(session.id, "skill-b")

        new_session = store.compact_session(session.id)

        assert new_session.id != session.id
        assert store.get_suppressed_skills(new_session.id) == {"skill-a", "skill-b"}
        assert session.id not in store._suppressed_skills

    def test_compaction_without_suppression_is_noop(self, tmp_path):
        """Compacting a session that never had suppressions leaves the dict empty."""
        store = SessionStore(tmp_path / "session_store.json")
        session = store.get_or_create_interactive("alice", "test-agent")
        new_session = store.compact_session(session.id)

        assert store.get_suppressed_skills(new_session.id) == set()
        assert new_session.id not in store._suppressed_skills

    def test_suppress_is_thread_safe(self, tmp_path):
        """Concurrent suppress calls converge without lost updates."""
        import threading

        store = SessionStore(tmp_path / "session_store.json")
        names = [f"skill-{i}" for i in range(40)]

        def worker(subset):
            for name in subset:
                store.suppress_skill("session-1", name)

        mid = len(names) // 2
        t1 = threading.Thread(target=worker, args=(names[:mid],))
        t2 = threading.Thread(target=worker, args=(names[mid:],))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert store.get_suppressed_skills("session-1") == set(names)


class TestSessionStoreStickySkills:
    """Per-session sticky skills + TTL counter state on SessionStore."""

    def test_default_empty(self, tmp_path):
        store = SessionStore(tmp_path / "session_store.json")
        assert store.get_sticky_skills("any") == {}

    def test_mark_sticky_creates_entry_with_zero_counter(self, tmp_path):
        store = SessionStore(tmp_path / "session_store.json")
        store.mark_sticky("session-1", "skill-a")
        assert store.get_sticky_skills("session-1") == {"skill-a": 0}

    def test_mark_sticky_resets_counter(self, tmp_path):
        store = SessionStore(tmp_path / "session_store.json")
        store.mark_sticky("session-1", "skill-a")
        store.bump_unused_counters("session-1", referenced=set())
        assert store.get_sticky_skills("session-1") == {"skill-a": 1}
        store.mark_sticky("session-1", "skill-a")
        assert store.get_sticky_skills("session-1") == {"skill-a": 0}

    def test_bump_increments_unreferenced_resets_referenced(self, tmp_path):
        store = SessionStore(tmp_path / "session_store.json")
        store.mark_sticky("session-1", "stale")
        store.mark_sticky("session-1", "fresh")
        store.bump_unused_counters("session-1", referenced={"fresh"})
        assert store.get_sticky_skills("session-1") == {"stale": 1, "fresh": 0}

    def test_bump_with_no_sticky_is_noop(self, tmp_path):
        store = SessionStore(tmp_path / "session_store.json")
        store.bump_unused_counters("session-1", referenced={"anything"})
        assert store.get_sticky_skills("session-1") == {}

    def test_drop_sticky(self, tmp_path):
        store = SessionStore(tmp_path / "session_store.json")
        store.mark_sticky("session-1", "skill-a")
        store.drop_sticky("session-1", "skill-a")
        assert store.get_sticky_skills("session-1") == {}
        assert "session-1" not in store._sticky_skills

    def test_drop_sticky_unknown_is_noop(self, tmp_path):
        store = SessionStore(tmp_path / "session_store.json")
        store.drop_sticky("ghost", "skill-a")
        store.mark_sticky("session-1", "skill-a")
        store.drop_sticky("session-1", "skill-b")
        assert store.get_sticky_skills("session-1") == {"skill-a": 0}

    def test_isolated_per_session(self, tmp_path):
        store = SessionStore(tmp_path / "session_store.json")
        store.mark_sticky("session-1", "skill-a")
        assert store.get_sticky_skills("session-2") == {}

    def test_get_returns_copy(self, tmp_path):
        store = SessionStore(tmp_path / "session_store.json")
        store.mark_sticky("session-1", "skill-a")
        snapshot = store.get_sticky_skills("session-1")
        snapshot["skill-b"] = 9
        assert store.get_sticky_skills("session-1") == {"skill-a": 0}

    def test_sticky_carries_through_compaction(self, tmp_path):
        store = SessionStore(tmp_path / "session_store.json")
        session = store.get_or_create_interactive("alice", "test-agent")
        store.mark_sticky(session.id, "skill-a")
        store.bump_unused_counters(session.id, referenced=set())

        new_session = store.compact_session(session.id)

        assert new_session.id != session.id
        assert store.get_sticky_skills(new_session.id) == {"skill-a": 1}
        assert session.id not in store._sticky_skills

    def test_sticky_not_persisted_across_reload(self, tmp_path):
        path = tmp_path / "session_store.json"
        first = SessionStore(path)
        first.mark_sticky("session-1", "skill-a")
        second = SessionStore(path)
        assert second.get_sticky_skills("session-1") == {}


class TestSessionStoreThreadSafety:
    """SessionStore operations are thread-safe."""

    def test_concurrent_get_or_create(self, tmp_path):
        import threading

        store = SessionStore(tmp_path / "session_store.json")
        results = {}

        def get_session(user_id):
            session = store.get_or_create_interactive(user_id, "test-agent")
            results[user_id] = session.id

        threads = [threading.Thread(target=get_session, args=(f"user-{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert len(set(results.values())) == 10


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


try:
    import cronsim  # noqa: F401

    _has_cronsim = True
except ImportError:
    _has_cronsim = False


@pytest.mark.skipif(not _has_cronsim, reason="cronsim not installed (required by http adapter)")
class TestHTTPAdapterResolveUser:
    """HTTPAgentAdapter.resolve_http_user resolves identity for direct session store calls."""

    def test_resolve_linked(self, workspace_dir, tmp_path, identity_map):
        from tsugite.daemon.adapters.http import HTTPAgentAdapter

        agent_config = AgentConfig(workspace_dir=workspace_dir, agent_file="default")
        store = SessionStore(tmp_path / "session_store.json")
        adapter = HTTPAgentAdapter("test-agent", agent_config, store, identity_map=identity_map)

        assert adapter.resolve_http_user("justyn") == "justyn"
        assert adapter.resolve_http_user("web-anonymous") == "justyn"

    def test_resolve_unlinked(self, workspace_dir, tmp_path, identity_map):
        from tsugite.daemon.adapters.http import HTTPAgentAdapter

        agent_config = AgentConfig(workspace_dir=workspace_dir, agent_file="default")
        store = SessionStore(tmp_path / "session_store.json")
        adapter = HTTPAgentAdapter("test-agent", agent_config, store, identity_map=identity_map)

        assert adapter.resolve_http_user("someone-else") == "someone-else"

    def test_resolve_no_map(self, workspace_dir, tmp_path):
        from tsugite.daemon.adapters.http import HTTPAgentAdapter

        agent_config = AgentConfig(workspace_dir=workspace_dir, agent_file="default")
        store = SessionStore(tmp_path / "session_store.json")
        adapter = HTTPAgentAdapter("test-agent", agent_config, store)

        assert adapter.resolve_http_user("web-anonymous") == "web-anonymous"


class TestCompactSessionClearsSkills:
    """Compaction should clear loaded skills so they get re-loaded in the new session."""

    @pytest.mark.asyncio
    async def test_compact_session_clears_loaded_skills(self, workspace_dir, tmp_path):
        from tsugite.history import SessionStorage
        from tsugite.history.models import Turn

        history_dir = tmp_path / "history"
        history_dir.mkdir()

        store = SessionStore(tmp_path / "session_store.json", context_limits={"test-agent": 128000})
        session = store.get_or_create_interactive("test-user", "test-agent")
        conv_id = session.id

        session_path = history_dir / f"{conv_id}.jsonl"
        storage = SessionStorage.create(
            agent_name="test-agent",
            model="openai:gpt-4o-mini",
            session_path=session_path,
        )
        for i in range(5):
            storage.record_turn(
                messages=[
                    {"role": "user", "content": f"message {i}"},
                    {"role": "assistant", "content": f"reply {i}"},
                ],
                final_answer=f"reply {i}",
            )

        adapter = _make_adapter(workspace_dir, store)

        from tsugite.tools.skills import get_skill_manager

        manager = get_skill_manager()
        manager._loaded_skills["test-skill"] = "skill content"

        with (
            patch("tsugite.daemon.memory.get_context_limit", return_value=128_000),
            patch("tsugite.daemon.memory.infer_compaction_model", return_value="openai:gpt-4o-mini"),
            patch("tsugite.daemon.memory.split_turns_for_compaction") as mock_split,
            patch("tsugite.daemon.memory.summarize_session", new_callable=AsyncMock, return_value="Summary"),
            patch("tsugite.history.get_history_dir", return_value=history_dir),
            patch("tsugite.history.storage.get_history_dir", return_value=history_dir),
            patch("tsugite.history.storage.get_machine_name", return_value="test"),
            patch("tsugite.hooks.fire_compact_hooks", new_callable=AsyncMock),
        ):
            old_turns = [
                Turn(
                    timestamp=datetime.now(timezone.utc),
                    messages=[{"role": "user", "content": "old"}],
                )
            ]
            recent_turns = [
                Turn(
                    timestamp=datetime.now(timezone.utc),
                    messages=[{"role": "user", "content": "recent"}],
                )
            ]
            mock_split.return_value = (old_turns, recent_turns)

            await adapter._compact_session(conv_id)

        assert manager._loaded_skills == {}


class TestCompactionLocking:
    """Tests for session compaction lock/queue mechanism."""

    def test_begin_compaction_returns_true_first_time(self, session_store):
        assert session_store.begin_compaction("user1", "test-agent") is True

    def test_begin_compaction_returns_false_if_already_compacting(self, session_store):
        session_store.begin_compaction("user1", "test-agent")
        assert session_store.begin_compaction("user1", "test-agent") is False

    def test_end_compaction_allows_new_begin(self, session_store):
        session_store.begin_compaction("user1", "test-agent")
        session_store.end_compaction("user1", "test-agent")
        assert session_store.begin_compaction("user1", "test-agent") is True

    def test_is_compacting(self, session_store):
        assert session_store.is_compacting("user1", "test-agent") is False
        session_store.begin_compaction("user1", "test-agent")
        assert session_store.is_compacting("user1", "test-agent") is True
        session_store.end_compaction("user1", "test-agent")
        assert session_store.is_compacting("user1", "test-agent") is False

    def test_wait_for_compaction_returns_immediately_if_not_compacting(self, session_store):
        assert session_store.wait_for_compaction("user1", "test-agent", timeout=0.1) is True

    def test_wait_for_compaction_blocks_until_done(self, session_store):
        session_store.begin_compaction("user1", "test-agent")
        result = [None]

        def waiter():
            result[0] = session_store.wait_for_compaction("user1", "test-agent", timeout=5)

        t = threading.Thread(target=waiter)
        t.start()
        session_store.end_compaction("user1", "test-agent")
        t.join(timeout=2)
        assert not t.is_alive()
        assert result[0] is True

    def test_wait_for_compaction_times_out(self, session_store):
        session_store.begin_compaction("user1", "test-agent")
        assert session_store.wait_for_compaction("user1", "test-agent", timeout=0.05) is False
        # Clean up
        session_store.end_compaction("user1", "test-agent")

    def test_different_users_independent(self, session_store):
        session_store.begin_compaction("user1", "test-agent")
        assert session_store.begin_compaction("user2", "test-agent") is True
        assert session_store.is_compacting("user1", "test-agent") is True
        assert session_store.is_compacting("user2", "test-agent") is True

    def test_end_compaction_idempotent(self, session_store):
        """end_compaction on a non-compacting session is a no-op."""
        session_store.end_compaction("user1", "test-agent")  # Should not raise
