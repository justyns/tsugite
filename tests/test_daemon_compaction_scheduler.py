"""Tests for scheduled auto-compaction."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import yaml

from tsugite.daemon.compaction_scheduler import CompactionScheduler
from tsugite.daemon.config import AgentConfig, AutoCompactConfig, load_daemon_config
from tsugite.daemon.session_store import SessionStore


def _make_session_store(tmp_path, agent="test-agent"):
    return SessionStore(tmp_path / "session_store.json", context_limits={agent: 128000})


def _make_adapter():
    adapter = MagicMock()
    adapter._compact_session = AsyncMock()
    return adapter


def _make_scheduler(tmp_path, adapter, store, min_turns=1):
    agent_config = AgentConfig(
        workspace_dir=tmp_path,
        agent_file="default",
        auto_compact=AutoCompactConfig(schedule="0 0 * * *", min_turns=min_turns),
    )
    return CompactionScheduler({"test-agent": agent_config}, store, {"test-agent": adapter}), agent_config


class TestAutoCompactConfig:
    def test_defaults(self):
        cfg = AutoCompactConfig()
        assert cfg.schedule is None
        assert cfg.min_turns == 1

    def test_with_schedule(self):
        cfg = AutoCompactConfig(schedule="0 0 * * *", min_turns=5)
        assert cfg.schedule == "0 0 * * *"
        assert cfg.min_turns == 5


class TestAgentConfigAutoCompact:
    def test_none_by_default(self):
        cfg = AgentConfig(workspace_dir=Path("/tmp"), agent_file="default")
        assert cfg.auto_compact is None

    def test_with_auto_compact(self):
        cfg = AgentConfig(
            workspace_dir=Path("/tmp"),
            agent_file="default",
            auto_compact=AutoCompactConfig(schedule="0 0 * * *"),
        )
        assert cfg.auto_compact.schedule == "0 0 * * *"


class TestConfigParsing:
    def test_load_auto_compact_from_yaml(self, tmp_path):
        config_data = {
            "agents": {
                "odyn": {
                    "workspace_dir": str(tmp_path),
                    "agent_file": "assistant.md",
                    "auto_compact": {
                        "schedule": "0 0 * * *",
                        "min_turns": 3,
                    },
                }
            },
        }
        config_file = tmp_path / "daemon.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_daemon_config(config_file)
        assert config.agents["odyn"].auto_compact is not None
        assert config.agents["odyn"].auto_compact.schedule == "0 0 * * *"
        assert config.agents["odyn"].auto_compact.min_turns == 3

    def test_load_without_auto_compact(self, tmp_path):
        config_data = {
            "agents": {
                "odyn": {
                    "workspace_dir": str(tmp_path),
                    "agent_file": "assistant.md",
                }
            },
        }
        config_file = tmp_path / "daemon.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_daemon_config(config_file)
        assert config.agents["odyn"].auto_compact is None


class TestCompactionScheduler:
    def test_check_agent_skips_low_turns(self, tmp_path):
        store = _make_session_store(tmp_path)
        store.get_or_create_interactive("user1", "test-agent")

        adapter = _make_adapter()
        scheduler, agent_config = _make_scheduler(tmp_path, adapter, store, min_turns=5)

        # message_count is 0, min_turns is 5 → should skip
        asyncio.get_event_loop().run_until_complete(scheduler._check_agent("test-agent", agent_config))
        adapter._compact_session.assert_not_called()

    def test_check_agent_compacts_when_enough_turns(self, tmp_path):
        store = _make_session_store(tmp_path)
        session = store.get_or_create_interactive("user1", "test-agent")
        # Simulate enough message activity
        for _ in range(5):
            store.update_token_count(session.id, 100)

        adapter = _make_adapter()
        scheduler, agent_config = _make_scheduler(tmp_path, adapter, store)

        asyncio.get_event_loop().run_until_complete(scheduler._check_agent("test-agent", agent_config))
        adapter._compact_session.assert_called_once_with(session.id, reason="scheduled")

    def test_check_agent_skips_if_already_compacting(self, tmp_path):
        store = _make_session_store(tmp_path)
        session = store.get_or_create_interactive("user1", "test-agent")
        for _ in range(5):
            store.update_token_count(session.id, 100)
        store.begin_compaction("user1", "test-agent")

        adapter = _make_adapter()
        scheduler, agent_config = _make_scheduler(tmp_path, adapter, store)

        asyncio.get_event_loop().run_until_complete(scheduler._check_agent("test-agent", agent_config))
        adapter._compact_session.assert_not_called()

        store.end_compaction("user1", "test-agent")

    def test_check_agent_no_sessions(self, tmp_path):
        store = _make_session_store(tmp_path)
        adapter = _make_adapter()
        scheduler, agent_config = _make_scheduler(tmp_path, adapter, store)

        asyncio.get_event_loop().run_until_complete(scheduler._check_agent("test-agent", agent_config))
        adapter._compact_session.assert_not_called()

    def test_check_agent_no_adapter(self, tmp_path):
        store = _make_session_store(tmp_path)
        session = store.get_or_create_interactive("user1", "test-agent")
        for _ in range(5):
            store.update_token_count(session.id, 100)

        agent_config = AgentConfig(
            workspace_dir=tmp_path,
            agent_file="default",
            auto_compact=AutoCompactConfig(schedule="0 0 * * *", min_turns=1),
        )
        scheduler = CompactionScheduler({"test-agent": agent_config}, store, {})

        asyncio.get_event_loop().run_until_complete(scheduler._check_agent("test-agent", agent_config))

    def test_compaction_failure_handled(self, tmp_path):
        store = _make_session_store(tmp_path)
        session = store.get_or_create_interactive("user1", "test-agent")
        for _ in range(5):
            store.update_token_count(session.id, 100)

        adapter = _make_adapter()
        adapter._compact_session.side_effect = RuntimeError("boom")
        scheduler, agent_config = _make_scheduler(tmp_path, adapter, store)

        asyncio.get_event_loop().run_until_complete(scheduler._check_agent("test-agent", agent_config))

        assert not store.is_compacting("user1", "test-agent")


class TestSessionStoreListInteractive:
    def test_list_interactive_by_agent(self, tmp_path):
        store = _make_session_store(tmp_path)
        store.get_or_create_interactive("user1", "test-agent")
        store.get_or_create_interactive("user2", "test-agent")
        store.get_or_create_interactive("user1", "other-agent")

        sessions = store.list_interactive_by_agent("test-agent")
        assert len(sessions) == 2
        assert all(s.agent == "test-agent" for s in sessions)

    def test_list_interactive_empty(self, tmp_path):
        store = _make_session_store(tmp_path)
        sessions = store.list_interactive_by_agent("nonexistent")
        assert sessions == []
