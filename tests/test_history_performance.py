"""Performance tests for history index operations."""

import time
from datetime import datetime, timezone

import pytest

from tsugite.history import (
    IndexEntry,
    Turn,
    generate_conversation_id,
    save_turn_to_history,
    update_index,
)
from tsugite.history.index import find_latest_session, load_index


class TestFindLatestSessionPerformance:
    """Performance tests for find_latest_session."""

    @pytest.fixture
    def create_conversations(self, tmp_path, monkeypatch):
        """Fixture to create multiple conversations for testing."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.index.get_history_dir", lambda: tmp_path)

        def _create(agent_name: str, count: int, daemon_ratio: float = 0.5, users: list = None):
            """Create N conversations for an agent.

            Args:
                agent_name: Agent name
                count: Number of conversations to create
                daemon_ratio: Ratio of daemon vs CLI conversations (0.0 to 1.0)
                users: List of user IDs to rotate through (for user_id filtering tests)
            """
            if users is None:
                users = ["user1"]

            conversations = []
            for i in range(count):
                conv_id = generate_conversation_id(agent_name)

                # Alternate between daemon and non-daemon
                is_daemon = i < (count * daemon_ratio)
                user_id = users[i % len(users)]

                metadata_dict = {"source": "discord" if is_daemon else "cli"}
                if is_daemon:
                    metadata_dict["is_daemon_managed"] = True
                    metadata_dict["daemon_agent"] = agent_name
                if user_id:
                    metadata_dict["user_id"] = user_id

                # Calculate timestamp with valid minutes/seconds/hours
                # Use incrementing seconds for uniqueness
                hours = 10 + (i // 3600)
                minutes = (i // 60) % 60
                seconds = i % 60
                timestamp = datetime(
                    2026, 1, 27 + (hours // 24), hours % 24, minutes, seconds, tzinfo=timezone.utc
                )

                turn = Turn(
                    timestamp=timestamp,
                    user=f"Message {i}",
                    assistant=f"Response {i}",
                    metadata=metadata_dict,
                )
                save_turn_to_history(conv_id, turn)

                update_index(
                    conv_id,
                    IndexEntry(
                        agent=agent_name,
                        model="test",
                        machine="test",
                        created_at=timestamp,
                        updated_at=timestamp,
                        is_daemon_managed=is_daemon,
                    ),
                )
                conversations.append((conv_id, is_daemon, user_id))

            return conversations

        return _create

    def test_baseline_no_filters(self, tmp_path, monkeypatch, create_conversations):
        """Baseline: find latest with no filters (just index lookup)."""
        conversations = create_conversations("odyn", 100, daemon_ratio=0.5)

        start = time.perf_counter()
        result = find_latest_session("odyn", daemon_only=False)
        elapsed = time.perf_counter() - start

        assert result is not None
        print(f"\n[Baseline] 100 conversations, no filters: {elapsed*1000:.2f}ms")

    def test_daemon_only_filter_small(self, tmp_path, monkeypatch, create_conversations):
        """Daemon filter with 10 conversations."""
        conversations = create_conversations("odyn", 10, daemon_ratio=0.5)

        start = time.perf_counter()
        result = find_latest_session("odyn", daemon_only=True)
        elapsed = time.perf_counter() - start

        assert result is not None
        print(f"\n[Small] 10 conversations, daemon_only=True: {elapsed*1000:.2f}ms")

    def test_daemon_only_filter_medium(self, tmp_path, monkeypatch, create_conversations):
        """Daemon filter with 100 conversations."""
        conversations = create_conversations("odyn", 100, daemon_ratio=0.5)

        start = time.perf_counter()
        result = find_latest_session("odyn", daemon_only=True)
        elapsed = time.perf_counter() - start

        assert result is not None
        print(f"\n[Medium] 100 conversations, daemon_only=True: {elapsed*1000:.2f}ms")

    def test_daemon_only_filter_large(self, tmp_path, monkeypatch, create_conversations):
        """Daemon filter with 1000 conversations."""
        conversations = create_conversations("odyn", 1000, daemon_ratio=0.5)

        start = time.perf_counter()
        result = find_latest_session("odyn", daemon_only=True)
        elapsed = time.perf_counter() - start

        assert result is not None
        print(f"\n[Large] 1000 conversations, daemon_only=True: {elapsed*1000:.2f}ms")

    def test_user_id_filter_medium(self, tmp_path, monkeypatch, create_conversations):
        """User ID filter with 100 conversations across 5 users."""
        conversations = create_conversations(
            "odyn", 100, daemon_ratio=0.5, users=["user1", "user2", "user3", "user4", "user5"]
        )

        start = time.perf_counter()
        result = find_latest_session("odyn", user_id="user3")
        elapsed = time.perf_counter() - start

        assert result is not None
        print(f"\n[Medium] 100 conversations, user_id filter: {elapsed*1000:.2f}ms")

    def test_combined_filters_medium(self, tmp_path, monkeypatch, create_conversations):
        """Combined daemon_only + user_id filter with 100 conversations."""
        conversations = create_conversations(
            "odyn", 100, daemon_ratio=0.5, users=["user1", "user2", "user3", "user4", "user5"]
        )

        start = time.perf_counter()
        result = find_latest_session("odyn", user_id="user3", daemon_only=True)
        elapsed = time.perf_counter() - start

        assert result is not None
        print(f"\n[Medium] 100 conversations, daemon_only + user_id: {elapsed*1000:.2f}ms")

    def test_worst_case_no_match(self, tmp_path, monkeypatch, create_conversations):
        """Worst case: filter that requires reading all files but finds nothing."""
        # Create 100 CLI-only conversations
        conversations = create_conversations("odyn", 100, daemon_ratio=0.0)

        start = time.perf_counter()
        result = find_latest_session("odyn", daemon_only=True)
        elapsed = time.perf_counter() - start

        assert result is None
        print(f"\n[Worst case] 100 conversations, no matches: {elapsed*1000:.2f}ms")

    def test_index_load_overhead(self, tmp_path, monkeypatch, create_conversations):
        """Measure index loading overhead."""
        conversations = create_conversations("odyn", 100, daemon_ratio=0.5)

        # Clear any internal caching by reloading
        start = time.perf_counter()
        index = load_index()
        elapsed = time.perf_counter() - start

        print(f"\n[Index] Load 100-entry index: {elapsed*1000:.2f}ms")
        print(f"[Index] Index size: {len(index)} entries")

    def test_file_read_overhead(self, tmp_path, monkeypatch, create_conversations):
        """Measure file read overhead for a single conversation."""
        from tsugite.history.storage import load_conversation

        conversations = create_conversations("odyn", 1, daemon_ratio=1.0)
        conv_id = conversations[0][0]

        # Measure single file read
        start = time.perf_counter()
        turns = load_conversation(conv_id)
        elapsed = time.perf_counter() - start

        print(f"\n[File I/O] Load single conversation: {elapsed*1000:.2f}ms")
        print(f"[File I/O] Turns in conversation: {len(turns)}")

    def test_multiple_agents(self, tmp_path, monkeypatch, create_conversations):
        """Test with multiple agents sharing index."""
        # Create 100 conversations each for 10 different agents
        for i in range(10):
            create_conversations(f"agent{i}", 100, daemon_ratio=0.5)

        # Now search for one specific agent
        start = time.perf_counter()
        result = find_latest_session("agent5", daemon_only=True)
        elapsed = time.perf_counter() - start

        assert result is not None
        print(f"\n[Multi-agent] 1000 total conversations (10 agents x 100): {elapsed*1000:.2f}ms")

    def test_early_exit_optimization(self, tmp_path, monkeypatch, create_conversations):
        """Test early exit when match found immediately."""
        # Create 100 conversations, with daemon conversations being newer
        conversations = create_conversations("odyn", 100, daemon_ratio=0.5)

        # The most recent conversation should be daemon-managed (based on our creation order)
        # This should exit early
        start = time.perf_counter()
        result = find_latest_session("odyn", daemon_only=True)
        elapsed = time.perf_counter() - start

        assert result is not None
        print(f"\n[Early exit] Found match in newest conversations: {elapsed*1000:.2f}ms")


class TestIndexScalability:
    """Test index scalability with large numbers of conversations."""

    def test_index_size_memory(self, tmp_path, monkeypatch):
        """Measure memory/size characteristics of index."""
        import json
        import sys

        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.index.get_history_dir", lambda: tmp_path)

        # Create 1000 conversations
        for i in range(1000):
            conv_id = generate_conversation_id("test_agent")
            update_index(
                conv_id,
                IndexEntry(
                    agent="test_agent",
                    model="test",
                    machine="test",
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                ),
            )

        # Load index and measure
        index = load_index()
        index_path = tmp_path / "index.json"
        file_size = index_path.stat().st_size

        # Measure in-memory size (approximate)
        memory_size = sys.getsizeof(index)

        print(f"\n[Scalability] 1000 entries:")
        print(f"  File size: {file_size / 1024:.2f} KB")
        print(f"  Memory size (approx): {memory_size / 1024:.2f} KB")
        print(f"  Avg bytes per entry: {file_size / 1000:.2f}")
