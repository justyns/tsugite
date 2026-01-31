"""Performance tests for history operations."""

import time
from datetime import datetime, timezone

import pytest

from tsugite.history import (
    SessionStorage,
    list_session_files,
)


class TestSessionStoragePerformance:
    """Performance tests for SessionStorage."""

    @pytest.fixture
    def create_sessions(self, tmp_path, monkeypatch):
        """Fixture to create multiple sessions for testing."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test_machine")

        def _create(agent_name: str, count: int, metadata: dict = None):
            """Create N sessions for an agent.

            Args:
                agent_name: Agent name
                count: Number of sessions to create
                metadata: Optional metadata to add to turns
            """
            sessions = []
            for i in range(count):
                # Calculate timestamp with valid minutes/seconds/hours
                hours = 10 + (i // 3600)
                minutes = (i // 60) % 60
                seconds = i % 60
                timestamp = datetime(2026, 1, 27 + (hours // 24), hours % 24, minutes, seconds, tzinfo=timezone.utc)

                storage = SessionStorage.create(
                    agent_name=agent_name,
                    model="test",
                    timestamp=timestamp,
                )

                messages = [
                    {"role": "user", "content": f"Message {i}"},
                    {"role": "assistant", "content": f"Response {i}"},
                ]
                storage.record_turn(
                    messages=messages,
                    final_answer=f"Response {i}",
                    metadata=metadata,
                )
                sessions.append(storage.session_id)

            return sessions

        return _create

    def test_baseline_no_filters(self, tmp_path, monkeypatch, create_sessions):
        """Baseline: list sessions with no filters."""
        create_sessions("odyn", 100)

        start = time.perf_counter()
        files = list_session_files()
        elapsed = time.perf_counter() - start

        assert len(files) == 100
        print(f"\n[Baseline] 100 sessions, list files: {elapsed * 1000:.2f}ms")

    def test_small_session_count(self, tmp_path, monkeypatch, create_sessions):
        """List with 10 sessions."""
        create_sessions("odyn", 10)

        start = time.perf_counter()
        files = list_session_files()
        elapsed = time.perf_counter() - start

        assert len(files) == 10
        print(f"\n[Small] 10 sessions: {elapsed * 1000:.2f}ms")

    def test_medium_session_count(self, tmp_path, monkeypatch, create_sessions):
        """List with 100 sessions."""
        create_sessions("odyn", 100)

        start = time.perf_counter()
        files = list_session_files()
        elapsed = time.perf_counter() - start

        assert len(files) == 100
        print(f"\n[Medium] 100 sessions: {elapsed * 1000:.2f}ms")

    def test_large_session_count(self, tmp_path, monkeypatch, create_sessions):
        """List with 1000 sessions."""
        create_sessions("odyn", 1000)

        start = time.perf_counter()
        files = list_session_files()
        elapsed = time.perf_counter() - start

        assert len(files) == 1000
        print(f"\n[Large] 1000 sessions: {elapsed * 1000:.2f}ms")

    def test_session_load_overhead(self, tmp_path, monkeypatch, create_sessions):
        """Measure session load overhead."""
        sessions = create_sessions("odyn", 1)
        session_id = sessions[0]

        session_path = tmp_path / f"{session_id}.jsonl"

        start = time.perf_counter()
        storage = SessionStorage.load(session_path)
        elapsed = time.perf_counter() - start

        print(f"\n[Load] Load single session: {elapsed * 1000:.2f}ms")
        print(f"[Load] Turn count: {storage.turn_count}")

    def test_record_load_overhead(self, tmp_path, monkeypatch, create_sessions):
        """Measure record loading overhead."""
        sessions = create_sessions("odyn", 1)
        session_id = sessions[0]

        session_path = tmp_path / f"{session_id}.jsonl"
        storage = SessionStorage.load(session_path)

        start = time.perf_counter()
        records = storage.load_records()
        elapsed = time.perf_counter() - start

        print(f"\n[Records] Load records: {elapsed * 1000:.2f}ms")
        print(f"[Records] Record count: {len(records)}")

    def test_multiple_agents(self, tmp_path, monkeypatch, create_sessions):
        """Test with multiple agents."""
        # Create 100 sessions each for 10 different agents
        for i in range(10):
            create_sessions(f"agent{i}", 100)

        start = time.perf_counter()
        files = list_session_files()
        elapsed = time.perf_counter() - start

        assert len(files) == 1000
        print(f"\n[Multi-agent] 1000 total sessions (10 agents x 100): {elapsed * 1000:.2f}ms")


class TestSessionScalability:
    """Test session scalability with large numbers of sessions."""

    def test_session_file_size(self, tmp_path, monkeypatch):
        """Measure file size characteristics."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test_machine")

        storage = SessionStorage.create(agent_name="test_agent", model="test")

        # Add 100 turns
        for i in range(100):
            messages = [
                {"role": "user", "content": f"Question {i}"},
                {"role": "assistant", "content": f"Answer {i}"},
            ]
            storage.record_turn(
                messages=messages,
                final_answer=f"Answer {i}",
                tokens=50,
                cost=0.001,
            )

        file_size = storage.session_path.stat().st_size

        print("\n[Scalability] 100 turns in single session:")
        print(f"  File size: {file_size / 1024:.2f} KB")
        print(f"  Avg bytes per turn: {file_size / 100:.2f}")

    def test_reconstruction_performance(self, tmp_path, monkeypatch):
        """Measure message reconstruction performance."""
        from tsugite.history import reconstruct_messages

        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test_machine")

        storage = SessionStorage.create(agent_name="test_agent", model="test")

        # Add 50 turns
        for i in range(50):
            messages = [
                {"role": "user", "content": f"Question {i}"},
                {"role": "assistant", "content": f"Answer {i}"},
            ]
            storage.record_turn(
                messages=messages,
                final_answer=f"Answer {i}",
            )

        start = time.perf_counter()
        messages = reconstruct_messages(storage.session_path)
        elapsed = time.perf_counter() - start

        print("\n[Reconstruction] 50 turns:")
        print(f"  Time: {elapsed * 1000:.2f}ms")
        print(f"  Messages: {len(messages)}")
