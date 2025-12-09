"""Tests for the memory system."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Check if duckdb is available
try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

requires_duckdb = pytest.mark.skipif(not HAS_DUCKDB, reason="duckdb not installed")


class TestMemorySchema:
    """Tests for Memory dataclass."""

    def test_memory_creation(self):
        """Test creating a Memory object."""
        from tsugite.memory.schema import Memory

        memory = Memory(
            id=1,
            content="Test content",
            memory_type="fact",
            agent_name="test_agent",
            tags=["test", "example"],
            metadata={"source": "user"},
        )

        assert memory.id == 1
        assert memory.content == "Test content"
        assert memory.memory_type == "fact"
        assert memory.agent_name == "test_agent"
        assert memory.tags == ["test", "example"]
        assert memory.metadata == {"source": "user"}
        assert memory.score is None

    def test_memory_defaults(self):
        """Test Memory default values."""
        from tsugite.memory.schema import Memory

        memory = Memory(id=1, content="Test")

        assert memory.memory_type == "note"
        assert memory.agent_name is None
        assert memory.tags == []
        assert memory.metadata == {}
        assert memory.created_at is None
        assert memory.updated_at is None
        assert memory.score is None


class TestDateFilter:
    """Tests for date filter parsing."""

    def test_parse_relative_days(self):
        """Test parsing relative day format."""
        from datetime import datetime, timedelta

        from tsugite.memory.manager import parse_date_filter

        result = parse_date_filter("7d")
        expected = datetime.now() - timedelta(days=7)
        assert abs((result - expected).total_seconds()) < 1

    def test_parse_relative_weeks(self):
        """Test parsing relative week format."""
        from datetime import datetime, timedelta

        from tsugite.memory.manager import parse_date_filter

        result = parse_date_filter("2w")
        expected = datetime.now() - timedelta(weeks=2)
        assert abs((result - expected).total_seconds()) < 1

    def test_parse_relative_months(self):
        """Test parsing relative month format."""
        from datetime import datetime, timedelta

        from tsugite.memory.manager import parse_date_filter

        result = parse_date_filter("1m")
        expected = datetime.now() - timedelta(days=30)
        assert abs((result - expected).total_seconds()) < 1

    def test_parse_iso_date(self):
        """Test parsing ISO date format."""
        from datetime import datetime

        from tsugite.memory.manager import parse_date_filter

        result = parse_date_filter("2024-12-01")
        assert result == datetime(2024, 12, 1)


class TestEmbeddings:
    """Tests for embedding generation."""

    def test_get_embedding_dimension_known_model(self):
        """Test getting dimension for known models without loading."""
        from tsugite.memory.embeddings import get_embedding_dimension

        assert get_embedding_dimension("BAAI/bge-small-en-v1.5") == 384
        assert get_embedding_dimension("all-MiniLM-L6-v2") == 384


@requires_duckdb
class TestMemoryManager:
    """Tests for MemoryManager."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_memories.duckdb"

    @pytest.fixture
    def mock_embedding(self):
        """Mock embedding function to avoid loading model."""
        with patch("tsugite.memory.manager.MemoryManager._get_embedding") as mock:
            mock.return_value = [0.1] * 384
            yield mock

    def test_manager_init(self, temp_db, mock_embedding):
        """Test MemoryManager initialization."""
        from tsugite.memory.manager import MemoryManager

        manager = MemoryManager(db_path=temp_db)
        assert manager.db_path == temp_db
        assert manager.conn is not None
        manager.close()

    def test_store_memory(self, temp_db, mock_embedding):
        """Test storing a memory."""
        from tsugite.memory.manager import MemoryManager

        manager = MemoryManager(db_path=temp_db)
        memory_id = manager.store("Test content", memory_type="fact", tags=["test"])

        assert memory_id == 1
        mock_embedding.assert_called_once_with("Test content")
        manager.close()

    def test_get_memory(self, temp_db, mock_embedding):
        """Test retrieving a memory by ID."""
        from tsugite.memory.manager import MemoryManager

        manager = MemoryManager(db_path=temp_db)
        memory_id = manager.store("Test content", memory_type="fact")
        memory = manager.get(memory_id)

        assert memory is not None
        assert memory.id == memory_id
        assert memory.content == "Test content"
        assert memory.memory_type == "fact"
        manager.close()

    def test_get_nonexistent_memory(self, temp_db, mock_embedding):
        """Test retrieving a nonexistent memory."""
        from tsugite.memory.manager import MemoryManager

        manager = MemoryManager(db_path=temp_db)
        memory = manager.get(9999)
        assert memory is None
        manager.close()

    def test_update_memory(self, temp_db, mock_embedding):
        """Test updating a memory."""
        from tsugite.memory.manager import MemoryManager

        manager = MemoryManager(db_path=temp_db)
        memory_id = manager.store("Original content")
        success = manager.update(memory_id, "Updated content")

        assert success is True
        memory = manager.get(memory_id)
        assert memory.content == "Updated content"
        manager.close()

    def test_update_nonexistent_memory(self, temp_db, mock_embedding):
        """Test updating a nonexistent memory."""
        from tsugite.memory.manager import MemoryManager

        manager = MemoryManager(db_path=temp_db)
        success = manager.update(9999, "New content")
        assert success is False
        manager.close()

    def test_delete_memory(self, temp_db, mock_embedding):
        """Test deleting a memory."""
        from tsugite.memory.manager import MemoryManager

        manager = MemoryManager(db_path=temp_db)
        memory_id = manager.store("Test content")
        success = manager.delete(memory_id)

        assert success is True
        assert manager.get(memory_id) is None
        manager.close()

    def test_delete_nonexistent_memory(self, temp_db, mock_embedding):
        """Test deleting a nonexistent memory."""
        from tsugite.memory.manager import MemoryManager

        manager = MemoryManager(db_path=temp_db)
        success = manager.delete(9999)
        assert success is False
        manager.close()

    def test_count_memories(self, temp_db, mock_embedding):
        """Test counting memories."""
        from tsugite.memory.manager import MemoryManager

        manager = MemoryManager(db_path=temp_db)
        assert manager.count() == 0

        manager.store("Memory 1")
        manager.store("Memory 2")
        manager.store("Memory 3", agent_name="agent1")

        assert manager.count() == 3
        assert manager.count(agent_name="agent1") == 1
        manager.close()

    def test_list_recent(self, temp_db, mock_embedding):
        """Test listing recent memories."""
        from tsugite.memory.manager import MemoryManager

        manager = MemoryManager(db_path=temp_db)
        manager.store("Memory 1", memory_type="fact")
        manager.store("Memory 2", memory_type="note")
        manager.store("Memory 3", memory_type="fact", agent_name="agent1")

        all_memories = manager.list_recent()
        assert len(all_memories) == 3

        facts = manager.list_recent(memory_type="fact")
        assert len(facts) == 2

        agent_memories = manager.list_recent(agent_name="agent1")
        assert len(agent_memories) == 1
        manager.close()

    def test_search_semantic(self, temp_db, mock_embedding):
        """Test semantic search."""
        from tsugite.memory.manager import MemoryManager

        manager = MemoryManager(db_path=temp_db)
        manager.store("The cat sat on the mat")
        manager.store("Dogs are loyal pets")
        manager.store("Programming is fun")

        results = manager.search("feline animal")
        assert len(results) <= 5
        manager.close()

    def test_search_with_filters(self, temp_db, mock_embedding):
        """Test search with filters."""
        from tsugite.memory.manager import MemoryManager

        manager = MemoryManager(db_path=temp_db)
        manager.store("Fact 1", memory_type="fact", agent_name="agent1", tags=["tag1"])
        manager.store("Note 1", memory_type="note", agent_name="agent2", tags=["tag2"])
        manager.store("Fact 2", memory_type="fact", agent_name="agent1", tags=["tag1", "tag3"])

        results = manager.search("query", agent_name="agent1")
        assert all(m.agent_name == "agent1" for m in results)

        results = manager.search("query", memory_type="fact")
        assert all(m.memory_type == "fact" for m in results)

        results = manager.search("query", tags=["tag1"])
        assert len(results) >= 1
        manager.close()

    def test_store_with_metadata(self, temp_db, mock_embedding):
        """Test storing memory with metadata."""
        from tsugite.memory.manager import MemoryManager

        manager = MemoryManager(db_path=temp_db)
        metadata = {"source": "user_stated", "confidence": "high"}
        memory_id = manager.store("User's cat is Luna", memory_type="fact", metadata=metadata)

        memory = manager.get(memory_id)
        assert memory is not None
        assert memory.metadata == metadata
        manager.close()

    def test_get_memory_includes_metadata(self, temp_db, mock_embedding):
        """Test that get() returns metadata."""
        from tsugite.memory.manager import MemoryManager

        manager = MemoryManager(db_path=temp_db)
        memory_id = manager.store("Vet appointment", memory_type="event", metadata={"event_date": "2024-12-15"})

        memory = manager.get(memory_id)
        assert memory.metadata == {"event_date": "2024-12-15"}
        manager.close()

    def test_search_returns_metadata(self, temp_db, mock_embedding):
        """Test that search results include metadata."""
        from tsugite.memory.manager import MemoryManager

        manager = MemoryManager(db_path=temp_db)
        manager.store("Fact with source", memory_type="fact", metadata={"source": "inferred"})

        results = manager.search("fact")
        assert len(results) >= 1
        assert results[0].metadata == {"source": "inferred"}
        manager.close()

    def test_list_recent_returns_metadata(self, temp_db, mock_embedding):
        """Test that list_recent results include metadata."""
        from tsugite.memory.manager import MemoryManager

        manager = MemoryManager(db_path=temp_db)
        manager.store("Note with meta", metadata={"key": "value"})

        results = manager.list_recent()
        assert len(results) == 1
        assert results[0].metadata == {"key": "value"}
        manager.close()


@requires_duckdb
class TestMemorySingleton:
    """Tests for singleton pattern."""

    def test_reset_memory_manager(self):
        """Test resetting the singleton."""
        from tsugite.memory.manager import reset_memory_manager

        reset_memory_manager()


class TestMemoryConfig:
    """Tests for memory configuration."""

    def test_config_has_memory_fields(self):
        """Test that Config has memory fields."""
        from tsugite.config import Config

        config = Config()
        assert hasattr(config, "memory_enabled")
        assert hasattr(config, "memory_db_path")
        assert hasattr(config, "memory_embedding_model")
        assert hasattr(config, "memory_embedding_dimension")

    def test_config_memory_defaults(self):
        """Test Config memory default values."""
        from tsugite.config import Config

        config = Config()
        assert config.memory_enabled is False
        assert config.memory_db_path is None
        assert config.memory_embedding_model == "BAAI/bge-small-en-v1.5"
        assert config.memory_embedding_dimension == 384


class TestAgentConfigMemory:
    """Tests for memory_enabled in AgentConfig."""

    def test_agent_config_memory_enabled_field(self):
        """Test that AgentConfig has memory_enabled field."""
        from tsugite.md_agents import AgentConfig

        config = AgentConfig(name="test")
        assert hasattr(config, "memory_enabled")
        assert config.memory_enabled is None

    def test_agent_config_memory_enabled_true(self):
        """Test setting memory_enabled to True."""
        from tsugite.md_agents import AgentConfig

        config = AgentConfig(name="test", memory_enabled=True)
        assert config.memory_enabled is True

    def test_agent_config_memory_enabled_false(self):
        """Test setting memory_enabled to False."""
        from tsugite.md_agents import AgentConfig

        config = AgentConfig(name="test", memory_enabled=False)
        assert config.memory_enabled is False


class TestExecutionOptionsMemory:
    """Tests for memory_enabled in ExecutionOptions."""

    def test_execution_options_memory_field(self):
        """Test that ExecutionOptions has memory_enabled field."""
        from tsugite.options import ExecutionOptions

        opts = ExecutionOptions()
        assert hasattr(opts, "memory_enabled")
        assert opts.memory_enabled is None

    def test_execution_options_memory_enabled(self):
        """Test setting memory_enabled."""
        from tsugite.options import ExecutionOptions

        opts = ExecutionOptions(memory_enabled=True)
        assert opts.memory_enabled is True

        opts = ExecutionOptions(memory_enabled=False)
        assert opts.memory_enabled is False
