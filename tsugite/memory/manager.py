"""Memory manager with DuckDB backend and semantic search."""

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from tsugite.memory.schema import Memory

logger = logging.getLogger(__name__)


def parse_date_filter(value: str) -> datetime:
    """Parse date filter value - relative (7d, 1w, 30d) or ISO format.

    Args:
        value: Date string like "7d", "2w", "30d", or "2024-12-01"

    Returns:
        datetime object
    """
    # Match relative patterns: 7d, 2w, 1m
    match = re.match(r"^(\d+)([dwm])$", value.lower())
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        if unit == "d":
            return datetime.now() - timedelta(days=amount)
        elif unit == "w":
            return datetime.now() - timedelta(weeks=amount)
        elif unit == "m":
            return datetime.now() - timedelta(days=amount * 30)

    # Try ISO format
    return datetime.fromisoformat(value)


# Singleton instance
_memory_manager: Optional["MemoryManager"] = None


class MemoryManager:
    """Manages persistent memory with DuckDB and vector search."""

    def __init__(
        self,
        db_path: Path,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        embedding_dimension: int = 384,
    ):
        """Initialize memory manager.

        Args:
            db_path: Path to DuckDB database file
            embedding_model: HuggingFace model name for embeddings
            embedding_dimension: Dimension of embedding vectors
        """
        try:
            import duckdb
        except ImportError as e:
            raise ImportError("duckdb is required for memory system. Install with: pip install duckdb") from e

        self.db_path = db_path
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension

        # Ensure directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self.conn = duckdb.connect(str(db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        # Create memories table
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                content VARCHAR NOT NULL,
                memory_type VARCHAR DEFAULT 'note',
                agent_name VARCHAR,
                tags VARCHAR[],
                metadata JSON DEFAULT '{{}}',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                embedding FLOAT[{self.embedding_dimension}]
            )
        """)

        # Migration: add metadata column if missing (for existing DBs)
        try:
            self.conn.execute("ALTER TABLE memories ADD COLUMN metadata JSON DEFAULT '{}'")
        except Exception:
            pass  # Column already exists

        # Create sequence for auto-increment if not exists
        try:
            self.conn.execute("CREATE SEQUENCE IF NOT EXISTS memories_id_seq START 1")
        except Exception:
            pass  # Sequence might already exist

        # Try to create FTS index (may fail if already exists)
        try:
            self.conn.execute("INSTALL fts; LOAD fts;")
            self.conn.execute("""
                PRAGMA create_fts_index('memories', 'id', 'content',
                    stemmer='english', stopwords='english', overwrite=0)
            """)
        except Exception as e:
            # Index likely already exists
            logger.debug(f"FTS index setup: {e}")

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        from tsugite.memory.embeddings import get_embedding

        return get_embedding(text, self.embedding_model)

    def store(
        self,
        content: str,
        memory_type: str = "note",
        tags: Optional[List[str]] = None,
        agent_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Store a memory with auto-generated embedding.

        Args:
            content: The information to remember
            memory_type: Type of memory (fact, event, instruction, note)
            tags: List of tags for categorization
            agent_name: Agent namespace (None = global)
            metadata: Type-specific data (source, event_date, etc.)

        Returns:
            ID of the stored memory
        """
        embedding = self._get_embedding(content)
        tags = tags or []
        metadata = metadata or {}

        # Get next ID
        result = self.conn.execute("SELECT NEXTVAL('memories_id_seq')").fetchone()
        memory_id = result[0]

        self.conn.execute(
            """
            INSERT INTO memories (id, content, memory_type, agent_name, tags, metadata, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            [memory_id, content, memory_type, agent_name, tags, json.dumps(metadata), embedding],
        )

        logger.debug(f"Stored memory {memory_id}: {content[:50]}...")
        return memory_id

    def search(
        self,
        query: str,
        limit: int = 5,
        agent_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        memory_type: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> List[Memory]:
        """Search memories using semantic similarity.

        Args:
            query: Search query (natural language)
            limit: Maximum results to return
            agent_name: Filter by agent namespace (None = all)
            tags: Filter by tags (any match)
            memory_type: Filter by memory type
            since: Only memories created after this date (ISO or relative like "7d")
            until: Only memories created before this date (ISO or relative like "7d")

        Returns:
            List of matching memories with scores
        """
        query_embedding = self._get_embedding(query)

        # Build WHERE clause for filters
        where_clauses = []
        params = []

        if agent_name is not None:
            where_clauses.append("agent_name = ?")
            params.append(agent_name)

        if memory_type is not None:
            where_clauses.append("memory_type = ?")
            params.append(memory_type)

        if tags:
            # Match any tag
            tag_conditions = " OR ".join(["list_contains(tags, ?)" for _ in tags])
            where_clauses.append(f"({tag_conditions})")
            params.extend(tags)

        if since is not None:
            since_dt = parse_date_filter(since)
            where_clauses.append("created_at >= ?")
            params.append(since_dt)

        if until is not None:
            until_dt = parse_date_filter(until)
            where_clauses.append("created_at <= ?")
            params.append(until_dt)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        return self._semantic_search(query_embedding, limit, where_sql, params)

    def _semantic_search(
        self,
        query_embedding: List[float],
        limit: int,
        where_sql: str,
        params: List,
    ) -> List[Memory]:
        """Pure semantic search."""
        sql = f"""
            SELECT id, content, memory_type, agent_name, tags, metadata, created_at, updated_at,
                   array_cosine_similarity(embedding, ?::FLOAT[{self.embedding_dimension}]) as score
            FROM memories
            {where_sql}
            ORDER BY score DESC
            LIMIT ?
        """
        all_params = [query_embedding] + params + [limit]
        results = self.conn.execute(sql, all_params).fetchall()

        return [
            Memory(
                id=row[0],
                content=row[1],
                memory_type=row[2],
                agent_name=row[3],
                tags=row[4] or [],
                metadata=json.loads(row[5]) if row[5] else {},
                created_at=row[6],
                updated_at=row[7],
                score=row[8],
            )
            for row in results
        ]

    def list_recent(
        self,
        limit: int = 20,
        agent_name: Optional[str] = None,
        memory_type: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> List[Memory]:
        """List recent memories.

        Args:
            limit: Maximum results
            agent_name: Filter by agent namespace
            memory_type: Filter by type
            since: Only memories created after this date (ISO or relative like "7d")
            until: Only memories created before this date (ISO or relative like "7d")

        Returns:
            List of memories ordered by creation time
        """
        where_clauses = []
        params = []

        if agent_name is not None:
            where_clauses.append("agent_name = ?")
            params.append(agent_name)

        if memory_type is not None:
            where_clauses.append("memory_type = ?")
            params.append(memory_type)

        if since is not None:
            since_dt = parse_date_filter(since)
            where_clauses.append("created_at >= ?")
            params.append(since_dt)

        if until is not None:
            until_dt = parse_date_filter(until)
            where_clauses.append("created_at <= ?")
            params.append(until_dt)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        sql = f"""
            SELECT id, content, memory_type, agent_name, tags, metadata, created_at, updated_at
            FROM memories
            {where_sql}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)

        results = self.conn.execute(sql, params).fetchall()

        return [
            Memory(
                id=row[0],
                content=row[1],
                memory_type=row[2],
                agent_name=row[3],
                tags=row[4] or [],
                metadata=json.loads(row[5]) if row[5] else {},
                created_at=row[6],
                updated_at=row[7],
            )
            for row in results
        ]

    def get(self, memory_id: int) -> Optional[Memory]:
        """Get a memory by ID."""
        result = self.conn.execute(
            """
            SELECT id, content, memory_type, agent_name, tags, metadata, created_at, updated_at
            FROM memories WHERE id = ?
        """,
            [memory_id],
        ).fetchone()

        if result is None:
            return None

        return Memory(
            id=result[0],
            content=result[1],
            memory_type=result[2],
            agent_name=result[3],
            tags=result[4] or [],
            metadata=json.loads(result[5]) if result[5] else {},
            created_at=result[6],
            updated_at=result[7],
        )

    def update(self, memory_id: int, content: str) -> bool:
        """Update a memory's content (re-generates embedding).

        Args:
            memory_id: ID of memory to update
            content: New content

        Returns:
            True if updated, False if not found
        """
        existing = self.get(memory_id)
        if existing is None:
            return False

        embedding = self._get_embedding(content)

        self.conn.execute(
            """
            UPDATE memories
            SET content = ?, embedding = ?, updated_at = NOW()
            WHERE id = ?
        """,
            [content, embedding, memory_id],
        )

        logger.debug(f"Updated memory {memory_id}")
        return True

    def delete(self, memory_id: int) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: ID of memory to delete

        Returns:
            True if deleted, False if not found
        """
        result = self.conn.execute("DELETE FROM memories WHERE id = ? RETURNING id", [memory_id]).fetchone()

        if result:
            logger.debug(f"Deleted memory {memory_id}")
            return True
        return False

    def count(self, agent_name: Optional[str] = None) -> int:
        """Count total memories."""
        if agent_name is not None:
            result = self.conn.execute("SELECT COUNT(*) FROM memories WHERE agent_name = ?", [agent_name]).fetchone()
        else:
            result = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()

        return result[0] if result else 0

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None


def get_memory_manager() -> MemoryManager:
    """Get the singleton MemoryManager instance.

    Returns:
        MemoryManager instance

    Raises:
        RuntimeError: If memory is not enabled
    """
    global _memory_manager

    if _memory_manager is None:
        from tsugite.config import get_xdg_data_path, load_config

        config = load_config()

        # Get configuration
        db_path = config.memory_db_path if hasattr(config, "memory_db_path") and config.memory_db_path else None
        if db_path is None:
            db_path = get_xdg_data_path("memory") / "memories.duckdb"

        embedding_model = getattr(config, "memory_embedding_model", "BAAI/bge-small-en-v1.5")
        embedding_dimension = getattr(config, "memory_embedding_dimension", 384)

        _memory_manager = MemoryManager(
            db_path=Path(db_path),
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
        )

    return _memory_manager


def reset_memory_manager() -> None:
    """Reset the singleton (for testing or reconfiguration)."""
    global _memory_manager
    if _memory_manager is not None:
        _memory_manager.close()
        _memory_manager = None
