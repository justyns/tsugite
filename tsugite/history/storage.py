"""Session storage V2 - JSONL-based conversation history with context tracking.

This module provides unified session storage used by CLI, workspace, and daemon modes.
Key features:
- Context stored once with delta updates on changes
- Full messages per turn for exact reconstruction
- Content-addressable storage for attachments
"""

import hashlib
import json
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import portalocker

from tsugite.attachments.base import Attachment, AttachmentContentType
from tsugite.cache import compute_context_hash, store_content
from tsugite.config import get_xdg_data_path, load_config

from .models import (
    AttachmentRef,
    CompactionSummary,
    ContextSnapshot,
    ContextUpdate,
    SessionMeta,
    SessionRecord,
    Turn,
)


def get_history_dir() -> Path:
    """Get path to conversation history directory.

    Returns:
        Path to history directory in XDG data location
        (~/.local/share/tsugite/history/)
    """
    return get_xdg_data_path("history")


def generate_session_id(agent_name: str, timestamp: Optional[datetime] = None) -> str:
    """Generate unique session ID.

    Format: YYYYMMDD_HHMMSS_{agent}_{hash}
    Example: 20251024_103000_chat_abc123

    Args:
        agent_name: Name of the agent
        timestamp: Optional timestamp (defaults to now)

    Returns:
        Unique session ID
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    date_str = timestamp.strftime("%Y%m%d_%H%M%S")
    clean_agent = "".join(c if c.isalnum() or c == "-" else "_" for c in agent_name)
    clean_agent = clean_agent[:20]
    hash_input = f"{timestamp.isoformat()}_{agent_name}".encode()
    hash_str = hashlib.sha256(hash_input).hexdigest()[:6]

    return f"{date_str}_{clean_agent}_{hash_str}"


def get_machine_name() -> str:
    """Get machine name for session tracking.

    Checks config for custom machine_name, falls back to hostname.

    Returns:
        Machine name string
    """
    config = load_config()
    if hasattr(config, "machine_name") and config.machine_name:
        return config.machine_name
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


class SessionStorage:
    """Unified session storage for CLI, workspace, and daemon modes.

    Tracks context state, records turns, and handles context updates.
    Uses JSONL format with content-addressable storage for attachments.
    """

    def __init__(self, session_path: Path):
        """Initialize session storage.

        Args:
            session_path: Path to the JSONL session file
        """
        self.session_path = session_path
        self.session_id = session_path.stem

        # Context tracking state
        self._current_context_hash: Optional[str] = None
        self._current_attachments: Dict[str, AttachmentRef] = {}
        self._current_skills: List[str] = []
        self._turn_count: int = 0
        self._total_tokens: int = 0
        self._total_cost: float = 0.0
        self._meta: Optional[SessionMeta] = None

    @classmethod
    def create(
        cls,
        agent_name: str,
        model: str,
        workspace: Optional[str] = None,
        compacted_from: Optional[str] = None,
        session_path: Optional[Path] = None,
        timestamp: Optional[datetime] = None,
    ) -> "SessionStorage":
        """Create a new session.

        Args:
            agent_name: Name of the agent
            model: Model identifier
            workspace: Optional workspace name
            compacted_from: Optional previous session ID if compacted
            session_path: Optional explicit path (defaults to history dir)
            timestamp: Optional timestamp for session creation (defaults to now)

        Returns:
            New SessionStorage instance
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        session_id = generate_session_id(agent_name, timestamp)

        if session_path is None:
            session_path = get_history_dir() / f"{session_id}.jsonl"

        session_path.parent.mkdir(parents=True, exist_ok=True)

        storage = cls(session_path)
        storage._meta = SessionMeta(
            workspace=workspace,
            agent=agent_name,
            model=model,
            machine=get_machine_name(),
            created_at=timestamp,
            compacted_from=compacted_from,
        )

        storage._write_record(storage._meta)
        return storage

    @classmethod
    def load(cls, session_path: Path) -> "SessionStorage":
        """Load existing session and replay to get current state.

        Args:
            session_path: Path to session JSONL file

        Returns:
            SessionStorage with replayed state

        Raises:
            FileNotFoundError: If session doesn't exist
        """
        if not session_path.exists():
            raise FileNotFoundError(f"Session not found: {session_path}")

        storage = cls(session_path)
        storage._replay_state()
        return storage

    @classmethod
    def get_or_create(
        cls,
        session_id: str,
        agent_name: str,
        model: str,
        workspace: Optional[str] = None,
        session_path: Optional[Path] = None,
    ) -> "SessionStorage":
        """Get existing session or create new one.

        Args:
            session_id: Session ID
            agent_name: Agent name (used if creating)
            model: Model identifier (used if creating)
            workspace: Optional workspace name
            session_path: Optional explicit path

        Returns:
            SessionStorage instance
        """
        if session_path is None:
            session_path = get_history_dir() / f"{session_id}.jsonl"

        if session_path.exists():
            return cls.load(session_path)
        else:
            return cls.create(agent_name, model, workspace, session_path=session_path)

    def _write_record(self, record: SessionRecord) -> None:
        """Append a record to the session file with locking."""
        self.session_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.session_path, "a", encoding="utf-8") as f:
                portalocker.lock(f, portalocker.LOCK_EX)
                try:
                    f.write(record.model_dump_json(exclude_none=True))
                    f.write("\n")
                    f.flush()
                finally:
                    portalocker.unlock(f)
        except portalocker.exceptions.LockException:
            raise RuntimeError(f"Failed to acquire lock on {self.session_path}")
        except IOError as e:
            raise RuntimeError(f"Failed to write to {self.session_path}: {e}")

    def _replay_state(self) -> None:
        """Replay records to rebuild current state."""
        for record in self.load_records():
            if isinstance(record, SessionMeta):
                self._meta = record
            elif isinstance(record, ContextSnapshot):
                self._current_context_hash = record.hash
                self._current_attachments = record.attachments
                self._current_skills = list(record.skills)
            elif isinstance(record, ContextUpdate):
                self._current_context_hash = record.hash
                for name, ref in record.changed.items():
                    self._current_attachments[name] = ref
                for name in record.removed:
                    self._current_attachments.pop(name, None)
                for skill in record.added_skills:
                    if skill not in self._current_skills:
                        self._current_skills.append(skill)
                for skill in record.removed_skills:
                    if skill in self._current_skills:
                        self._current_skills.remove(skill)
            elif isinstance(record, Turn):
                self._turn_count += 1
                self._total_tokens += record.tokens or 0
                self._total_cost += record.cost or 0.0

    def load_records(self) -> List[SessionRecord]:
        """Load all records from session file.

        Returns:
            List of session records
        """
        if not self.session_path.exists():
            return []

        records = []
        with open(self.session_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    record = self._parse_record(data)
                    if record:
                        records.append(record)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON at line {line_num}: {e}")

        return records

    def _parse_record(self, data: Dict[str, Any]) -> Optional[SessionRecord]:
        """Parse a record dict into the appropriate model.

        Handles old format records gracefully by skipping them with a warning.
        Old format Turn records have 'user', 'assistant', 'tools', 'steps' fields
        instead of 'messages', 'final_answer', 'functions_called'.
        """
        from pydantic import ValidationError

        record_type = data.get("type")

        try:
            if record_type == "session_meta":
                return SessionMeta.model_validate(data)
            elif record_type == "context":
                return ContextSnapshot.model_validate(data)
            elif record_type == "context_update":
                return ContextUpdate.model_validate(data)
            elif record_type == "turn":
                return Turn.model_validate(data)
            elif record_type == "compaction_summary":
                return CompactionSummary.model_validate(data)
            # Handle old format "metadata" type (V1 format)
            elif record_type == "metadata":
                print("Warning: Skipping old V1 metadata record (incompatible format)")
                return None
            else:
                return None
        except ValidationError as e:
            # Old format records will fail validation - skip them
            print(f"Warning: Skipping incompatible record (old format?): {e.error_count()} validation errors")
            return None

    def record_initial_context(
        self,
        attachments: Optional[List[Attachment]] = None,
        skills: Optional[List[str]] = None,
    ) -> None:
        """Record initial context snapshot.

        Args:
            attachments: List of Attachment objects
            skills: List of skill names
        """
        attachments = attachments or []
        skills = skills or []

        att_refs = self._build_attachment_refs(attachments)
        context_hash = compute_context_hash(
            {name: ref.model_dump(exclude_none=True) for name, ref in att_refs.items()}, skills
        )

        snapshot = ContextSnapshot(attachments=att_refs, skills=skills, hash=context_hash)
        self._write_record(snapshot)

        self._current_context_hash = context_hash
        self._current_attachments = att_refs
        self._current_skills = list(skills)

    def _build_attachment_refs(self, attachments: List[Attachment]) -> Dict[str, AttachmentRef]:
        """Build AttachmentRef dict from Attachment list."""
        refs = {}
        for att in attachments:
            if att.source_url:
                refs[att.name] = AttachmentRef(
                    url=att.source_url,
                    type=att.content_type.value,
                    source="url",
                    mime_type=att.mime_type,
                )
            else:
                is_binary = att.content_type in (
                    AttachmentContentType.IMAGE,
                    AttachmentContentType.AUDIO,
                    AttachmentContentType.DOCUMENT,
                )
                content_hash = store_content(att.content, is_binary=is_binary)
                refs[att.name] = AttachmentRef(
                    hash=content_hash,
                    type=att.content_type.value,
                    source="file",
                    mime_type=att.mime_type,
                )
        return refs

    def check_and_record_context_change(
        self,
        attachments: Optional[List[Attachment]] = None,
        skills: Optional[List[str]] = None,
    ) -> bool:
        """Check for context changes and record update if changed.

        Args:
            attachments: Current attachments
            skills: Current skills

        Returns:
            True if context changed, False otherwise
        """
        attachments = attachments or []
        skills = skills or []

        new_refs = self._build_attachment_refs(attachments)
        new_hash = compute_context_hash(
            {name: ref.model_dump(exclude_none=True) for name, ref in new_refs.items()}, skills
        )

        if new_hash == self._current_context_hash:
            return False

        # Compute delta
        changed = {}
        for name, ref in new_refs.items():
            if name not in self._current_attachments or self._current_attachments[name] != ref:
                changed[name] = ref

        removed = [name for name in self._current_attachments if name not in new_refs]
        added_skills = [s for s in skills if s not in self._current_skills]
        removed_skills = [s for s in self._current_skills if s not in skills]

        update = ContextUpdate(
            changed=changed,
            removed=removed,
            added_skills=added_skills,
            removed_skills=removed_skills,
            timestamp=datetime.now(timezone.utc),
            hash=new_hash,
        )
        self._write_record(update)

        self._current_context_hash = new_hash
        self._current_attachments = new_refs
        self._current_skills = list(skills)
        return True

    def record_turn(
        self,
        messages: List[Dict[str, Any]],
        final_answer: Optional[str] = None,
        tokens: Optional[int] = None,
        cost: Optional[float] = None,
        model: Optional[str] = None,
        duration_ms: Optional[int] = None,
        functions_called: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a conversation turn.

        Args:
            messages: Full LiteLLM message array for this turn
            final_answer: Optional final answer text
            tokens: Token count for this turn
            cost: Cost for this turn
            model: Model used (may differ from session default)
            duration_ms: Execution duration
            functions_called: List of function/tool names called
            metadata: Channel routing metadata
        """
        user_summary = None
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_summary = content[:100] + "..." if len(content) > 100 else content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            user_summary = text[:100] + "..." if len(text) > 100 else text
                            break
                break

        turn = Turn(
            messages=messages,
            final_answer=final_answer,
            user_summary=user_summary,
            tokens=tokens,
            cost=cost,
            timestamp=datetime.now(timezone.utc),
            model=model,
            duration_ms=duration_ms,
            functions_called=functions_called or [],
            metadata=metadata,
        )
        self._write_record(turn)

        self._turn_count += 1
        self._total_tokens += tokens or 0
        self._total_cost += cost or 0.0

    def record_compaction_summary(self, summary: str, previous_turns: int) -> None:
        """Record compaction summary.

        Args:
            summary: LLM-generated summary of previous conversation
            previous_turns: Number of turns in compacted session
        """
        record = CompactionSummary(summary=summary, previous_turns=previous_turns)
        self._write_record(record)

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def agent(self) -> Optional[str]:
        return self._meta.agent if self._meta else None

    @property
    def model(self) -> Optional[str]:
        return self._meta.model if self._meta else None

    @property
    def machine(self) -> Optional[str]:
        return self._meta.machine if self._meta else None

    @property
    def created_at(self) -> Optional[datetime]:
        return self._meta.created_at if self._meta else None


def list_session_files() -> List[Path]:
    """List all session JSONL files.

    Returns:
        List of session file paths (sorted by modification time, newest first)
    """
    history_dir = get_history_dir()

    if not history_dir.exists():
        return []

    try:
        files = list(history_dir.glob("*.jsonl"))
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files
    except OSError:
        return []
