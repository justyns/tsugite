"""Memory data structures."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Memory:
    """A stored memory with semantic embedding."""

    id: int
    content: str
    memory_type: str = "note"  # fact, event, instruction, note
    agent_name: Optional[str] = None  # Namespace (None = global)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Type-specific data (source, event_date, etc.)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    score: Optional[float] = None  # Populated during search
