"""Data models for usage tracking."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class UsageRecord:
    """A single usage record for an agent run.

    Cost is stored per-record because pricing varies by model and may change
    over time. Each record captures the cost at time of execution.
    """

    timestamp: str  # ISO datetime
    agent: str
    model: str
    session_id: Optional[str] = None
    schedule_id: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0
    duration_ms: Optional[int] = None
    id: Optional[int] = None  # auto-increment, set by DB
