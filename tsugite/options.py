"""Option dataclasses for bundling related parameters."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class UIOptions:
    """Controls output display and user interaction."""

    plain: bool = False
    headless: bool = False
    no_color: bool = False
    final_only: bool = False
    verbose: bool = False
    show_reasoning: bool = True
    non_interactive: bool = False
    log_json: bool = False


@dataclass
class ExecutionOptions:
    """Controls how the agent executes."""

    model_override: Optional[str] = None
    debug: bool = False
    stream: bool = False
    trust_mcp_code: bool = False
    dry_run: bool = False
    force_text_mode: bool = False
    return_token_usage: bool = False
    memory_enabled: Optional[bool] = None  # None = use agent/config default


@dataclass
class HistoryOptions:
    """Controls conversation history."""

    enabled: bool = True
    continue_id: Optional[str] = None
    storage_dir: Optional[Path] = None
    max_turns: int = 50


@dataclass
class AttachmentOptions:
    """Controls file attachments."""

    sources: List[str] = field(default_factory=list)
    refresh_cache: bool = False
    auto_context: Optional[bool] = None


@dataclass
class DockerOptions:
    """Controls Docker execution."""

    enabled: bool = False
    keep: bool = False
    container: Optional[str] = None
    network: str = "host"
