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

    @classmethod
    def from_cli(
        cls,
        plain: bool = False,
        headless: bool = False,
        no_color: bool = False,
        final_only: bool = False,
        verbose: bool = False,
        show_reasoning: bool = True,
        non_interactive: bool = False,
        log_json: bool = False,
    ) -> "UIOptions":
        return cls(
            plain=plain,
            headless=headless,
            no_color=no_color,
            final_only=final_only,
            verbose=verbose,
            show_reasoning=show_reasoning,
            non_interactive=non_interactive,
            log_json=log_json,
        )


@dataclass
class ExecutionOptions:
    """Controls how the agent executes."""

    model_override: Optional[str] = None
    max_turns_override: Optional[int] = None
    debug: bool = False
    stream: bool = False
    trust_mcp_code: bool = False
    dry_run: bool = False
    return_token_usage: bool = False
    sandbox: bool = False
    allow_domains: List[str] = field(default_factory=list)
    no_network: bool = False

    @classmethod
    def from_cli(
        cls,
        model: Optional[str] = None,
        debug: bool = False,
        stream: bool = False,
        trust_mcp_code: bool = False,
        dry_run: bool = False,
        sandbox: bool = False,
        no_sandbox: bool = False,
        allow_domain: Optional[List[str]] = None,
        no_network: bool = False,
    ) -> "ExecutionOptions":
        return cls(
            model_override=model,
            debug=debug,
            stream=stream,
            trust_mcp_code=trust_mcp_code,
            dry_run=dry_run,
            sandbox=sandbox and not no_sandbox,
            allow_domains=list(allow_domain) if allow_domain else [],
            no_network=no_network,
        )


@dataclass
class HistoryOptions:
    """Controls conversation history."""

    enabled: bool = True
    continue_id: Optional[str] = None
    storage_dir: Optional[Path] = None
    max_turns: int = 50

    @classmethod
    def from_cli(
        cls,
        no_history: bool = False,
        continue_conversation: bool = False,
        conversation_id: Optional[str] = None,
        history_dir: Optional[str] = None,
        max_turns: int = 50,
    ) -> "HistoryOptions":
        return cls(
            enabled=not no_history,
            continue_id=conversation_id if continue_conversation else None,
            storage_dir=Path(history_dir) if history_dir else None,
            max_turns=max_turns,
        )


@dataclass
class AttachmentOptions:
    """Controls file attachments."""

    sources: List[str] = field(default_factory=list)
    refresh_cache: bool = False
    auto_context: Optional[bool] = None

    @classmethod
    def from_cli(
        cls,
        sources: Optional[List[str]] = None,
        refresh_cache: bool = False,
        auto_context: Optional[bool] = None,
    ) -> "AttachmentOptions":
        return cls(
            sources=list(sources) if sources else [],
            refresh_cache=refresh_cache,
            auto_context=auto_context,
        )
