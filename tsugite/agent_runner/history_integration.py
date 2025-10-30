"""History integration for agent runs."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def load_conversation_context(conversation_id: str) -> list:
    """Load conversation history as chat context.

    Args:
        conversation_id: Conversation ID to load

    Returns:
        List of ChatTurn-like dicts for context injection

    Raises:
        FileNotFoundError: If conversation doesn't exist
        RuntimeError: If load fails
    """
    from tsugite.ui.chat_history import load_conversation_history

    turns = load_conversation_history(conversation_id)

    # Convert Turn objects to ChatTurn-like format for context
    chat_turns = []
    for turn in turns:
        from dataclasses import dataclass

        @dataclass
        class ChatTurn:
            timestamp: datetime
            user_message: str
            agent_response: str
            tool_calls: list = None
            token_count: Optional[int] = None
            cost: Optional[float] = None
            steps: Optional[list] = None
            messages: Optional[list] = None

        chat_turn = ChatTurn(
            timestamp=turn.timestamp,
            user_message=turn.user,
            agent_response=turn.assistant,
            tool_calls=turn.tools or [],
            token_count=turn.tokens,
            cost=turn.cost,
            steps=turn.steps,  # Include execution steps if available
            messages=turn.messages,  # Include message history if available
        )
        chat_turns.append(chat_turn)

    return chat_turns


def save_run_to_history(
    agent_path: Path,
    agent_name: str,
    prompt: str,
    result: str,
    model: str,
    token_count: Optional[int] = None,
    cost: Optional[float] = None,
    execution_steps: Optional[list] = None,
    continue_conversation_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
    attachments: Optional[list] = None,
) -> Optional[str]:
    """Save a single agent run to history.

    Args:
        agent_path: Path to agent file
        agent_name: Name of the agent
        prompt: User prompt/task
        result: Agent's final answer
        model: Model used
        token_count: Number of tokens used
        cost: Cost of execution
        execution_steps: List of execution steps (from agent memory)
        continue_conversation_id: Optional conversation ID to continue (for multi-turn run mode)
        system_prompt: System prompt sent to LLM
        attachments: List of (name, content) tuples for attachments

    Returns:
        Conversation ID if saved, None if history disabled or failed
    """
    # Don't save subagent runs to history
    import os

    if os.environ.get("TSUGITE_SUBAGENT_MODE") == "1":
        return None

    try:
        from tsugite.config import load_config
        from tsugite.ui.chat_history import save_chat_turn, start_conversation

        # Check if history is enabled
        config = load_config()
        if not getattr(config, "history_enabled", True):
            return None

        # Check agent-level disable_history
        try:
            from tsugite.md_agents import parse_agent_file

            agent = parse_agent_file(agent_path)
            if getattr(agent.config, "disable_history", False):
                return None
        except Exception:
            # If we can't parse agent, assume history is enabled
            pass

        timestamp = datetime.now(timezone.utc)

        # Use existing conversation ID or start new one
        if continue_conversation_id:
            conv_id = continue_conversation_id
        else:
            conv_id = start_conversation(
                agent_name=agent_name,
                model=model,
                timestamp=timestamp,
            )

        # Build messages list with system prompt and attachments
        messages = []
        if system_prompt or attachments:
            # Build system message with attachments
            if attachments:
                # System blocks format (for providers that support prompt caching)
                system_blocks = [{"type": "text", "text": system_prompt or ""}]
                for name, content in attachments:
                    system_blocks.append(
                        {
                            "type": "text",
                            "text": f"<Attachment: {name}>\n{content}\n</Attachment: {name}>",
                            "cache_control": {"type": "ephemeral"},
                        }
                    )
                messages.append({"role": "system", "content": system_blocks})
            else:
                # Simple string format
                messages.append({"role": "system", "content": system_prompt})

            # User message
            messages.append({"role": "user", "content": prompt})

        # Save turn to conversation with full execution steps and messages
        save_chat_turn(
            conversation_id=conv_id,
            user_message=prompt,
            agent_response=result,
            tool_calls=_extract_tool_calls(execution_steps) if execution_steps else [],
            token_count=token_count,
            cost=cost,
            timestamp=timestamp,
            execution_steps=execution_steps,  # Pass full execution steps
            messages=messages if messages else None,  # Include system prompt and attachments
        )

        return conv_id

    except Exception as e:
        # Don't fail the run if history fails
        import sys

        print(f"Warning: Failed to save run to history: {e}", file=sys.stderr)
        return None


def _extract_tool_calls(execution_steps: list) -> list[str]:
    """Extract list of tool names called during execution.

    Args:
        execution_steps: List of execution step objects

    Returns:
        List of unique tool names called
    """
    tool_calls = set()
    for step in execution_steps:
        if hasattr(step, "tools_called") and step.tools_called:
            tool_calls.update(step.tools_called)
    return sorted(list(tool_calls))
