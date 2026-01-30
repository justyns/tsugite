"""History integration for agent runs."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _process_messages_field(turn, messages: list) -> None:
    """Process turn with messages field (full message history)."""
    for msg in turn.messages:
        if msg.get("role") != "system":
            messages.append(msg)


def _process_steps_field(turn, messages: list) -> None:
    """Process turn with steps field (execution steps)."""
    messages.append({"role": "user", "content": turn.user})
    for step in turn.steps:
        code = step.get("code", "")
        xml_observation = step.get("xml_observation", "")

        # Assistant message is just code
        messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
        messages.append({"role": "user", "content": xml_observation})

    messages.append({"role": "assistant", "content": turn.assistant})


def _process_simple_turn(turn, messages: list) -> None:
    """Process simple turn (user/assistant only)."""
    messages.append({"role": "user", "content": turn.user})
    messages.append({"role": "assistant", "content": turn.assistant})


def load_conversation_messages(conversation_id: str) -> list[dict]:
    """Load conversation history as message list for LLM.

    Loads complete message history including tool calls and intermediate steps.
    System messages are skipped as they will be reconstructed with current context.

    Args:
        conversation_id: Conversation ID to load

    Returns:
        List of message dicts including all tool calls and observations

    Raises:
        FileNotFoundError: If conversation doesn't exist
        RuntimeError: If load fails
    """
    from tsugite.ui.chat_history import load_conversation_history

    turns = load_conversation_history(conversation_id)

    messages = []
    for turn in turns:
        if turn.messages:
            _process_messages_field(turn, messages)
        elif turn.steps:
            _process_steps_field(turn, messages)
        else:
            _process_simple_turn(turn, messages)

    return messages


def apply_cache_control_to_messages(messages: list[dict]) -> list[dict]:
    """Apply cache control markers to all conversation messages.

    Following industry best practices from Anthropic and OpenAI, we cache
    all conversation history.

    Args:
        messages: List of message dicts (user/assistant pairs)

    Returns:
        List of message dicts with cache_control added to all messages
    """
    if not messages:
        return messages

    return [{**msg, "cache_control": {"type": "ephemeral"}} for msg in messages]


def load_and_apply_history(conversation_id: str) -> list[dict]:
    """Load conversation history and apply cache control markers.

    Consolidates the common pattern of loading conversation messages
    and applying cache control for optimal performance.

    Args:
        conversation_id: Conversation ID to load

    Returns:
        List of message dicts with cache control applied

    Raises:
        ValueError: If conversation not found
        RuntimeError: If loading fails
    """
    try:
        messages = load_conversation_messages(conversation_id)
        if messages:
            messages = apply_cache_control_to_messages(messages)
        return messages
    except FileNotFoundError:
        raise ValueError(f"Conversation not found: {conversation_id}")
    except Exception as e:
        raise RuntimeError(f"Failed to load conversation history: {e}")


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
    channel_metadata: Optional[dict] = None,
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
        attachments: List of Attachment objects for attachments
        channel_metadata: Optional channel routing metadata (source, channel_id, user_id, reply_to)

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

        config = load_config()
        if not getattr(config, "history_enabled", True):
            return None

        try:
            from tsugite.md_agents import parse_agent_file

            agent = parse_agent_file(agent_path)
            if getattr(agent.config, "disable_history", False):
                return None
        except Exception as e:
            import sys

            print(f"Warning: Could not check agent history settings: {e}", file=sys.stderr)

        timestamp = datetime.now(timezone.utc)

        if continue_conversation_id:
            conv_id = continue_conversation_id
        else:
            conv_id = start_conversation(
                agent_name=agent_name,
                model=model,
                timestamp=timestamp,
            )

        messages = []

        if system_prompt or attachments:
            if attachments:
                from tsugite.attachments.base import AttachmentContentType

                system_blocks = [{"type": "text", "text": system_prompt or ""}]
                for attachment in attachments:
                    # For history, we only store text attachments fully
                    # Binary attachments are stored as references
                    if attachment.content_type == AttachmentContentType.TEXT:
                        system_blocks.append(
                            {
                                "type": "text",
                                "text": f"<Attachment: {attachment.name}>\n{attachment.content}\n</Attachment: {attachment.name}>",
                                "cache_control": {"type": "ephemeral"},
                            }
                        )
                    elif attachment.source_url:
                        # URL-based attachment (image/document)
                        system_blocks.append(
                            {
                                "type": "text",
                                "text": f"<Attachment: {attachment.name}>\n[{attachment.content_type.value}: {attachment.source_url}]\n</Attachment: {attachment.name}>",
                                "cache_control": {"type": "ephemeral"},
                            }
                        )
                    else:
                        # Base64 encoded binary - just store a placeholder
                        system_blocks.append(
                            {
                                "type": "text",
                                "text": f"<Attachment: {attachment.name}>\n[{attachment.content_type.value} file: {attachment.mime_type}]\n</Attachment: {attachment.name}>",
                                "cache_control": {"type": "ephemeral"},
                            }
                        )
                messages.append({"role": "system", "content": system_blocks})
            else:
                messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        if execution_steps:
            for step in execution_steps:
                code = getattr(step, "code", "")
                xml_observation = getattr(step, "xml_observation", "")

                # Assistant message is just code
                messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
                messages.append({"role": "user", "content": xml_observation})

        messages.append({"role": "assistant", "content": result})

        save_chat_turn(
            conversation_id=conv_id,
            user_message=prompt,
            agent_response=result,
            tool_calls=_extract_tool_calls(execution_steps) if execution_steps else [],
            token_count=token_count,
            cost=cost,
            timestamp=timestamp,
            execution_steps=execution_steps,
            messages=messages,
            metadata=channel_metadata,
        )

        return conv_id

    except Exception as e:
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
