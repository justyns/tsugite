"""Memory management helpers for daemon."""

from pathlib import Path
from typing import List

from tsugite.config import get_daily_memory_path


def get_workspace_attachments(workspace_dir: Path, memory_enabled: bool, inject_days: int) -> List[str]:
    """Get workspace file paths for attachment resolution.

    Args:
        workspace_dir: Agent workspace directory
        memory_enabled: Whether memory injection is enabled
        inject_days: Number of recent daily memory files to include

    Returns:
        List of absolute file paths
    """
    from tsugite.workspace import Workspace
    from tsugite.workspace.context import build_workspace_attachments

    # Use unified workspace attachment builder
    workspace = Workspace.load(workspace_dir)
    attachment_objects = build_workspace_attachments(workspace, memory_days=inject_days if memory_enabled else 0)
    return [str(att.source) for att in attachment_objects]


async def summarize_session(conversation_history: List[dict], model: str = "openai:gpt-4o-mini") -> str:
    """Summarize conversation history using LLM.

    Args:
        conversation_history: List of {role, content} dicts
        model: Model to use for summarization

    Returns:
        Concise summary of conversation
    """
    from litellm import acompletion

    messages = [
        {
            "role": "system",
            "content": "Summarize this conversation concisely. Focus on: decisions made, important facts learned, user preferences, action items. Keep it under 500 words.",
        }
    ]

    convo_text = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in conversation_history])
    messages.append({"role": "user", "content": convo_text})

    response = await acompletion(model=model, messages=messages)
    return response.choices[0].message.content


async def extract_memories(conversation_history: List[dict], workspace_dir: Path, agent_path: Path) -> None:
    """Run memory extraction agent to review conversation and write memories.

    Args:
        conversation_history: Recent conversation turns
        workspace_dir: Agent workspace directory
        agent_path: Path to memory extraction agent
    """
    from tsugite.agent_runner import run_agent_async
    from tsugite.options import ExecutionOptions

    convo_text = "\n\n".join(
        [
            f"{msg['role'].upper()} ({msg.get('timestamp', 'unknown')}): {msg['content']}"
            for msg in conversation_history[-20:]  # Last 20 messages
        ]
    )

    daily_memory_path = get_daily_memory_path(workspace_dir)

    prompt = f"""Review this conversation and extract important information to add to the daily memory file.

Focus on:
- Important decisions or conclusions
- Facts about the user (preferences, context, situation)
- Action items or commitments
- Technical details that would be useful later

Conversation:
{convo_text}

Add extracted memories to: {daily_memory_path}

Format each memory as:
## HH:MM - Category
Brief description of the memory

Categories: Decision, Fact, Preference, Action Item, Technical Detail
"""

    await run_agent_async(
        agent_path=agent_path,
        prompt=prompt,
        exec_options=ExecutionOptions(return_token_usage=False),
    )
