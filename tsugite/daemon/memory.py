"""Memory management helpers for daemon."""

from typing import List


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
