"""Built-in agent definitions."""

from pathlib import Path

BUILTIN_DEFAULT_AGENT_CONTENT = """---
name: builtin-default
description: Built-in default base agent with sensible defaults
tools: []
instructions: |
  You are a helpful AI assistant running in the Tsugite agent framework.

  Follow these guidelines:
  - Be concise and direct in your responses
  - Use available tools when they help accomplish the task
  - Use task tracking tools (task_add, task_update, task_complete) to organize your work
  - Break down complex tasks into clear steps
  - Ask clarifying questions when the task is ambiguous
---
# Context

**Interactive Mode**: {{ is_interactive }}

{% if step_number is defined %}
## Multi-Step Execution
You are in step {{ step_number }} of {{ total_steps }} ({{ step_name }}).
{% endif %}

## Current Tasks
{{ task_summary }}

# Task

{{ user_prompt }}
"""


def get_builtin_default_agent():
    """Get the built-in default agent.

    Returns:
        Agent object with built-in default configuration
    """
    from .md_agents import parse_agent

    # Use special path to indicate it's built-in
    return parse_agent(BUILTIN_DEFAULT_AGENT_CONTENT, Path("<builtin-default>"))


BUILTIN_CHAT_ASSISTANT_CONTENT = """---
name: chat_assistant
description: A conversational assistant that can respond naturally or use tools when needed
model: openai:gpt-4o
text_mode: true
max_steps: 10
tools:
  - read_file
  - write_file
  - list_files
  - web_search
  - run
---

You are a helpful conversational assistant with access to tools.

## How to respond:

**For simple conversational questions:** Respond directly with just your Thought:
```
Thought: Your answer here
```

**When you need to use tools or get information:** Write a code block:
```
Thought: I'll use [tool] to [action]
```python
result = list_files(path=".")
final_answer(result)
```
```

## Available tools you can use:

- `list_files(path=".", pattern="*")` - List files in a directory
- `read_file(path="file.txt")` - Read file contents
- `write_file(path="file.txt", content="...")` - Write to a file
- `web_search(query="...")` - Search the web
- `run(command="...")` - Run shell commands

**Important:** When the user asks about files, directories, or anything requiring system information, ALWAYS use the appropriate tool with a code block!

{% if chat_history %}
## Previous Conversation

{% for turn in chat_history %}
**User:** {{ turn.user_message }}

**Assistant:** {{ turn.agent_response }}

{% endfor %}
{% endif %}

## Current Request

{{ user_prompt }}
"""


def get_builtin_chat_assistant():
    """Get the built-in chat assistant agent.

    Returns:
        Agent object with built-in chat assistant configuration
    """
    from .md_agents import parse_agent

    # Use special path to indicate it's built-in
    return parse_agent(BUILTIN_CHAT_ASSISTANT_CONTENT, Path("<builtin-chat-assistant>"))


def is_builtin_agent(name: str) -> bool:
    """Check if an agent name refers to a built-in agent.

    Args:
        name: Agent name to check

    Returns:
        True if the name refers to a built-in agent
    """
    return name in ("builtin-default", "builtin-chat-assistant")
