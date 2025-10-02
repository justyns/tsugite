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


def is_builtin_agent(name: str) -> bool:
    """Check if an agent name refers to a built-in agent.

    Args:
        name: Agent name to check

    Returns:
        True if the name refers to a built-in agent
    """
    return name == "builtin-default"
