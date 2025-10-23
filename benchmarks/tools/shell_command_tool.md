---
name: shell_command_tool
description: Test agent's ability to execute shell commands
max_turns: 5
tools: [run]
---

# Shell Command Tool Test

You are an agent that can execute shell commands. You have access to the `run(command)` tool.

Task: {{ user_prompt }}
