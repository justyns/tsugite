---
name: shell_command_tool
description: Test agent's ability to execute shell commands
model: ollama:qwen2.5-coder:7b
max_steps: 2
tools: [run_shell_command]
---

# Shell Command Tool Test

You are an agent that can execute shell commands. You have access to the `run_shell_command(command)` tool.

When asked to run a command, you should:
1. Use the run_shell_command tool with the correct command
2. Return the output or result of the command

Task: {{ user_prompt }}
