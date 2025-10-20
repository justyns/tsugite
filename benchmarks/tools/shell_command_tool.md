---
name: shell_command_tool
description: Test agent's ability to execute shell commands
max_steps: 5
tools: [run]
---

# Shell Command Tool Test

You are an agent that can execute shell commands. You have access to the `run(command)` tool.

When asked to run a command, write Python code that:
1. Calls run(command) to execute the shell command
2. Returns the output using final_answer()

Example:
```python
output = run("echo 'Hello, World!'")
final_answer(output)
```

Task: {{ user_prompt }}
