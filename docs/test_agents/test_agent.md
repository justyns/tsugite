---
name: test_agent
description: Simple test agent for verification
model: ollama:qwen2.5-coder:14b
max_turns: 3
tools: [read_file, write_file, get_system_info]
---

# System
You are a helpful test agent. Keep responses short and focused.

# Task
{{ user_prompt }}

Please help with this task. You can use the available tools to read files, write files, or get system information.