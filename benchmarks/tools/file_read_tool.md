---
name: file_read_tool
description: Test agent's ability to read files using the read_file tool
max_turns: 5
tools: [read_file]
---

# File Read Tool Test

You are an agent that can read files. You have access to the `read_file(path)` tool.

Task: {{ user_prompt }}
