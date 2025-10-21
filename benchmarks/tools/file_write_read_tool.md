---
name: file_write_read_tool
description: Test agent's ability to chain multiple tool calls (write then read)
max_steps: 6
tools: [write_file, read_file]
---

# File Write and Read Tool Test

You are an agent that can write and read files. You have access to:
- `write_file(path, content)` - Write content to a file
- `read_file(path)` - Read content from a file

Task: {{ user_prompt }}
