---
name: file_write_tool
description: Test agent's ability to write files using the write_file tool
max_turns: 5
tools: [write_file, read_file]
---

# File Write Tool Test

You are an agent that can write files. You have access to the `write_file(path, content)` tool.

Task: {{ user_prompt }}
