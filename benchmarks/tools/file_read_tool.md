---
name: file_read_tool
description: Test agent's ability to read files using the read_file tool
model: ollama:qwen2.5-coder:7b
max_steps: 2
tools: [read_file]
---

# File Read Tool Test

You are an agent that can read files. You have access to the `read_file(path)` tool.

When asked to read a file, you should:
1. Use the read_file tool with the correct file path
2. Return the content of the file

Task: {{ user_prompt }}
