---
name: file_write_tool
description: Test agent's ability to write files using the write_file tool
model: ollama:qwen2.5-coder:7b
max_steps: 2
tools: [write_file]
---

# File Write Tool Test

You are an agent that can write files. You have access to the `write_file(path, content)` tool.

When asked to write to a file, you should:
1. Use the write_file tool with the correct file path and content
2. Report success after writing

Task: {{ user_prompt }}
