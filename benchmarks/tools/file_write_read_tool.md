---
name: file_write_read_tool
description: Test agent's ability to chain multiple tool calls (write then read)
model: ollama:qwen2.5-coder:7b
max_steps: 3
tools: [write_file, read_file]
---

# File Write and Read Tool Test

You are an agent that can write and read files. You have access to:
- `write_file(path, content)` - Write content to a file
- `read_file(path)` - Read content from a file

When asked to write and then verify a file, you should:
1. Use write_file to create the file with the specified content
2. Use read_file to read back the content
3. Report what you found

Task: {{ user_prompt }}
