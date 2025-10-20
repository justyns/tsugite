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

When asked to write and then verify a file, write Python code that:
1. Calls write_file() to create the file
2. Calls read_file() to read it back
3. Returns the content using final_answer()

Example:
```python
write_file("test.txt", "Hello, World!")
content = read_file("test.txt")
final_answer(content)
```

Task: {{ user_prompt }}
