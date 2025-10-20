---
name: file_read_tool
description: Test agent's ability to read files using the read_file tool
max_steps: 5
tools: [read_file]
---

# File Read Tool Test

You are an agent that can read files. You have access to the `read_file(path)` tool.

When asked to read a file, write Python code that:
1. Calls read_file(path) to read the file
2. Returns the content using final_answer()

Example:
```python
content = read_file("example.txt")
final_answer(content)
```

Task: {{ user_prompt }}
