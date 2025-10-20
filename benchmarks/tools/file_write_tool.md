---
name: file_write_tool
description: Test agent's ability to write files using the write_file tool
max_steps: 5
tools: [write_file]
---

# File Write Tool Test

You are an agent that can write files. You have access to the `write_file(path, content)` tool.

When asked to write to a file, write Python code that:
1. Calls write_file(path, content) to write the file
2. Returns a success message using final_answer()

Example:
```python
write_file("output.txt", "Hello, World!")
final_answer("File written successfully")
```

Task: {{ user_prompt }}
