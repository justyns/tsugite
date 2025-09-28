---
name: prefetch_readme_test
model: openai:gpt-4o-mini
tools: [read_file]
prefetch:
  - tool: read_file
    args: { path: "README.md" }
    assign: readme_content
---
# Task

The README contains:
{{ readme_content[:100] }}...

Summarize what you learned from the readme.

