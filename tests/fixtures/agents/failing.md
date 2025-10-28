---
name: failing_test_agent
model: ollama:qwen2.5-coder:7b
max_turns: 2
tools: [run, final_answer]
---

Task: {{ user_prompt }}

This agent is designed to fail for error handling tests.

When the prompt contains "fail", deliberately use an invalid tool or raise an error.
Otherwise complete the task normally.
