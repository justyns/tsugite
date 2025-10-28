---
name: simple_test_agent
model: ollama:qwen2.5-coder:7b
max_turns: 2
tools: [final_answer]
---

Task: {{ user_prompt }}

This is a simple test agent that returns a result immediately.

Complete the task and call final_answer() with your result.
