---
name: slow_test_agent
model: ollama:qwen2.5-coder:7b
max_turns: 3
tools: [run, final_answer]
---

Task: {{ user_prompt }}

This agent intentionally sleeps to test timeout handling.

Use the run tool to execute a sleep command, then return a result.
