---
name: nested_test_agent
model: ollama:qwen2.5-coder:7b
max_turns: 3
tools: [spawn_agent, final_answer]
---

Task: {{ user_prompt }}

This agent tests nested subagent spawning by delegating work to another agent.

Use spawn_agent() to delegate the task to tests/fixtures/agents/simple.md, then return that result.
