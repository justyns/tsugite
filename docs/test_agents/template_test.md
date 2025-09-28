---
name: template_test
model: openai:gpt-4o-mini
tools: [read_file, get_system_info]
---

# Context

- Current time: {{ now() }}
- Today's date: {{ today() }}
- Slugified text: {{ "Hello World!" | slugify }}
- Environment: {{ env.HOME }}

{{ user_prompt }}
