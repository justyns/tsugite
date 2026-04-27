---
name: attachment_assign_demo
description: Demonstrates binding attachment content to a Jinja variable via `assign:`.
model: openai:gpt-4o-mini
tools: [read_file]
attachments:
  - path: MEMORY.md
    assign: memory_content
  - path: USER.md
    assign: user_prefs
    attach: false
instructions: |
  {% if user_prefs %}
  User preferences (from USER.md, not directly attached): {{ user_prefs }}
  {% endif %}
---

# Memory-aware agent

Task: {{ user_prompt }}

{% if memory_content %}
## What memory says

{{ memory_content }}

{% if "Vikunja" in memory_content %}
**Heads up:** memory mentions Vikunja. Remember to check the vikunja skill before answering task questions.
{% endif %}
{% else %}
No memory available for this workspace.
{% endif %}

When answering, cite the relevant memory section.
