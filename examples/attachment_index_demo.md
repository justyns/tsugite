---
name: attachment_index_demo
description: "Demonstrates `mode: index` for two-tier memory loading."
model: openai:gpt-4o-mini
tools: [read_file]
attachments:
  - path: "memory/topics/*.md"
    mode: index
    name: topic_index
    assign: topics
    index_format: first_heading
---

# Topic-aware agent

Task: {{ user_prompt }}

You have a topic index loaded as the `topic_index` attachment with {{ topics | length }} entries. Read individual topic files with `read_file(path=...)` only when relevant to the task; the index alone is enough to know what is available.

{% if topics %}
## Topic summary

{% for topic in topics %}
- **{{ topic.heading or topic.path }}** ({{ topic.path }})
{% endfor %}
{% endif %}
