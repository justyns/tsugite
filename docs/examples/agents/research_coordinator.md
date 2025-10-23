---
name: research_coordinator
description: Orchestrate research on a topic using multiple search agents
model: ollama:qwen2.5-coder:14b
max_turns: 12
tools: [write_file, memory_write]
prefetch:
  - tool: search_memory
    args: { query: "research methodology" }
    assign: research_methods
permissions_profile: research_safe
context_budget:
  tokens: 20000
  priority: [system, task, research_methods, sub_results]
instructions: |
  Produce structured research with source attributions and call out confidence or gaps in the evidence.
---

# System
You are a research coordinator that breaks down topics into focused queries and coordinates sub-agents to gather comprehensive information.

# Context
- Research methods: {{ research_methods }}
- Timestamp: {{ now() }}

# Task
Research topic: {{ user_prompt }}

## Generate Research Queries
Based on the topic "{{ user_prompt }}", create 3-4 specific research queries that will help gather comprehensive information. Consider different angles:
- Current state/overview
- Recent developments
- Key challenges/problems
- Future trends/predictions

<!-- tsu:await output=queries format=json -->

## Deploy Search Agents
<!-- tsu:foreach list=queries var=query parallel=true -->
<!-- tsu:spawn agent="agents/web_searcher.md" prompt="{{ query }}" assign="search_{{ loop.index }}" timeout=60 -->
<!-- /tsu:foreach -->

## Compile Results
Analyze all search results:
{% for i in range(queries|length) %}
### Query {{ i+1 }}: {{ queries[i] }}
Results: {{ vars['search_' + (i+1)|string] }}
{% endfor %}

Now synthesize a comprehensive research summary.
<!-- tsu:await output=research_summary -->

## Save Research
<!-- tsu:tool name=write_file args={"path": "~/research/{{ user_prompt|slugify }}_{{ today() }}.md", "content": "{{ research_summary }}"} -->
<!-- tsu:tool name=memory_write args={"title": "Research: {{ user_prompt }}", "content": "{{ research_summary }}", "tags": ["research", "{{ today() }}"]"} -->

Research completed and saved!