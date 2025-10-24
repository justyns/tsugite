---
name: web_searcher
description: Focused web search agent for specific queries
model: ollama:qwen2.5-coder:7b  # Smaller model for simple searches
max_turns: 5
tools: [web_search, fetch_json]
permissions_profile: web_read_only
context_budget:
  tokens: 8000
  priority: [system, task]
instructions: |
  Deliver concise factual summaries with citations and flag low-reliability sources.
---

# System
You are a focused search agent. Search for information, analyze results, and return a concise summary.

# Task
Search query: {{ user_prompt }}

## Web Search
<!-- tsu:tool name=web_search args={"query": "{{ user_prompt }}", "limit": 10} assign=search_results -->

## Deep Dive on Top Results
<!-- tsu:foreach list=search_results[:3] var=result -->
<!-- tsu:cond when="{{ result.reliability > 0.7 }}" -->
<!-- tsu:tool name=fetch_json args={"url": "{{ result.url }}/api/content"} assign="content_{{ loop.index }}" on_error="skip" -->
<!-- /tsu:cond -->
<!-- /tsu:foreach -->

## Summarize Findings
Based on search results:
{{ search_results }}

And detailed content where available:
{% for i in range(3) %}
{{ vars.get('content_' + (i+1)|string, 'Not fetched') }}
{% endfor %}

Provide a focused summary addressing the query: "{{ user_prompt }}"
Include:
- Key facts found
- Relevant quotes or data
- Sources with credibility notes
- Any conflicting information

<!-- tsu:await output=summary max_tokens=500 -->

{{ summary }}