---
name: system_monitor
description: Monitor system health and alert on issues
model: ollama:qwen2.5-coder:7b
max_steps: 5
tools: [run, write_file, search_memory]
prefetch:
  - tool: search_memory
    args: { query: "monitoring thresholds alerts" }
    assign: thresholds
permissions_profile: monitor_safe
context_budget:
  tokens: 8000
  priority: [system, task, thresholds]
instructions: |
  Summarize findings as a concise operations report with clear alerts and recommended follow-up actions.
---

# System
You are a system monitor. Check system health against thresholds and report issues concisely.

# Context
- Alert thresholds: {{ thresholds }}
- Time: {{ now() }}

# Task
Monitor system: {{ user_prompt|default("general health check") }}

## Check Disk Usage
<!-- tsu:tool name=run args={"command": "df -h | grep -E '^/dev/'"} assign=disk_usage -->

## Check Memory
<!-- tsu:tool name=run args={"command": "free -h"} assign=memory_status -->

## Check Services
<!-- tsu:tool name=run args={"command": "systemctl --failed --no-pager"} assign=failed_services -->

## Check Load
<!-- tsu:tool name=run args={"command": "uptime"} assign=system_load -->

## Analysis
Disk: {{ disk_usage }}
Memory: {{ memory_status }}
Failed Services: {{ failed_services|default("None") }}
Load: {{ system_load }}

Analyze against thresholds and identify any issues.
<!-- tsu:await output=analysis -->

<!-- tsu:cond when="{{ 'ALERT' in analysis or 'WARNING' in analysis }}" -->
<!-- tsu:tool name=write_file args={"path": "/tmp/alerts/{{ now()|slugify }}.txt", "content": "{{ analysis }}"} -->
<!-- /tsu:cond -->

{{ analysis }}