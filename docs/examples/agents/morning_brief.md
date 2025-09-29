---
name: morning_brief
description: Generate a comprehensive morning brief with weather, calendar, news, and tasks
model: ollama:qwen2.5-coder:14b
max_steps: 10
tools: [fetch_json, search_memory, write_file, run]
prefetch:
  - tool: search_memory
    args: { query: "routine preferences morning" }
    assign: preferences
permissions_profile: automation_safe
context_budget:
  tokens: 16000
  priority: [system, task, preferences]
instructions: |
  Keep the briefing punchy, highlight risks, and tailor tone to the stored user preferences.
---

# System
You are a helpful assistant that creates personalized morning briefings. Be concise and highlight important items.

# Context
- Current time: {{ now() }}
- User preferences: {{ preferences }}

# Task: Generate Morning Brief

## Gather Data

### Weather
Fetch current weather and forecast
<!-- tsu:tool name=fetch_json args={"url": "https://api.weather.com/v1/location/{{ env.LOCATION }}/forecast"} assign=weather_data -->

### Calendar
Get today's calendar events from memory
<!-- tsu:tool name=search_memory args={"query": "calendar {{ today() }}"} assign=calendar_events -->

### Tasks
Check pending tasks and deadlines
<!-- tsu:tool name=search_memory args={"query": "todo pending high-priority"} assign=pending_tasks -->

### News
Get top news headlines
<!-- tsu:tool name=fetch_json args={"url": "https://api.news.com/headlines?category=tech,business&limit=5"} assign=news -->

### System Status
Check system status if homelab user
<!-- tsu:cond when="{{ 'homelab' in preferences }}" -->
<!-- tsu:tool name=run args={"command": "systemctl status --no-pager | head -20"} assign=system_status -->
<!-- /tsu:cond -->

## Generate Brief
Now create the morning brief combining all information:

**Weather**: {{ weather_data.current.temp }}°F, {{ weather_data.current.condition }}
Forecast: {{ weather_data.forecast[0].summary }}

**Calendar** ({{ calendar_events|length }} events):
{{ calendar_events }}

**Priority Tasks**: {{ pending_tasks|length }} items pending

**News Headlines**:
{{ news.articles[:3] }}

{{ "**System Status**: " + system_status if system_status else "" }}

Generate a well-formatted markdown brief highlighting:
- Any weather concerns
- Important meetings/events
- Top 3 priority tasks
- Relevant news items
- System alerts (if any)

<!-- tsu:await output=brief_content -->

## Save Brief
<!-- tsu:tool name=write_file args={"path": "~/briefs/{{ today() }}_morning.md", "content": "{{ brief_content }}"} -->

## Notify
<!-- tsu:cond when="{{ preferences.send_email }}" -->
Send the brief via email
<!-- tsu:tool name=send_email args={"to": "{{ env.USER_EMAIL }}", "subject": "Morning Brief - {{ today() }}", "body": "{{ brief_content }}"} -->
<!-- /tsu:cond -->

Brief saved to ~/briefs/{{ today() }}_morning.md