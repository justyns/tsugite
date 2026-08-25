---
name: morning_brief
description: Generate a morning brief with weather, news, and system status
model: ollama:qwen2.5-coder:14b
max_turns: 10
tools: [fetch_json, read_file, write_file, run]
prefetch:
  - tool: read_file
    args: { path: "~/.config/tsugite/morning_preferences.md" }
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

**News Headlines**:
{{ news.articles[:3] }}

{{ "**System Status**: " + system_status if system_status else "" }}

Generate a well-formatted markdown brief highlighting:
- Any weather concerns
- Relevant news items
- System alerts (if any)

<!-- tsu:await output=brief_content -->

## Save Brief
<!-- tsu:tool name=write_file args={"path": "~/briefs/{{ today() }}_morning.md", "content": "{{ brief_content }}"} -->

Brief saved to ~/briefs/{{ today() }}_morning.md