---
name: scheduling
description: How to create, manage, and monitor scheduled agent tasks using Tsugite's schedule tools
---

# Scheduling Agent Tasks

Schedule tools let agents create recurring (cron) or one-off scheduled tasks that run on the daemon. These tools access the scheduler directly and are only available when running inside the daemon (`tsu daemon`). They won't appear in standalone `tsu run` mode.

**IMPORTANT — Safety Rules:**
- Only use `schedule_*` tools for scheduling. NEVER create shell scripts, cron jobs, or other workarounds.
- If schedule tools are not available, tell the user the daemon isn't running and stop. Do not improvise alternatives.
- **NEVER create a schedule without explicit user confirmation.** Always show the user the exact prompt, schedule, and timezone before calling `schedule_create`.
- **NEVER schedule destructive or dangerous actions** (deleting files, dropping databases, force-pushing, modifying infrastructure, sending messages to external services, etc.). If the user asks for something potentially destructive, warn them and refuse unless they explicitly acknowledge the risk.
- Scheduled tasks run unattended — the user will NOT be present to intervene. Every scheduled prompt must be safe to execute autonomously.
- The `agent` parameter is optional — it defaults to the current agent. Only set it if the user asks to schedule a different agent.

## Available Tools

| Tool | Purpose |
|------|---------|
| `schedule_list()` | List all schedules with status |
| `schedule_create(...)` | Create a cron or one-off schedule |
| `schedule_remove(id)` | Delete a schedule |
| `schedule_enable(id)` | Re-enable a disabled schedule |
| `schedule_disable(id)` | Disable without deleting |

## Crafting the Prompt

The `prompt` field is what the scheduled agent will receive as its task. **Do NOT pass the user's words verbatim.** Instead, interpret their intent and write a clear, direct instruction for the agent.

**Why:** The user is talking *to you* about what they want scheduled. The prompt is what *the scheduled agent* will execute later, alone, with no user present.

**Rules for prompt crafting:**
1. **Remove meta-language.** If the user says "ask the agent to say hello", the prompt should be `Say hello`, not `Ask the agent to say hello` (which would cause the agent to try spawning a subagent).
2. **Be direct and specific.** Write the prompt as a clear instruction the agent can act on immediately.
3. **Include necessary context.** If the task needs specific details (file paths, URLs, formats), include them in the prompt.
4. **Keep it self-contained.** The agent won't have conversational context when it runs — the prompt must stand alone.
5. **Avoid ambiguity.** If the user's request is vague, ask clarifying questions before creating the schedule.

**Examples of prompt rewriting:**

| User says | Bad prompt (verbatim) | Good prompt (rewritten) |
|---|---|---|
| "Have the agent say hello every morning" | "Have the agent say hello every morning" | "Say hello and greet the user with a friendly message" |
| "Ask it to check if my server is up" | "Ask it to check if my server is up" | "Check if the server at example.com is responding to HTTP requests and report the status" |
| "I want a daily summary of my git repos" | "I want a daily summary of my git repos" | "List recent commits from the last 24 hours across all git repositories in ~/projects and summarize the changes" |
| "Tell the agent to remind me about the meeting" | "Tell the agent to remind me about the meeting" | "Remind the user about their upcoming meeting" |

## Creating Schedules

### Recurring (Cron)

```python
schedule_create(
    id="daily-digest",
    prompt="Summarize today's inbox and create a digest",
    cron="0 9 * * *",
    timezone="America/Chicago"
)
```

### One-Off

```python
schedule_create(
    id="deploy-reminder",
    prompt="Remind the user to deploy the staging branch",
    run_at="2026-02-14T10:00:00-06:00"
)
```

**Parameters:**
- `id` — Unique name for the schedule (lowercase, hyphens)
- `prompt` — A clear, direct instruction for the agent (see "Crafting the Prompt" above). Do NOT copy the user's words verbatim.
- `agent` — *(optional)* Defaults to the current agent. Only set if scheduling a different agent.
- `cron` — Standard 5-field cron expression (mutually exclusive with `run_at`)
- `run_at` — ISO 8601 datetime for one-off tasks (mutually exclusive with `cron`)
- `timezone` — IANA timezone name (default: `UTC`)
- `notify` — *(optional)* List of notification channel names. The final result is automatically sent to these channels when the task completes.
- `notify_tool` — *(optional, default: false)* When `true`, the agent gets the `notify_user` tool so it can send messages during execution. Requires `notify` to be set.

## Cron Expression Reference

```
┌───────── minute (0-59)
│ ┌─────── hour (0-23)
│ │ ┌───── day of month (1-31)
│ │ │ ┌─── month (1-12)
│ │ │ │ ┌─ day of week (0-6, Sun=0)
│ │ │ │ │
* * * * *
```

| Expression | Meaning |
|------------|---------|
| `0 9 * * *` | Daily at 9:00 AM |
| `*/15 * * * *` | Every 15 minutes |
| `0 9 * * 1-5` | Weekdays at 9:00 AM |
| `0 0 1 * *` | First of each month at midnight |
| `30 8,17 * * *` | 8:30 AM and 5:30 PM daily |

## Listing and Checking Status

```python
schedules = schedule_list()
print(schedules)
```

Each schedule includes: `id`, `agent`, `schedule_type`, `enabled`, `next_run`, `last_run`, `last_status`, `last_error`.

## Managing Schedules

```python
# Pause a schedule
schedule_disable("daily-digest")

# Resume it
schedule_enable("daily-digest")

# Remove permanently
schedule_remove("deploy-reminder")
```

## Workflow: User Asks to Schedule Something

1. **Check existing schedules** to avoid duplicates:
```python
schedules = schedule_list()
print(schedules)
```

2. **Clarify the request.** If anything is ambiguous, ask:
   - What exactly should the agent do? (get specific actions, not vague goals)
   - What schedule/frequency?
   - What timezone?
   - Are there any safety concerns? (file modifications, external calls, etc.)

3. **Craft the prompt** following the rules in "Crafting the Prompt" above. Rewrite the user's intent into a direct, self-contained instruction.

4. **Show the user exactly what will be created** and ask for confirmation:
   > I'll create this schedule:
   > - **ID:** `weekly-report`
   > - **Prompt:** "Generate the weekly project status report covering all commits and open PRs"
   > - **Schedule:** Every Friday at 5:00 PM (America/New_York)
   >
   > Does this look right?

5. **Only after the user confirms**, create the schedule:
```python
result = schedule_create(
    id="weekly-report",
    prompt="Generate the weekly project status report covering all commits and open PRs",
    cron="0 17 * * 5",
    timezone="America/New_York"
)
print(result)
```

6. Confirm to the user with the `next_run` from the result.

## Notifications

Scheduled tasks can send notifications to the user via configured channels (Discord DMs, webhooks, etc.). Channels are defined in `daemon.yaml` under `notification_channels`.

### Auto-Notify (deliver result on completion)

Set `notify` to automatically send the final result when the task finishes:

```python
schedule_create(
    id="daily-report",
    prompt="Generate a daily status report",
    cron="0 9 * * *",
    notify=["my-discord", "alerts"]
)
```

### LLM-Driven Notifications (send messages during execution)

Set both `notify` and `notify_tool=True` to give the agent the `notify_user` tool. The agent can send progress updates, alerts, or findings at any point during execution:

```python
schedule_create(
    id="monitoring",
    prompt="Check server health and notify if any issues are found",
    cron="*/30 * * * *",
    notify=["alerts"],
    notify_tool=True
)
```

### Both (auto-notify + tool)

```python
schedule_create(
    id="full-check",
    prompt="Run security audit and report findings",
    cron="0 9 * * *",
    notify=["my-discord"],
    notify_tool=True
)
```

The agent can send intermediate alerts during execution, and the final result is also delivered automatically on completion.

**Rules:**
- `notify_tool=True` requires `notify` to be set (the tool needs channels to send to)
- Channel names must match entries in the daemon's `notification_channels` config
- Failed notifications are logged but do not affect the schedule's status

## Error Handling

- **Tools not available** — Schedule tools only appear in daemon mode. Tell the user to start the daemon (`tsu daemon`) and do NOT create shell scripts or other workarounds.
- **"Schedule already exists"** — IDs must be unique. Remove or pick a different name.
- **"Invalid cron expression"** — Check syntax against the reference above.
- **Agent not found** — The `agent` parameter must match a key in the daemon's `agents:` config. Omit `agent` to use the current agent.
