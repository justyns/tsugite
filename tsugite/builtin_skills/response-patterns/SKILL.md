---
name: response-patterns
description: How to respond effectively - when to use print/send_message/return_value/ask_user
---

# Response Patterns

## Output Channels

| Function | Who Sees | Execution | Use For |
|----------|----------|-----------|---------|
| `print(x)` | You (next turn) | Continues | Intermediate data |
| `send_message(msg)` | User | Continues | Progress updates |
| `return_value(value)` | User | **Stops** | Structured final result |
| `ask_user(q)` | User | **Blocks** | Get user input |

## Simple Responses

No functions needed? Reply in plain text:

- "Thanks" replies `You're welcome!`
- "What's 2+2?" replies `4`
- "What can you do?" replies `I can read/write files, run commands, and help with tasks.`

Call `return_value(value)` only when the caller needs a structured result.

## Progress Updates

For longer tasks, keep user informed:

```python-exec
send_message("Searching codebase...")
results = run("grep -rn 'def main' .")
print(results)  # You'll see this next turn
```

```python-exec
send_message(f"Found {len(results.splitlines())} matches. Analyzing...")
# ... more work ...
return_value("Here's what I found: ...")
```

## Asking the User

When you need clarification or a decision, use `ask_user()`. It blocks until the user responds, then returns their answer as a string. Works in terminal, web UI, and Discord.

**Question types:**

```python-exec
# Freeform text
name = ask_user("What should the file be named?", question_type="text")

# Yes/No (returns "yes" or "no")
confirm = ask_user("Delete these files?", question_type="yes_no")

# Multiple choice (returns the selected option string)
fmt = ask_user("Which format?", question_type="choice", options=["json", "yaml", "toml"])
```

**Batch questions** — ask multiple at once:

```python-exec
answers = ask_user_batch(questions=[
    {"id": "name", "question": "Project name?", "type": "text"},
    {"id": "confirm", "question": "Create directory?", "type": "yes_no"},
    {"id": "lang", "question": "Language?", "type": "choice", "options": ["python", "rust", "go"]},
])
# answers = {"name": "myapp", "confirm": "yes", "lang": "python"}
```

Use `ask_user` when:
- The task is ambiguous and you need to choose between approaches
- A destructive action needs explicit confirmation
- User input is required (names, paths, config values)

## Anti-patterns

❌ Don't:
- Build menus or numbered option lists — use `ask_user(question_type="choice")` instead
- Write code just to print text
- Offer choices when a direct answer works
- Over-engineer simple responses
- Use print() for user-facing output
- Ask unnecessary questions when a reasonable default exists

✅ Do:
- Plain text immediately for simple responses
- `send_message()` for progress on tasks > 5 seconds
- `ask_user()` when you genuinely need user input
- `print()` only for data YOU need to see
- Match complexity to the task
- Before ending a conversation, check: did the user tell you anything worth remembering? If so, write it to a memory file first
