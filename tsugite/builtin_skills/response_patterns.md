---
name: response_patterns
description: How to respond effectively - when to use print/send_message/final_answer
---

# Response Patterns

## Output Channels

| Function | Who Sees | Execution | Use For |
|----------|----------|-----------|---------|
| `print(x)` | You (next turn) | Continues | Intermediate data |
| `send_message(msg)` | User | Continues | Progress updates |
| `final_answer(msg)` | User | **Stops** | Final response |

## Simple Responses

No functions needed? One line:

```python
final_answer("Hello! What can I help you with?")
```

More examples:
- "Thanks" → `final_answer("You're welcome!")`
- "What's 2+2?" → `final_answer("4")`
- "What can you do?" → `final_answer("I can read/write files, run commands, search code, and help with tasks.")`

## Progress Updates

For longer tasks, keep user informed:

```python
send_message("Searching codebase...")
results = search_code("def main")
print(results)  # You'll see this next turn
```

```python
send_message(f"Found {len(results)} matches. Analyzing...")
# ... more work ...
final_answer("Here's what I found: ...")
```

## Anti-patterns

❌ Don't:
- Build menus or numbered option lists
- Write code just to print text
- Offer choices when a direct answer works
- Over-engineer simple responses
- Use print() for user-facing output

✅ Do:
- `final_answer()` immediately for simple responses
- `send_message()` for progress on tasks > 5 seconds
- `print()` only for data YOU need to see
- Match complexity to the task
