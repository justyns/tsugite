---
name: memory_best_practices
description: Best practices for using the persistent memory system effectively
---

# Memory System Best Practices

You have access to persistent memory tools for storing and recalling information across sessions.

## Memory Types

Choose the right type for each piece of information:

| Type | Purpose | Required Metadata |
|------|---------|-------------------|
| `fact` | Persistent truths (preferences, names, relationships) | `source` |
| `event` | Time-bound occurrences (meetings, milestones) | `event_date` |
| `instruction` | Directives for future behavior | - |
| `note` | General information (default) | - |

## When to Store Memories

**DO store:**
- User preferences and settings
- Personal information shared by the user
- Important events and milestones
- Instructions for future sessions
- Decisions and their rationale

**DON'T store:**
- Transient/temporary information
- Easily re-derivable data
- Sensitive credentials or secrets
- Every conversation detail

## Storing Best Practices

```python
# Facts - always include source
memory_store(
    content="User's cat is named Luna",
    memory_type="fact",
    tags="pet,personal",
    metadata='{"source": "user_stated"}'
)

# Events - always include event_date
memory_store(
    content="Vet appointment for Luna",
    memory_type="event",
    tags="pet,appointment",
    metadata='{"event_date": "2024-12-15", "location": "Downtown Vet"}'
)

# Instructions - for behavioral guidance
memory_store(
    content="User prefers concise responses",
    memory_type="instruction",
    tags="preferences"
)
```

## Retrieval Patterns

```python
# Semantic search - find by meaning
results = memory_search("user pets", limit=5)
print(results)

# Recent memories - time-based
recent = memory_list(since="7d", limit=10)
print(recent)

# Filter by type
facts = memory_list(memory_type="fact", limit=20)
print(facts)

# Date range search
memory_search("project updates", since="2024-11-01", until="2024-11-30")
```

## Before Answering Personal Questions

Always search memories first when the user asks about themselves or past interactions:

```python
# User asks: "What's my cat's name?"
results = memory_search("cat pet name", memory_type="fact")
print(results)  # Check results before answering
```

## Tag Conventions

Use consistent, descriptive tags:
- `personal` - User personal information
- `work` - Work/professional context
- `preferences` - User preferences
- `project:<name>` - Project-specific info
- `pet` - Pet-related information
