---
name: python_best_practices
description: Python coding best practices and common patterns
---

# Python Best Practices

When writing Python code, follow these patterns:

## Error Handling

Always use try/except for operations that might fail:

```python
try:
    result = risky_operation()
except SpecificException as e:
    # Handle the specific error
    print(f"Error: {e}")
    # Consider alternative approach
```

## File Operations

Use context managers for file handling:

```python
with open("file.txt", "r") as f:
    content = f.read()
    # File automatically closed
```

## Path Handling

Use `pathlib.Path` for cross-platform path operations:

```python
from pathlib import Path

file_path = Path("data") / "file.txt"
if file_path.exists():
    content = file_path.read_text()
```

## Iteration

Use enumerate when you need both index and value:

```python
for i, item in enumerate(items):
    print(f"{i}: {item}")
```

## Debugging

Add informative print statements:

```python
print(f"Processing {len(items)} items...")
print(f"Current value: {variable!r}")  # !r shows repr()
```

## Code Organization

Break complex operations into smaller, testable functions:

```python
def process_data(data):
    cleaned = clean_data(data)
    validated = validate_data(cleaned)
    return transform_data(validated)
```
