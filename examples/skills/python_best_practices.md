---
name: python_best_practices
description: Python coding best practices, modern features, and common patterns for writing clean, maintainable code
---

# Python Best Practices

Modern patterns and conventions for writing clean, maintainable Python code.

## Type Hints

Use type hints for function signatures and complex variables:

```python
from typing import List, Dict, Optional, Union, Tuple

def process_data(items: List[str], config: Dict[str, any]) -> Optional[str]:
    """Process items with configuration."""
    if not items:
        return None
    return items[0]

# Modern Python 3.10+ syntax
def merge_data(data: list[str] | None = None) -> dict[str, int]:
    """Merge data using modern type union syntax."""
    return {item: len(item) for item in (data or [])}

# For complex types
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...

def render(obj: Drawable) -> None:
    obj.draw()
```

**Key types:**
- `List[T]`, `Dict[K, V]`, `Set[T]`, `Tuple[T, ...]` - Collections
- `Optional[T]` - T or None (equivalent to `T | None`)
- `Union[A, B]` - A or B (Python 3.10+: `A | B`)
- `Any` - Any type (use sparingly)
- `Protocol` - Structural subtyping (duck typing with types)

## Dataclasses and Pydantic

Prefer dataclasses for simple data containers:

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class User:
    name: str
    email: str
    age: int = 0
    tags: List[str] = field(default_factory=list)

    def is_adult(self) -> bool:
        return self.age >= 18

# Usage
user = User(name="Alice", email="alice@example.com", age=25)
print(user.name)  # Automatic __init__, __repr__, __eq__
```

Use Pydantic for validation and API models:

```python
from pydantic import BaseModel, EmailStr, Field, validator

class UserModel(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    age: int = Field(..., ge=0, le=150)

    @validator('name')
    def name_must_be_capitalized(cls, v):
        if not v[0].isupper():
            raise ValueError('must start with capital letter')
        return v

# Automatic validation on instantiation
user = UserModel(name="Alice", email="alice@example.com", age=25)
```

## Error Handling

Be specific with exceptions and use context managers:

```python
# Bad - catches everything
try:
    result = risky_operation()
except:
    pass

# Good - specific exceptions
try:
    result = parse_json(data)
except json.JSONDecodeError as e:
    logger.error(f"Invalid JSON: {e}")
    return default_value
except FileNotFoundError:
    logger.warning("Config file not found, using defaults")
    return default_config

# Custom exceptions for domain errors
class ValidationError(Exception):
    """Raised when data validation fails."""
    pass

def validate_age(age: int) -> None:
    if age < 0 or age > 150:
        raise ValidationError(f"Invalid age: {age}")
```

## File Operations

Always use context managers and pathlib:

```python
from pathlib import Path

# Reading files
config_path = Path("config.json")
if config_path.exists():
    content = config_path.read_text()
    data = config_path.read_bytes()

# Writing files
output = Path("output") / "results.txt"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("Results here")

# Context manager for complex operations
with open("data.txt", "r") as f:
    for line in f:
        process(line.strip())

# Temporary files
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir) / "temp_file.txt"
    tmp_path.write_text("temporary data")
    # Cleanup automatic
```

## Iteration and Comprehensions

Use comprehensions and itertools for clean, efficient code:

```python
# List comprehensions
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]

# Dict comprehensions
word_lengths = {word: len(word) for word in ["a", "bb", "ccc"]}

# Set comprehensions
unique_lengths = {len(word) for word in words}

# Generator expressions (memory efficient)
sum_of_squares = sum(x**2 for x in range(1000000))

# enumerate for index and value
for i, item in enumerate(items, start=1):
    print(f"{i}. {item}")

# zip for parallel iteration
names = ["Alice", "Bob"]
ages = [25, 30]
for name, age in zip(names, ages):
    print(f"{name} is {age}")

# itertools for advanced iteration
from itertools import groupby, chain, islice, cycle

# Group consecutive items
data = [1, 1, 2, 2, 2, 3]
for key, group in groupby(data):
    print(key, list(group))

# Flatten nested lists
nested = [[1, 2], [3, 4]]
flat = list(chain.from_iterable(nested))  # [1, 2, 3, 4]
```

## Async/Await

Use async for I/O-bound operations:

```python
import asyncio
import aiohttp

async def fetch_url(session: aiohttp.ClientSession, url: str) -> str:
    """Fetch URL content asynchronously."""
    async with session.get(url) as response:
        return await response.text()

async def fetch_multiple(urls: List[str]) -> List[str]:
    """Fetch multiple URLs concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

# Run async code
urls = ["http://example.com", "http://example.org"]
results = asyncio.run(fetch_multiple(urls))

# Async context managers
async with aiofiles.open("file.txt", "r") as f:
    content = await f.read()

# Async iteration
async for item in async_generator():
    process(item)
```

## Logging

Use logging module, not print statements:

```python
import logging

# Configure logging (do once at application startup)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Use appropriate levels
logger.debug("Detailed debugging information")
logger.info("General informational messages")
logger.warning("Warning messages")
logger.error("Error messages")
logger.exception("Log exception with traceback")

# Structured logging with extra context
logger.info("User logged in", extra={"user_id": 123, "ip": "192.168.1.1"})

# Log exceptions properly
try:
    risky_operation()
except Exception as e:
    logger.exception("Operation failed")  # Includes full traceback
```

## Context Managers

Create custom context managers for resource management:

```python
from contextlib import contextmanager
from typing import Generator

@contextmanager
def temporary_setting(config: Dict, key: str, value: Any) -> Generator:
    """Temporarily change a config setting."""
    old_value = config.get(key)
    config[key] = value
    try:
        yield
    finally:
        if old_value is None:
            config.pop(key, None)
        else:
            config[key] = old_value

# Usage
with temporary_setting(app_config, "debug", True):
    # Debug mode enabled
    run_debug_commands()
# Debug mode restored

# Class-based context manager
class DatabaseConnection:
    def __enter__(self):
        self.conn = connect_to_database()
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()
        return False  # Don't suppress exceptions
```

## Modern Python Features

### Match/Case (Python 3.10+)

```python
def process_command(command: str, args: List[str]) -> None:
    match command:
        case "start":
            start_service()
        case "stop":
            stop_service()
        case "status":
            print_status()
        case "config" if len(args) > 0:
            update_config(args[0])
        case _:
            print(f"Unknown command: {command}")

# Structural pattern matching
def describe_shape(shape: Tuple):
    match shape:
        case (0, 0):
            return "origin"
        case (x, 0):
            return f"on x-axis at {x}"
        case (0, y):
            return f"on y-axis at {y}"
        case (x, y):
            return f"point at ({x}, {y})"
```

### Walrus Operator (Python 3.8+)

```python
# Assign and use in same expression
if (n := len(data)) > 100:
    print(f"Large dataset: {n} items")

# In list comprehensions
valid_items = [item for line in file if (item := parse(line)) is not None]

# While loops
while (line := file.readline()):
    process(line)
```

### F-string Improvements (Python 3.8+)

```python
name = "Alice"
age = 25

# Self-documenting expressions (3.8+)
print(f"{name=}, {age=}")  # name='Alice', age=25

# Format specifiers
value = 1234.5678
print(f"{value:,.2f}")  # 1,234.57
print(f"{value:>10.2f}")  # Right-align in 10 chars

# Date formatting
from datetime import datetime
now = datetime.now()
print(f"{now:%Y-%m-%d %H:%M:%S}")

# Debug formatting
x = 10
print(f"{x=:#x}")  # x=0xa (hex with prefix)
```

## Code Organization

Break code into small, testable functions:

```python
# Bad - monolithic function
def process_order(order_data: Dict) -> Dict:
    # 200 lines of mixed validation, calculation, and side effects
    pass

# Good - separated concerns
def validate_order(order_data: Dict) -> None:
    """Validate order data, raise ValueError if invalid."""
    if not order_data.get("items"):
        raise ValueError("Order must have items")

def calculate_total(items: List[Dict]) -> Decimal:
    """Calculate order total from items."""
    return sum(item["price"] * item["quantity"] for item in items)

def save_order(order: Dict) -> str:
    """Save order to database, return order ID."""
    return database.insert("orders", order)

def process_order(order_data: Dict) -> Dict:
    """Process and save order."""
    validate_order(order_data)
    total = calculate_total(order_data["items"])
    order_id = save_order({**order_data, "total": total})
    return {"order_id": order_id, "total": total}
```

## Testing Patterns

Write testable code with pytest:

```python
import pytest
from unittest.mock import Mock, patch

# Simple test
def test_calculate_total():
    items = [{"price": 10, "quantity": 2}, {"price": 5, "quantity": 3}]
    assert calculate_total(items) == 35

# Parametrized tests
@pytest.mark.parametrize("input,expected", [
    ([1, 2, 3], 6),
    ([], 0),
    ([10], 10),
])
def test_sum_list(input, expected):
    assert sum(input) == expected

# Fixtures for setup
@pytest.fixture
def database():
    db = Database(":memory:")
    db.create_tables()
    yield db
    db.close()

def test_insert(database):
    database.insert("users", {"name": "Alice"})
    assert database.count("users") == 1

# Mocking external dependencies
@patch('requests.get')
def test_fetch_data(mock_get):
    mock_get.return_value = Mock(status_code=200, text="data")
    result = fetch_data("http://api.example.com")
    assert result == "data"

# Test exceptions
def test_validation_error():
    with pytest.raises(ValidationError, match="Invalid age"):
        validate_age(-5)
```

## Virtual Environments

Always use virtual environments:

```bash
# Using venv (built-in)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Using uv (modern, fast)
uv venv
source .venv/bin/activate
uv pip install package-name

# Using poetry
poetry init
poetry add package-name
poetry shell
```

## Common Anti-patterns to Avoid

```python
# ❌ Mutable default arguments
def append_to(item, list=[]):  # Bug! Shared across calls
    list.append(item)
    return list

# ✅ Use None and create new list
def append_to(item, list=None):
    if list is None:
        list = []
    list.append(item)
    return list

# ❌ Catching and ignoring exceptions
try:
    risky_operation()
except:  # Too broad and silent
    pass

# ✅ Be specific and handle appropriately
try:
    risky_operation()
except ValueError as e:
    logger.warning(f"Invalid value: {e}")
    return default_value

# ❌ String concatenation in loops
result = ""
for item in items:
    result += str(item)  # Creates new string each time

# ✅ Use join
result = "".join(str(item) for item in items)

# ❌ Checking type with type()
if type(x) == list:  # Doesn't work with subclasses
    pass

# ✅ Use isinstance
if isinstance(x, list):  # Works with subclasses
    pass
```

## Summary

**Key principles:**
- Use type hints for clarity
- Prefer dataclasses/Pydantic for data models
- Use specific exceptions and context managers
- Leverage comprehensions and itertools
- Use async for I/O-bound operations
- Log instead of print
- Write small, testable functions
- Always use virtual environments
- Stay current with modern Python features

**Resources:**
- PEP 8: Python style guide
- PEP 20: The Zen of Python (`import this`)
- Python docs: docs.python.org
- Real Python: realpython.com
