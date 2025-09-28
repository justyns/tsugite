---
name: code_generation
test_id: scenarios_002
description: "Generate and test Python code"
model: "{{ model }}"
tools: [write_file, run]
timeout: 90
expected_type: "code"
weight: 3.0
evaluation_criteria:
  code_created:
    type: "keyword"
    keywords: ["def", "fibonacci", "return"]
    weight: 0.4
  code_works:
    type: "keyword"
    keywords: ["0", "1", "1", "2", "3", "5", "8"]
    weight: 0.6
---

# Task
Generate Python code that implements the Fibonacci sequence function and test it.

Requirements:
1. Function should be named `fibonacci`
2. Takes one parameter `n`
3. Returns the nth Fibonacci number
4. Handle edge cases (n=0, n=1)
5. Test with n=7 and show the result

<!-- tsu:tool name=write_file args={"path": "fibonacci.py", "content": "def fibonacci(n):\n    \"\"\"Calculate the nth Fibonacci number.\"\"\"\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        a, b = 0, 1\n        for _ in range(2, n + 1):\n            a, b = b, a + b\n        return b\n\n# Test the function\nif __name__ == '__main__':\n    for i in range(8):\n        print(f'fibonacci({i}) = {fibonacci(i)}')\n    \n    print(f'\\nResult for n=7: {fibonacci(7)}')"} assign=code_written -->

<!-- tsu:tool name=run args={"command": "python3 fibonacci.py"} assign=test_output -->

## Generated Code

I've created a Fibonacci function with the following implementation:

```python
def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
```

## Test Results

{{ test_output }}

The function correctly implements the Fibonacci sequence and handles edge cases. For n=7, the result is 13.