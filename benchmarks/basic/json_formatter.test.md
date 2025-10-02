---
agent_under_test: json_formatter.md
test_id: json_formatter_basic
timeout: 30
category: basic
description: Test ability to generate valid JSON output
expected_type: json
---

# Test Cases for JSON Formatter Agent

## Test Case 1: Simple Object
**Prompt:** "Create JSON with name: Alice and age: 30"

**Expected Output:** {"name": "Alice", "age": 30}

**Evaluation:**
- valid_json: true
- contains_keys: ["name", "age"]

## Test Case 2: Array
**Prompt:** "Create JSON array with numbers 1, 2, 3"

**Expected Output:** [1, 2, 3]

**Evaluation:**
- valid_json: true

## Test Case 3: Nested Object
**Prompt:** "Create JSON with user object containing name: Bob and city: NYC"

**Expected Output:** {"user": {"name": "Bob", "city": "NYC"}}

**Evaluation:**
- valid_json: true
- contains_keys: ["user"]
