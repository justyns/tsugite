---
agent_under_test: echo.md
test_id: echo_basic
timeout: 30
category: basic
description: Test ability to repeat exact phrases
---

# Test Cases for Echo Agent

## Test Case 1: Simple Phrase
**Prompt:** "Repeat exactly: The quick brown fox"

**Expected Output:** "The quick brown fox"

**Evaluation:**
- exact_match: "The quick brown fox"

## Test Case 2: With Punctuation
**Prompt:** "Repeat exactly: Hello, world!"

**Expected Output:** "Hello, world!"

**Evaluation:**
- exact_match: "Hello, world!"

## Test Case 3: Numbers and Letters
**Prompt:** "Repeat exactly: Test123"

**Expected Output:** "Test123"

**Evaluation:**
- exact_match: "Test123"
