---
agent_under_test: list_maker.md
test_id: list_maker_basic
timeout: 30
category: basic
description: Test ability to create formatted lists with required items
---

# Test Cases for List Maker Agent

## Test Case 1: Primary Colors
**Prompt:** "List the three primary colors"

**Expected Behaviors:**
- Should list red, blue, and yellow

**Evaluation:**
- contains: ["red", "blue", "yellow"]
- min_length: 20

## Test Case 2: Days of Week
**Prompt:** "List the days of the week"

**Expected Behaviors:**
- Should list all seven days

**Evaluation:**
- contains: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
- min_length: 50

## Test Case 3: Simple Numbers
**Prompt:** "List numbers 1 through 5"

**Expected Behaviors:**
- Should contain numbers 1, 2, 3, 4, 5

**Evaluation:**
- contains: ["1", "2", "3", "4", "5"]
- min_length: 10
