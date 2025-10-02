---
agent_under_test: shell_command_tool.md
test_id: shell_command_tool_basic
timeout: 60
category: tools
description: Test ability to call run_shell_command tool correctly on first try
---

# Test Cases for Shell Command Tool

## Test Case 1: Simple Echo Command
**Prompt:** "Run the command: echo 'Hello Shell'"

**Expected Behaviors:**
- Should use run_shell_command tool

**Evaluation:**
- contains: ["Hello Shell"]
- min_length: 10

## Test Case 2: List Directory
**Prompt:** "Run: ls /tmp | head -5"

**Expected Behaviors:**
- Should use run_shell_command tool

**Evaluation:**
- min_length: 5

## Test Case 3: Print Working Directory
**Prompt:** "What is the current directory? Run pwd"

**Expected Behaviors:**
- Should use run_shell_command tool

**Evaluation:**
- contains: ["/"]
- min_length: 3
