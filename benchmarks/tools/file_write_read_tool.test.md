---
agent_under_test: file_write_read_tool.md
test_id: file_write_read_tool_chaining
timeout: 90
category: tools
description: Test ability to chain multiple tool calls (write then read) correctly
---

# Test Cases for File Write and Read Tool

## Test Case 1: Write and Verify Simple Text
**Prompt:** "Write 'Test content 123' to /tmp/test_chain_1.txt and then read it back to verify"

**Expected Behaviors:**
- Should use both write_file and read_file tools

**Evaluation:**
- contains: ["Test content 123"]
- min_length: 15

## Test Case 2: Write and Read Back
**Prompt:** "Create /tmp/test_chain_2.txt with the text 'Hello from benchmark' and confirm what was written"

**Expected Behaviors:**
- Should use both tools in sequence

**Evaluation:**
- contains: ["Hello from benchmark"]
- min_length: 15

## Test Case 3: Write Multiple Lines and Read
**Prompt:** "Write a file to /tmp/test_chain_3.txt with two lines: 'First line' and 'Second line', then read it back"

**Expected Behaviors:**
- Should use both tools correctly

**Evaluation:**
- contains: ["First line", "Second line"]
- min_length: 20
