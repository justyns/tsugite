---
agent_under_test: file_write_read_tool.md
test_id: file_write_read_tool_chaining
timeout: 90
category: tools
description: Test ability to chain multiple tool calls (write then read) correctly
---

# Test Cases for File Write and Read Tool

## Test Case 1: Write and Verify Simple Text
**Prompt:** "Write 'Test content 123' to /tmp/test_chain_1.txt and then read the file and return it"

**Evaluation:**
- tool_called: ["write_file", "read_file"]
- contains: ["Test content 123"]
- min_length: 15

## Test Case 2: Write and Read Back
**Prompt:** "Create /tmp/test_chain_2.txt with the text 'Hello from benchmark' and then read the file and return it"

**Evaluation:**
- tool_called: ["write_file", "read_file"]
- contains: ["Hello from benchmark"]
- min_length: 15

## Test Case 3: Write Multiple Lines and Read
**Prompt:** "Write a file to /tmp/test_chain_3.txt with two lines: 'First line' and 'Second line', then read the file and return it"

**Evaluation:**
- tool_called: ["write_file", "read_file"]
- contains: ["First line", "Second line"]
- min_length: 20
