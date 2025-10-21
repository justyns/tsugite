---
agent_under_test: file_write_tool.md
test_id: file_write_tool_basic
timeout: 60
category: tools
description: Test ability to call write_file tool correctly on first try
---

# Test Cases for File Write Tool

## Test Case 1: Write Simple Text
**Prompt:** "Write 'Hello World' to /tmp/test_write_1.txt"

**Evaluation:**
- tool_called: write_file
- contains: ["Success", "wrote"]
- min_length: 10

## Test Case 2: Write Multiple Lines
**Prompt:** "Create a file at /tmp/test_write_2.txt with the content: Line 1\nLine 2\nLine 3"

**Evaluation:**
- tool_called: write_file
- contains: ["Success", "wrote"]
- min_length: 10

## Test Case 3: Write JSON Content
**Prompt:** "Write this JSON to /tmp/test_write_3.json: {\"name\": \"test\", \"value\": 123}"

**Evaluation:**
- tool_called: write_file
- contains: ["Success", "wrote"]
- min_length: 10
