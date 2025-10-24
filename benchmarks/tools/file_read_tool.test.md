---
agent_under_test: file_read_tool.md
test_id: file_read_tool_basic
timeout: 60
category: tools
description: Test ability to call read_file tool correctly on first try
---

# Test Cases for File Read Tool

## Test Case 1: Read Simple Text File
**Prompt:** "Read the file benchmarks/tools/fixtures/sample.txt and tell me what it contains"

**Evaluation:**
- tool_called: read_file
- contains: ["Hello from sample.txt"]
- min_length: 10

## Test Case 2: Read Data File
**Prompt:** "What is in the file benchmarks/tools/fixtures/data.txt?"

**Evaluation:**
- tool_called: read_file
- contains: ["test data", "line"]
- min_length: 10

## Test Case 3: Read JSON File
**Prompt:** "Read benchmarks/tools/fixtures/config.json"

**Evaluation:**
- tool_called: read_file
- contains: ["test"]
- min_length: 5
