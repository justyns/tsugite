---
name: file_operations
test_id: tools_001
description: "Test file read/write operations"
model: "{{ model }}"
tools: [write_file, read_file]
timeout: 30
expected_type: "string"
weight: 2.0
requires_tools: ["write_file", "read_file"]
evaluation_criteria:
  file_created:
    type: "keyword"
    keywords: ["Hello", "Benchmark"]
    weight: 0.5
  file_read:
    type: "keyword"
    keywords: ["successfully", "read"]
    weight: 0.5
---

# Task
1. Write "Hello Benchmark Test" to a file named "test.txt"
2. Read the content back from the file
3. Confirm the content matches what was written

<!-- tsu:tool name=write_file args={"path": "test.txt", "content": "Hello Benchmark Test"} assign=write_result -->

<!-- tsu:tool name=read_file args={"path": "test.txt"} assign=file_content -->

File content: {{ file_content }}

The file was successfully created and read. Content matches: {{ "Hello Benchmark Test" in file_content }}