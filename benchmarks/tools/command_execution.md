---
name: command_execution
test_id: tools_002
description: "Test system command execution"
model: "{{ model }}"
tools: [run]
timeout: 30
expected_type: "string"
weight: 2.0
requires_tools: ["run"]
evaluation_criteria:
  command_success:
    type: "keyword"
    keywords: ["Hello", "System"]
    weight: 1.0
---

# Task
Execute a system command to output "Hello System" and show the result.

<!-- tsu:tool name=run args={"command": "echo 'Hello System'"} assign=command_output -->

Command executed successfully.
Output: {{ command_output }}