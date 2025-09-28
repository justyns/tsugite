---
name: hello_world
test_id: basic_001
description: "Simplest template test - should output hello world"
model: "{{ model }}"
timeout: 10
expected_output: "Hello, World!"
expected_type: "string"
weight: 1.0
evaluation_criteria:
  completeness:
    type: "keyword"
    keywords: ["Hello", "World"]
    weight: 1.0
---

# Task
Output exactly: "Hello, World!"