---
name: speed_test
test_id: performance_002
description: "Quick response time test"
model: "{{ model }}"
timeout: 10
expected_output: "FAST"
expected_type: "string"
weight: 2.0
evaluation_criteria:
  speed:
    type: "keyword"
    keywords: ["FAST"]
    weight: 1.0
---

# Task
Respond immediately with exactly: "FAST"

This tests model response speed.