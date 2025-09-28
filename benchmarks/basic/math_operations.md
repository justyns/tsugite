---
name: math_operations
test_id: basic_002
description: "Basic arithmetic calculation test"
model: "{{ model }}"
timeout: 15
expected_output: "42"
expected_type: "number"
weight: 1.0
evaluation_criteria:
  accuracy:
    type: "keyword"
    keywords: ["42"]
    weight: 1.0
---

# Task
Calculate the result: (10 * 4) + 2

Return only the numeric result.