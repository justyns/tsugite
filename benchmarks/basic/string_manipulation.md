---
name: string_manipulation
test_id: basic_003
description: "String processing and manipulation test"
model: "{{ model }}"
timeout: 20
expected_output: "DLROW OLLEH"
expected_type: "string"
weight: 1.0
evaluation_criteria:
  accuracy:
    type: "keyword"
    keywords: ["DLROW", "OLLEH"]
    weight: 1.0
---

# Task
Take the string "HELLO WORLD" and:
1. Reverse the order of characters
2. Return the result

Expected output: "DLROW OLLEH"