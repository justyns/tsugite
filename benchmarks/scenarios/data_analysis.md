---
name: data_analysis
test_id: scenarios_001
description: "Analyze data and provide insights"
model: "{{ model }}"
tools: [write_file, read_file]
timeout: 90
expected_type: "string"
weight: 3.0
evaluation_criteria:
  analysis_present:
    type: "keyword"
    keywords: ["analysis", "average", "total", "summary"]
    weight: 0.3
  insights:
    type: "keyword"
    keywords: ["insights", "trends", "patterns", "conclusion"]
    weight: 0.3
  completeness:
    type: "length"
    min_length: 100
    max_length: 1000
    weight: 0.4
---

# Task
Analyze the following sales data and provide insights:

## Sample Data
```
Product,Sales,Price
Widget A,150,25.99
Widget B,230,19.99
Widget C,75,39.99
Gadget X,180,15.99
Gadget Y,95,29.99
```

Provide:
1. Total sales volume
2. Average price
3. Best selling product
4. Revenue analysis
5. Key insights and recommendations

<!-- tsu:tool name=write_file args={"path": "sales_data.csv", "content": "Product,Sales,Price\nWidget A,150,25.99\nWidget B,230,19.99\nWidget C,75,39.99\nGadget X,180,15.99\nGadget Y,95,29.99"} assign=data_written -->

## Analysis

**Total Sales Volume:** 730 units (150+230+75+180+95)

**Average Price:** $23.59 ((25.99+19.99+39.99+15.99+29.99)/5)

**Best Selling Product:** Widget B (230 units)

**Revenue Analysis:**
- Widget A: $3,898.50
- Widget B: $4,597.70
- Widget C: $2,999.25
- Gadget X: $2,878.20
- Gadget Y: $2,849.05
- **Total Revenue:** $17,222.70

**Key Insights:**
1. Widget B has the best volume-price balance
2. Widget C has highest price but lowest volume - pricing issue?
3. Gadgets generally have lower prices and moderate sales
4. Widget line outperforms Gadget line in revenue

**Recommendations:**
- Consider reducing Widget C price to increase volume
- Investigate if Gadget line needs better marketing
- Widget B's success model could be applied to other products