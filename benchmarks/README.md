# Benchmarks

This directory now holds the benchmark-ready agents alongside their `.test.md` specifications. Each agent lives in a category subdirectory (currently `basic/`) and is paired with a sibling test file of the same name plus the `.test.md` suffix.

```
benchmarks/
└── basic/
    ├── hello_world.md
    ├── hello_world.test.md
    ├── research_agent.md
    └── research_agent.test.md
```

Add new benchmarks by dropping additional agent/test pairs into category subdirectories. The CLI and test suite will automatically discover them using the shared `BenchmarkRunner` logic.
