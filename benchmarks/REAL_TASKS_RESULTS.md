# Real-World Task Results

## Overview

Real-world tasks exercise the tool subsystem (file, web, shell) through the
full Orchestrator pipeline. Unlike knowledge-only benchmarks, these tasks
require the model to delegate to tools and (optionally) synthesize tool output.

## Task Categories

| Category | Count | Description |
|----------|-------|-------------|
| file     | 4     | File list, read, search operations |
| web      | 3     | Web search and summarization |
| shell    | 3     | Shell command execution |

## Results

_Run `python scripts/run_real_tasks.py` to populate this section._

### Summary

| Metric           | Value |
|------------------|-------|
| Total tasks      | 10    |
| Success rate     | --/10 |
| Tool usage rate  | --/10 |
| Total tokens     | --    |
| Total cost       | $--   |

### Per-Task Results

| Task ID  | Category | Difficulty | Success | Tool Used | Model | Tokens | Latency |
|----------|----------|------------|---------|-----------|-------|--------|---------|
| real_01  | file     | 1          | --      | --        | --    | --     | --      |
| real_02  | file     | 2          | --      | --        | --    | --     | --      |
| real_03  | file     | 1          | --      | --        | --    | --     | --      |
| real_04  | file     | 2          | --      | --        | --    | --     | --      |
| real_05  | web      | 2          | --      | --        | --    | --     | --      |
| real_06  | shell    | 1          | --      | --        | --    | --     | --      |
| real_07  | web      | 3          | --      | --        | --    | --     | --      |
| real_08  | shell    | 2          | --      | --        | --    | --     | --      |
| real_09  | file     | 2          | --      | --        | --    | --     | --      |
| real_10  | web      | 2          | --      | --        | --    | --     | --      |

### Category Breakdown

| Category | Success | Tool Used |
|----------|---------|-----------|
| file     | --/4    | --/4      |
| web      | --/3    | --/3      |
| shell    | --/3    | --/3      |

## Analysis

_Fill in after running the benchmark suite._

### Key Observations

1. --
2. --
3. --

### Issues Found

- --

### Next Steps

- --
