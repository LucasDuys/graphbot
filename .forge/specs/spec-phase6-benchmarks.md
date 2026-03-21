---
domain: phase6-benchmarks
status: approved
created: 2026-03-21
complexity: complex
linked_repos: []
---

# Phase 6: Benchmarking, Decomposition Fixes, Knowledge Graph Population

## Overview

Prove the GraphBot thesis with real benchmarks. Fix decomposition reliability.
Populate the knowledge graph through actual usage. Compare against baselines.

## Requirements

### R001: Expanded Benchmark Task Suite
Build a comprehensive set of benchmark tasks beyond the 5 canonical ones.
**Acceptance Criteria:**
- [ ] `benchmarks/tasks.json` with 15 tasks across categories:
  - 5 simple (math, facts, definitions) -- should NOT decompose
  - 5 parallel (multi-entity comparison, weather, listings) -- should decompose to parallel leaves
  - 5 sequential (read->parse->format, multi-step reasoning) -- should decompose to sequential chain
- [ ] Each task has: description, expected_behavior (simple/parallel/sequential), expected_min_nodes, difficulty (1-5)
- [ ] `scripts/run_benchmarks.py` runs all 15 tasks through Orchestrator
- [ ] Records per-task: success, nodes, tokens, latency, cost, model_used, decomposition_worked
- [ ] Outputs results to `benchmarks/results/YYYY-MM-DD-full.json` and prints summary table
- [ ] Test: benchmark runner executes without error on mocked provider

### R002: A/B Comparison Against Baselines
Compare GraphBot vs single LLM call (baseline) and vs Nanobot-style execution.
**Acceptance Criteria:**
- [ ] Enhanced `scripts/compare.py` runs 3 configurations per task:
  - A) GraphBot full pipeline (free models, decomposition, parallel execution, graph context)
  - B) Single LLM call on same free model (8B, no decomposition, no context)
  - C) Single LLM call on 70B model (baseline "big model, no context")
- [ ] Comparison report includes quality assessment (output length, completeness heuristic)
- [ ] Aggregate report across all 15 tasks: win/loss/tie per metric
- [ ] Saved to `benchmarks/comparisons/aggregate-YYYY-MM-DD.md`

### R003: Fix Decomposition Reliability
Ensure decomposition produces valid trees consistently.
**Acceptance Criteria:**
- [ ] Decomposer uses complexity 3 (70B model) for decomposition -- NOT 8B
- [ ] Add JSON mode hint to OpenRouter calls when decomposing: `response_format={"type": "json_object"}`
- [ ] Improve DecompositionPrompt: add 2 more few-shot examples, simplify schema description
- [ ] Add retry with relaxed validation (accept partial trees, fix up missing fields)
- [ ] Benchmark: >80% valid decomposition rate on the 10 decomposable tasks (parallel + sequential)
- [ ] Test: decomposition success rate measured across 10 tasks with mocked valid/invalid responses

### R004: Knowledge Graph Population Through Benchmarks
Every benchmark run populates the knowledge graph with real data.
**Acceptance Criteria:**
- [ ] GraphUpdater called after every benchmark task execution
- [ ] After full benchmark run: verify graph has Task nodes, ExecutionTree nodes, PatternNodes
- [ ] Track pattern cache hit rate across repeated runs
- [ ] `scripts/graph_stats.py` prints graph statistics (node counts by type, edge counts, pattern count)
- [ ] Run benchmarks twice: second run should show some pattern cache hits
- [ ] Test: graph_stats script runs and outputs valid data

### R005: Benchmark Documentation
Document all benchmark results for the README and project docs.
**Acceptance Criteria:**
- [ ] `benchmarks/RESULTS.md` with formatted comparison tables
- [ ] README.md updated with benchmark results section
- [ ] PROGRESS.md updated with Phase 6 completion
- [ ] Include: task-by-task breakdown, aggregate metrics, GraphBot vs baseline analysis
