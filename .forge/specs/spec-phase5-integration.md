---
domain: phase5-integration
status: approved
created: 2026-03-21
complexity: medium
linked_repos: []
---

# Phase 5: Integration, Quality, and End-to-End Validation

## Overview

Final phase: run all 5 canonical test tasks end-to-end, build the A/B comparison
script, add the graph seeding script, update all documentation, and validate the
complete pipeline works with real models.

## Requirements

### R001: Graph Seed Script
Seed the knowledge graph with initial data for a real user session.
**Acceptance Criteria:**
- [ ] `scripts/seed_graph.py` seeds the graph with user profile, projects, memories
- [ ] Idempotent: running twice doesn't create duplicates (check before insert)
- [ ] Seeds: User (Lucas), Projects (GraphBot, Pitchr), Memories (5+ facts), Services (OpenRouter, Kuzu)
- [ ] Loads data from .env.local for API keys (doesn't hardcode)
- [ ] Test: run seed, verify nodes exist in graph

### R002: A/B Comparison Script
Compare GraphBot output vs a reference model.
**Acceptance Criteria:**
- [ ] `scripts/compare.py` takes a task description as CLI argument
- [ ] Runs task on GraphBot (full pipeline via Orchestrator)
- [ ] Runs same task on a single LLM call (baseline, no decomposition)
- [ ] Outputs side-by-side report: quality, tokens, latency, cost
- [ ] Reports saved to `benchmarks/comparisons/` as markdown
- [ ] Test: script runs without error on a simple task

### R003: End-to-End Test Suite
Run all 5 canonical test tasks from TESTS.md.
**Acceptance Criteria:**
- [ ] `tests/test_integration/test_canonical.py` with mocked provider
- [ ] Test 1: "What's 247 * 38?" -> single node, correct routing
- [ ] Test 2: "Weather in Amsterdam, London, Berlin" -> parallel decomposition
- [ ] Test 3: "Read README.md, find TODOs" -> sequential decomposition
- [ ] Test 4: Complex research task -> multi-wave decomposition
- [ ] Test 5: Pattern cache hit (run Test 2 twice, second should be faster)
- [ ] All tests use mocked provider (deterministic, no API cost)
- [ ] Each test verifies: success, node count, output structure

### R004: Documentation Update
Update all project docs to reflect completed state.
**Acceptance Criteria:**
- [ ] PROGRESS.md: all phases marked complete, metrics updated
- [ ] TESTS.md: benchmark table populated with real results
- [ ] README section or docs/QUICKSTART.md: how to set up and run GraphBot
- [ ] Final test count and code metrics updated
