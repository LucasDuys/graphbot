---
domain: phase4-patterns
status: approved
created: 2026-03-21
complexity: medium
linked_repos: []
---

# Phase 4: Pattern Cache + Graph Learning

## Overview

Implement the pattern extraction and caching system that learns from completed tasks.
After each successful execution, extract a reusable template. Before decomposition,
check if a matching pattern exists -- if so, instantiate the template (0 tokens, <5ms).
Also implement the graph update loop that records task outcomes and entity relationships.

## Requirements

### R001: Pattern Extractor
Extract reusable templates from completed task trees.
**Acceptance Criteria:**
- [ ] `PatternExtractor` class in `core_gb/patterns.py`
- [ ] `extract(nodes: list[TaskNode], result: ExecutionResult) -> Pattern | None`
- [ ] Extracts: trigger template (generalized description), tree structure, variable slots
- [ ] Generalization: replace specific entities with variable slots (e.g., "Amsterdam" -> "{city_1}")
- [ ] Only extract from successful executions with 2+ nodes
- [ ] Returns Pattern dataclass (from core_gb/types.py)
- [ ] Test: extract pattern from weather-3-cities tree -> trigger template with {city} slots
- [ ] Test: single-node execution returns None (not worth caching)
- [ ] Test: failed execution returns None

### R002: Pattern Matcher
Match incoming tasks against cached patterns.
**Acceptance Criteria:**
- [ ] `PatternMatcher` class in `core_gb/patterns.py`
- [ ] `match(task: str, patterns: list[Pattern]) -> tuple[Pattern, dict[str, str]] | None`
- [ ] Returns matched pattern + variable bindings, or None
- [ ] Uses Levenshtein similarity on trigger templates (threshold 0.8)
- [ ] Extracts variable bindings from the task text
- [ ] Test: "Weather in Paris, Tokyo, Sydney" matches weather pattern with cities bound
- [ ] Test: "What is 2+2?" doesn't match weather pattern
- [ ] Test: returns highest-confidence match when multiple patterns match

### R003: Pattern Store (Graph Integration)
Store and retrieve patterns from the knowledge graph.
**Acceptance Criteria:**
- [ ] `PatternStore` class in `core_gb/patterns.py`
- [ ] `save(pattern: Pattern) -> str` stores pattern in graph as PatternNode
- [ ] `load_all() -> list[Pattern]` retrieves all patterns from graph
- [ ] `increment_usage(pattern_id: str) -> None` updates success_count and last_used
- [ ] Uses GraphStore CRUD operations
- [ ] Test: save, load, verify round-trip
- [ ] Test: increment_usage updates counters

### R004: Graph Update Loop
After each execution, update the knowledge graph with outcomes.
**Acceptance Criteria:**
- [ ] `GraphUpdater` class in `graph/updater.py`
- [ ] `async update(task: str, nodes: list[TaskNode], result: ExecutionResult) -> None`
- [ ] Records Task nodes in graph with outcomes (tokens, latency, status)
- [ ] Records ExecutionTree node linking task to its tree
- [ ] Extracts and stores new patterns via PatternExtractor
- [ ] Updates entity relationships discovered during execution
- [ ] Test: after execution, Task node exists in graph
- [ ] Test: after multi-node execution, Pattern extracted and stored

### R005: Wire Patterns into Orchestrator
Check patterns before decomposition, update graph after execution.
**Acceptance Criteria:**
- [ ] Orchestrator checks PatternMatcher before calling Decomposer
- [ ] If pattern matches: instantiate template with variable bindings (0 tokens, skip LLM)
- [ ] If no match: proceed with normal decomposition
- [ ] After execution: call GraphUpdater to record outcome
- [ ] Test: cached pattern skips decomposition
- [ ] Test: new task goes through normal decomposition then gets cached
- [ ] Test: second run of same task type hits cache
