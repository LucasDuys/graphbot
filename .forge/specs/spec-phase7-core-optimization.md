---
domain: phase7-core-optimization
status: approved
created: 2026-03-21
complexity: complex
linked_repos: []
---

# Phase 7: Core Engine Optimization -- True 1-Sequential-N-Parallel

## Overview

Redesign the decomposition and aggregation pipeline to achieve the thesis:
1 sequential LLM call (decomposition) + N parallel calls (leaves) + 0 LLM calls (aggregation).
Based on LLMCompiler architecture (3.7x latency, 6.7x cost savings proven).

## Requirements

### R001: Enhanced Decomposition Output -- Template + Tree
The decomposer must output both a task tree AND an aggregation template.
**Acceptance Criteria:**
- [ ] DecompositionSchema updated: each decomposition includes `output_template` field
- [ ] Template has slot definitions: `{slot_id: description}` mapping to leaf provides
- [ ] Template has `aggregation_type`: "concatenate" | "merge_json" | "confidence_ranked" | "template_fill"
- [ ] DecompositionPrompt updated with template examples
- [ ] Decomposer parses and returns template alongside nodes
- [ ] Test: valid decomposition includes template with slots matching leaf provides
- [ ] Test: template validates against leaf provides (no missing slots)

### R002: Structured Leaf Output
Leaves must return structured JSON, not free text.
**Acceptance Criteria:**
- [ ] SimpleExecutor enhanced: when leaf has provides keys, prompt includes output schema
- [ ] Leaf prompt: "Return a JSON object with key '{provide_key}' containing your answer"
- [ ] response_format=json_object passed for leaf calls when structured output needed
- [ ] Fallback: if JSON parsing fails, wrap raw text as `{provide_key: raw_text}`
- [ ] Test: leaf execution returns parseable JSON with correct keys
- [ ] Test: fallback wraps invalid JSON gracefully

### R003: Deterministic Aggregator (Zero LLM)
Replace LLM synthesis with template-based slot-filling.
**Acceptance Criteria:**
- [ ] `Aggregator` class in `core_gb/aggregator.py`
- [ ] `aggregate(template: dict, leaf_outputs: dict[str, dict]) -> str`
- [ ] Supports 4 aggregation types:
  - concatenate: join outputs with newlines/headers
  - merge_json: deep merge JSON objects
  - confidence_ranked: sort by confidence score, take top-K
  - template_fill: fill slots in a text template
- [ ] Zero LLM calls -- purely deterministic
- [ ] Test: each aggregation type produces correct output
- [ ] Test: missing slot handled gracefully (placeholder text)
- [ ] Benchmark: aggregation < 1ms for 10 leaf outputs

### R004: Wire New Pipeline into DAGExecutor + Orchestrator
Connect the enhanced decomposition -> structured execution -> deterministic aggregation.
**Acceptance Criteria:**
- [ ] DAGExecutor uses structured leaf execution when provides keys exist
- [ ] DAGExecutor collects structured outputs and passes to Aggregator
- [ ] Orchestrator flow: intake -> pattern check -> decompose (1 LLM) -> parallel execute (N LLM) -> aggregate (0 LLM)
- [ ] Total sequential LLM calls = 1 for decomposed tasks
- [ ] Test: end-to-end with mocked provider, verify only 1+N LLM calls made
- [ ] Test: aggregated output matches expected template fill

### R005: CLI Chat Mode
Wire Orchestrator into a usable interactive chat.
**Acceptance Criteria:**
- [ ] `scripts/chat.py` -- standalone CLI chat using prompt_toolkit + rich
- [ ] Loads .env.local, creates persistent GraphStore (data/graph/), initializes Orchestrator
- [ ] Each message goes through Orchestrator.process()
- [ ] Displays: thinking indicator, node count, output, tokens/cost summary
- [ ] Graph persists between sessions (on-disk Kuzu)
- [ ] GraphUpdater records every interaction (knowledge graph grows)
- [ ] `/stats` command shows graph statistics
- [ ] `/history` shows last 10 interactions
- [ ] Test: chat script starts without error
