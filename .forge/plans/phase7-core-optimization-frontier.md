---
spec: phase7-core-optimization
total_tasks: 8
estimated_tokens: 70000
depth: standard
---

# Phase 7 Core Optimization Frontier

## Tier 1 (parallel -- no dependencies)
- [T041] Enhanced decomposition schema + prompt (template + tree output) | est: ~10k | reqs: R001
- [T042] Deterministic aggregator (4 aggregation types, 0 LLM) | est: ~8k | reqs: R003
- [T043] CLI chat mode (prompt_toolkit + rich + persistent graph) | est: ~8k | reqs: R005

## Tier 2 (depends on Tier 1)
- [T044] Structured leaf output (JSON schema enforcement) | est: ~8k | depends: T041 | reqs: R002
- [T045] Wire aggregator into DAGExecutor | est: ~8k | depends: T041, T042 | reqs: R004

## Tier 3 (depends on Tier 2)
- [T046] Wire full pipeline: orchestrator uses template decomposition + structured leaves + deterministic aggregation | est: ~10k | depends: T044, T045 | reqs: R004
- [T047] Integration tests: verify 1+N LLM call pattern | est: ~6k | depends: T046

## Tier 4
- [T048] Phase 7 benchmark + commit + push | est: ~5k | depends: T047
