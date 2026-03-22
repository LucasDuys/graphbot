---
spec: phase12-remaining-gaps
total_tasks: 6
estimated_tokens: 45000
depth: standard
---

# Phase 12 Remaining Gaps Frontier

## Tier 1 (parallel)
- [T097] Fix shell command extraction + execution | est: ~8k | reqs: R001
- [T098] Tool usage metric tracking in ExecutionResult | est: ~6k | reqs: R002

## Tier 2 (depends on Tier 1)
- [T099] SSE observability events for tool decisions | est: ~8k | depends: T097, T098 | reqs: R003

## Tier 3 (depends on all)
- [T100] Re-run all benchmarks + update documentation | est: ~5k | depends: T099 | reqs: R004
- [T101] Commit + push Phase 12 | est: ~2k | depends: T100
