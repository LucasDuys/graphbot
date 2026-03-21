---
spec: phase6-benchmarks
total_tasks: 7
estimated_tokens: 65000
depth: standard
---

# Phase 6 Benchmarks Frontier

## Tier 1 (parallel -- no dependencies)
- [T034] Benchmark task suite (15 tasks JSON + runner script) | est: ~8k tokens | reqs: R001
- [T035] Fix decomposition reliability (70B, JSON mode, better prompt) | est: ~10k tokens | reqs: R003

## Tier 2 (depends on Tier 1)
- [T036] Enhanced A/B comparison (3-way: GraphBot vs 8B vs 70B) | est: ~8k tokens | depends: T034 | reqs: R002
- [T037] Graph stats script + KG population tracking | est: ~6k tokens | depends: T034 | reqs: R004

## Tier 3 (depends on Tier 2)
- [T038] Run full benchmark suite with real models | est: ~5k tokens | depends: T035, T036, T037
- [T039] Benchmark documentation + README update | est: ~5k tokens | depends: T038 | reqs: R005

## Tier 4
- [T040] Commit + push Phase 6 | est: ~3k tokens | depends: T039
