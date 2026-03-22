---
spec: phase13-polish-and-scale
total_tasks: 11
estimated_tokens: 90000
depth: standard
---

# Phase 13 Frontier

## Tier 1 -- QUICK FIXES (parallel, fast)
- [T102] Fix shell output interpretation + pre-existing test failure | est: ~5k | reqs: R001
- [T103] Git history squash (Nanobot -> single commit) | est: ~3k | reqs: R001

## Tier 2 -- PARALLEL WORKSTREAMS
- [T104] UI: per-node live animations + dark mode toggle | est: ~12k | depends: T102 | reqs: R002
- [T105] UI: knowledge graph D3 panel | est: ~10k | reqs: R002
- [T106] Multi-provider rotation (OpenRouter + Google + Groq) | est: ~10k | reqs: R004
- [T107] Telegram channel integration | est: ~8k | reqs: R003

## Tier 3 -- BENCHMARKS
- [T108] GAIA Level 1 benchmark runner | est: ~8k | depends: T106 | reqs: R005
- [T109] Pattern cache warming script | est: ~6k | reqs: R006

## Tier 4 -- VALIDATE + SHIP
- [T110] Re-run all benchmarks, capture screenshots, update README | est: ~5k | depends: T104, T105, T108, T109
- [T111] Final commit + push | est: ~2k | depends: T110
