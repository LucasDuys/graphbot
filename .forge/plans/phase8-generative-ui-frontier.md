---
spec: phase8-generative-ui
total_tasks: 9
estimated_tokens: 85000
depth: thorough
---

# Phase 8 Generative UI Frontier

## Tier 1 (parallel -- no dependencies)
- [T049] Design system: OKLCH tokens, typography, component primitives | est: ~6k | reqs: R006
- [T050] Next.js app scaffold + React Flow + ELK.js setup | est: ~8k | reqs: R002
- [T051] FastAPI SSE backend + event protocol | est: ~10k | reqs: R001

## Tier 2 (depends on Tier 1)
- [T052] DAG visualization: node rendering + status animations + auto-layout | est: ~12k | depends: T049, T050 | reqs: R002
- [T053] SSE client + Jotai state management | est: ~8k | depends: T050, T051 | reqs: R001

## Tier 3 (depends on Tier 2)
- [T054] Context assembly panel | est: ~8k | depends: T052, T053 | reqs: R003
- [T055] Knowledge graph explorer (D3 force layout) | est: ~10k | depends: T049, T053 | reqs: R004
- [T056] Aggregation visualization | est: ~8k | depends: T052, T053 | reqs: R005

## Tier 4
- [T057] Integration: connect live Orchestrator to UI, full E2E demo | est: ~10k | depends: T054, T055, T056
