---
spec: phase10a-prompt-quality
total_tasks: 5
estimated_tokens: 45000
depth: standard
---

# Phase 10A Prompt Quality Frontier

## Tier 1 -- RESEARCH (no code)
- [T071] Research: Anthropic structured output + tool routing prompts | est: ~8k | reqs: R001

## Tier 2 -- IMPLEMENT (from research)
- [T072] Domain override fallback (_infer_domain_from_description) | est: ~8k | depends: T071 | reqs: R002
- [T073] Improved decomposition prompt (tool descriptions, negative examples) | est: ~10k | depends: T071 | reqs: R003

## Tier 3 -- WIRE + TEST
- [T074] SimpleExecutor tool awareness + re-run benchmarks | est: ~8k | depends: T072, T073 | reqs: R004
- [T075] Run real tasks, verify >50% tool usage, commit + push | est: ~5k | depends: T074
