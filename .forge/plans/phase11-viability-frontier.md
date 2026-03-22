---
spec: phase11-viability
total_tasks: 12
estimated_tokens: 100000
depth: thorough
---

# Phase 11 Viability Frontier

## Tier 1 -- RESEARCH (parallel, no code)
- [T085] Research: structured tool calling in agent frameworks (LangChain, OpenAI function calling, Anthropic tool use) | est: ~6k | reqs: R001
- [T086] Research: when to decompose vs single-call (papers on decomposition quality, task classification) | est: ~6k | reqs: R002
- [T087] Research: code editing agents (Aider, Cursor, OpenClaw apply_patch, diff-based editing) | est: ~6k | reqs: R003

## Tier 2 -- IMPLEMENT (parallel, from research)
- [T088] Structured tool params in TaskNode + DecompositionSchema + ToolRegistry | est: ~12k | depends: T085 | reqs: R001
- [T089] Smart decomposition with task_type classification (ATOMIC/DATA_PARALLEL/SEQUENTIAL/INTEGRATED) | est: ~10k | depends: T086 | reqs: R002
- [T090] Tool failure recovery (retry + LLM fallback in DAGExecutor) | est: ~6k | reqs: R004

## Tier 3 -- CODE AGENT (depends on tools working)
- [T091] CodeEditAgent (read -> analyze -> generate diff -> apply -> test) | est: ~15k | depends: T088, T090 | reqs: R003

## Tier 4 -- OBSERVABILITY + WIRING
- [T092] Execution observability (SSE events for tool decisions, UI updates) | est: ~8k | depends: T088, T091 | reqs: R005
- [T093] Wire everything into Orchestrator (task_type routing, code agent, structured tools) | est: ~10k | depends: T089, T091

## Tier 5 -- VALIDATION
- [T094] Run 10 real tasks + 5 code editing tasks, measure improvement | est: ~5k | depends: T093
- [T095] Update benchmarks, PROGRESS.md, README with honest results | est: ~4k | depends: T094
- [T096] Commit + push Phase 11 | est: ~2k | depends: T095
