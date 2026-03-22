# GraphBot Gap Analysis -- March 2026

## Research Sources
- OpenClaw architecture deep dive (3 research agents)
- Scientific literature on decomposition quality
- Honest codebase audit (all critical paths traced)

## Executive Summary

GraphBot is a well-engineered research prototype with 399+ tests and unique
architectural advantages (DAG execution, knowledge graph, pattern caching).
But it fails the basic requirements of a coding agent because:

1. Code editing is explicitly rejected in the tool registry
2. Tool routing is regex-based guessing with no structured parameters
3. Decomposition quality sometimes HURTS output (isolated nodes lose context)
4. No failure recovery when tools break
5. Users can't see why things failed

## OpenClaw Comparison

| Dimension | OpenClaw | GraphBot | Winner |
|-----------|----------|----------|--------|
| Capabilities | Full (files, shell, browser, MCP, channels) | Partial (files, shell, web search) | OpenClaw |
| Cost per task | $0.10-1.00 | $0.002 | GraphBot (50-500x cheaper) |
| Architecture | Serial ReAct loop | Parallel DAG | GraphBot |
| Memory | JSONL + Markdown | Temporal knowledge graph | GraphBot |
| Security | 6+ CVEs, 390K public instances | Sandboxed, scoped | GraphBot |
| Code editing | Full (apply_patch, diff-based) | Broken (explicitly rejected) | OpenClaw |
| Decomposition | Manual (user/LLM decides) | Automatic (but naive) | Tie |
| Production readiness | High (250K users) | Low (prototype) | OpenClaw |

## The 5 Fixes That Matter

1. Structured tool parameters (impact 9/10, effort 1-2 days)
2. Smart decomposition -- know when NOT to decompose (impact 8/10, effort 2-3 days)
3. Code editing agent with diff-based editing (impact 9/10, effort 2-3 days)
4. Tool failure recovery with retry + fallback (impact 6/10, effort 1 day)
5. User-facing observability for tool decisions (impact 5/10, effort 2 days)

## Positioning

GraphBot should NOT try to replicate OpenClaw feature-for-feature.
Instead: "OpenClaw for teams that can't afford $150/task or 2-3M tokens/day."

GraphBot's advantages are structural:
- 50-500x cheaper per task
- Parallel execution (vs OpenClaw's serial ReAct)
- Temporal knowledge graph (vs JSONL files)
- Pattern caching (no competitor has this)
- Security by design (vs 6+ CVEs)
