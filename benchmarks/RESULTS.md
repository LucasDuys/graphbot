# GraphBot Benchmark Results

## Run Date: 2026-03-21
## Configuration: OpenRouter (free tier), LangSmith tracing, in-memory graph

## Summary

| Metric | Value |
|--------|-------|
| Tasks | 15 |
| Success rate | 15/15 (100%) |
| Answer correct | 14/15 (93%) |
| Decomposition success | 10/10 (100% of decomposable tasks) |
| Total tokens | 34,860 |
| Total cost | $0.002950 |
| Avg cost per task | $0.000197 |

## Per-Task Results

### Simple Tasks (should NOT decompose)

| Task | Nodes | Tokens | Latency | Cost | Correct |
|------|-------|--------|---------|------|---------|
| What is 247 * 38? | 1 | 39 | 1.3s | $0.000001 | Yes |
| Capital of France? | 1 | 37 | 12.3s | $0.000001 | Yes |
| Define recursion | 1 | 396 | 7.8s | $0.000019 | Yes |
| 72F to Celsius | 1 | 100 | 6.6s | $0.000004 | No (rounding) |
| GraphBot language? | 2 | 1,342 | 29.5s | $0.000050 | Yes |

**Notes:**
- simple_04 (temperature conversion) answered correctly but didn't contain exact "22" string match
- simple_05 triggered decomposition unexpectedly (the word "GraphBot" matched graph context)
- Average simple task: 383 tokens, $0.000015

### Parallel Tasks (should decompose to 3+ concurrent leaves)

| Task | Nodes | Tokens | Latency | Cost | Correct |
|------|-------|--------|---------|------|---------|
| Weather: Amsterdam, London, Berlin | 4 | 1,921 | 23.1s | $0.000243 | Yes |
| Features: Python, JS, Rust | 4 | 1,954 | 29.2s | $0.000209 | Yes |
| Pros/cons: PostgreSQL, MongoDB, Redis | 4 | 4,637 | 36.7s | $0.000429 | Yes |
| Differences: TCP, UDP, WebSockets | 4 | 5,082 | 35.0s | $0.000467 | Yes |
| Compare: React, Vue, Svelte | 4 | 4,133 | 37.8s | $0.000394 | Yes |

**Notes:**
- All 5 parallel tasks decomposed correctly to 4 nodes (root + 3 leaves/aggregation)
- Average parallel task: 3,545 tokens, $0.000348
- Decomposition adds latency (LLM call for tree generation) but produces structured, comprehensive answers

### Sequential Tasks (should decompose to ordered chain)

| Task | Nodes | Tokens | Latency | Cost | Correct |
|------|-------|--------|---------|------|---------|
| BST: define, insert, complexity | 3 | 2,936 | 39.6s | $0.000505 | Yes |
| ML: define, supervised vs unsupervised, examples | 5 | 4,435 | 26.5s | $0.000160 | Yes |
| API: define, REST vs GraphQL, recommend | 3 | 2,483 | 39.0s | $0.000275 | Yes |
| HTTP lifecycle: DNS, TCP, TLS, request | 4 | 2,252 | 19.7s | $0.000078 | Yes |
| Docker: define, Compose, Kubernetes | 3 | 3,113 | 40.5s | $0.000115 | Yes |

**Notes:**
- All 5 sequential tasks decomposed correctly
- Node counts vary (3-5) based on how the model structures the chain
- Average sequential task: 3,044 tokens, $0.000227

## Cost Analysis

| Category | Avg Tokens | Avg Cost | Avg Latency |
|----------|-----------|----------|-------------|
| Simple (no decomposition) | 383 | $0.000015 | 11.5s |
| Parallel (decomposed) | 3,545 | $0.000348 | 32.4s |
| Sequential (decomposed) | 3,044 | $0.000227 | 33.1s |
| **All tasks** | **2,324** | **$0.000197** | **25.6s** |

## Key Findings

1. **100% success rate** -- all 15 tasks completed without errors
2. **100% decomposition success** -- all 10 decomposable tasks produced valid trees (up from ~50% before fixes)
3. **Total cost for 15 tasks: $0.003** -- effectively free
4. **Decomposition adds value**: parallel and sequential tasks produce structured, comprehensive answers
5. **Latency is the tradeoff**: decomposed tasks take 20-40s due to multiple LLM round-trips
6. **Pattern cache will reduce latency**: once patterns are cached, decomposition LLM calls are skipped entirely

## Comparison: GraphBot vs Single LLM Call

For the weather comparison task ("Compare weather in Amsterdam, London, Berlin"):

| Metric | GraphBot | Single 8B | Single 70B |
|--------|----------|-----------|------------|
| Tokens | 1,921 | ~200 | ~400 |
| Cost | $0.000243 | $0.000008 | $0.000040 |
| Nodes | 4 | 1 | 1 |
| Structure | Decomposed, parallel | Flat response | Flat response |

GraphBot uses more tokens but produces structured, comprehensive output with independent sub-answers that can be individually verified and cached.
