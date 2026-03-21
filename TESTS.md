# GraphBot Test Cases and Benchmarks

## The Five Canonical Test Tasks

### Test 1: Simple Atomic Task
- **Input**: "What's 247 * 38?"
- **Expected**: 1 node, 1 model call, <500ms, correct answer (9386), $0.00

### Test 2: Parallel Independent Task
- **Input**: "What's the weather in Amsterdam, London, and Berlin?"
- **Expected**: 5 nodes (1 root + 1 aggregation + 3 leaves), parallel execution, <1.5s, $0.00

### Test 3: Sequential Dependent Task
- **Input**: "Read README.md, find all TODOs, list them with line numbers"
- **Expected**: 4 nodes (1 root + 3 sequential), data forwarding between nodes, $0.00

### Test 4: Complex Research Task
- **Input**: "Compare TU/e vs ETH vs TUM for AI master's (admission, cost, placement)"
- **Expected**: 15-25 nodes, 3+ waves, 70B synthesis, <8s, <$0.01

### Test 5: Pattern Cache Hit
- **Input**: Run Test 2 twice
- **Expected**: Second run skips decomposition, fewer tokens, faster

## Component Benchmark Targets

| Metric | Target | Phase |
|--------|--------|-------|
| Context assembly latency | <10ms per node | 1 |
| Entity resolution accuracy | >90% on 50 phrases | 1 |
| Graph query (2-hop) | <5ms | 1 |
| Simple task end-to-end | <1s | 1 |
| Decomposition structural validity | >95% valid trees | 2 |
| Decomposition latency (8B model) | <1.5s | 2 |
| Topological sort (100 nodes) | <1ms | 3 |
| Critical path computation | <2ms | 3 |
| Parallel task (5 leaves) | <1.5s | 3 |
| Complex task (25 nodes) | <8s | 3 |
| Parallel speedup vs sequential | >2x | 3 |
| Pattern cache hit (warm) | <5ms | 4 |
| Token savings from pattern cache | >30% reduction | 4 |
| Tokens per simple task | <200 | 2+ |
| Tokens per complex task | <8000 | 3+ |

## Quality Evaluation (LLM-as-Judge)

### Binary Pass/Fail (inline, every task -- Phase 3+)
Used in the Verify stage. A different free model than the generator scores each leaf output.

| Check | Criteria | Method |
|-------|----------|--------|
| Schema validity | Output matches expected JSON contract | Deterministic, <1ms |
| Completeness | Response addresses the full task description | LLM judge, binary |
| Accuracy | Facts in response are correct (when verifiable) | LLM judge, binary |
| Relevance | Response relates to the task, not hallucinated tangent | LLM judge, binary |

**Judge model rule:** Always use a DIFFERENT provider/model than the generator to avoid self-enhancement bias. E.g., generate on Groq Llama 3.3 70B, judge on Google Gemini 2.5 Flash.

### Reference-Guided (regression tests -- Phase 2+)
For tasks with known-correct outputs, compare GraphBot's output to a reference.

| Task | Reference Answer | Scoring |
|------|-----------------|---------|
| "What's 247 * 38?" | "9386" | Exact match |
| "Weather in Amsterdam" | Contains: temperature, conditions, location | Keyword presence |
| "Read README, find TODOs" | List of actual TODOs from file | Subset match |

### Pairwise Comparison (North Star benchmark -- Phase 3+)
Run the same task on GraphBot and Nanobot+Sonnet. LLM judge picks the better response.

**Script:** `scripts/compare.py`
**Output:** Side-by-side report with:
- Quality winner (judge's pick + reasoning)
- Token usage (GraphBot vs Nanobot)
- Latency (GraphBot vs Nanobot)
- Cost (GraphBot: $0.00 target vs Nanobot: tracked)

## A/B Comparison Infrastructure

`scripts/compare.py` implementation:
1. Accept task description as CLI argument
2. Run on GraphBot (free models, full pipeline) -- capture output, tokens, latency, cost
3. Run on stock Nanobot with Claude Sonnet (via OpenRouter) -- capture same metrics
4. Run LLM-as-judge pairwise comparison (Gemini Flash as judge)
5. Output markdown report to `benchmarks/comparisons/YYYY-MM-DD-{task_slug}.md`

**First runnable at:** Phase 3 (when parallel execution works)
**Required for thesis proof:** Phase 3+

## Longitudinal Metrics

### Pattern Cache Hit Rate (Phase 4+)
Track what % of incoming tasks match a cached pattern over time.

| Time Period | Target Hit Rate |
|-------------|----------------|
| Week 1 | 10-20% (cold start) |
| Month 1 | 40% |
| Month 3 | 60% |
| Month 6 | 75% |

**Implementation:** `scripts/metrics.py` queries graph for PatternNode usage stats.
**Storage:** Append to `benchmarks/metrics_log.jsonl` with timestamp.

### Graph Growth Metrics (Phase 1+)
Track graph size at each benchmark run.

| Metric | How |
|--------|-----|
| Total nodes | `MATCH (n) RETURN count(n)` |
| Total edges | `MATCH ()-[r]->() RETURN count(r)` |
| Nodes by type | `MATCH (n) RETURN label(n), count(n)` |
| Memory nodes (active) | `MATCH (m:Memory) WHERE m.valid_until IS NULL RETURN count(m)` |
| Pattern nodes | `MATCH (p:PatternNode) RETURN count(p)` |

### Graph Query Performance at Scale (Phase 1+)
Benchmark 2-hop traversal at different graph sizes to detect degradation.

| Graph Size | Target Latency | Test |
|-----------|---------------|------|
| 100 nodes | <1ms | bench_context_assembly.py |
| 1,000 nodes | <2ms | bench_context_assembly.py |
| 5,000 nodes | <4ms | bench_context_assembly.py |
| 10,000 nodes | <5ms | bench_context_assembly.py |

**Implementation:** `scripts/seed_graph.py --scale N` generates synthetic graphs at target sizes.

### Task Quality Over Time (Phase 3+)
Average LLM-as-judge score per week, tracked longitudinally.

**Storage:** Each task execution stores its judge score in the graph (on the Task node).
**Query:** `MATCH (t:Task) WHERE t.completed_at > timestamp('2026-04-01') RETURN avg(t.quality_score)`

## Benchmark Tracking Table

Results appended here after each benchmark run.

| Date | Phase | Task | Nodes | Tokens | Latency (ms) | Cost ($) | Quality Score | Models Used | Notes |
|------|-------|------|-------|--------|--------------|----------|---------------|-------------|-------|
| -- | -- | -- | -- | -- | -- | -- | -- | -- | _No benchmarks run yet_ |

## Pipeline Tracing Targets (per stage)

Every stage records: duration_ms, tokens_in, tokens_out, model_used, provider, cache_hit.

| Stage | Target Duration | Token Budget |
|-------|----------------|-------------|
| Intake | <5ms | 0 |
| Graph Query | <10ms | 0 |
| Pattern Match | <5ms | 0 |
| Decompose | <1.5s | ~400 |
| Schedule | <1ms | 0 |
| Contextualize | <10ms | 0 |
| Execute (8B leaf) | <300ms | ~150 |
| Execute (70B leaf) | <800ms | ~300 |
| Verify (inline) | <5ms | 0 |
| Verify (async judge) | ~500ms | ~100 (async, not blocking) |
| Forward | <1ms | 0 |
| Aggregate | <500ms | ~200 |
| Graph Update | <10ms | 0 |
