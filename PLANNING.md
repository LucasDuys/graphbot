# GraphBot Architecture Planning

## Design Principles

1. **Graph-first**: Every component reads from and writes to the knowledge graph
2. **Parallel-default**: Independent work runs simultaneously unless explicitly sequential
3. **Cheapest-viable-model**: Always use the smallest model that handles the task with context
4. **Zero-waste**: No redundant computation, no stale context, no unnecessary model calls
5. **Learn-everything**: Every completed task makes the system smarter via pattern extraction
6. **CPU-like efficiency**: Pipeline stages overlap, data forwards directly, critical path prioritized

## Architecture Decision Records (ADRs)

### ADR-001: Kuzu for Graph Storage (with fork migration plan)
- **Status**: ACCEPTED (research validated)
- **Context**: Need embedded graph DB with no external server dependency
- **Decision**: Start with `kuzu` v0.11.3 (last release before archival Oct 2025). Plan migration to LadybugDB fork when stable.
- **Research findings**: Kuzu archived (Apple acquisition). Performance excellent: 2-hop on 100K nodes = 9ms, 374x faster than Neo4j. Temporal support via TIMESTAMP columns (no native bi-temporal, but standard approach). LadybugDB is same API, active development.
- **Consequences**: Same pip install today. Need to monitor LadybugDB. Vela fork available if concurrent writes needed.
- **Alternatives rejected**: DuckDB+DuckPGQ (experimental, SQL/PGQ not Cypher), SQLite+custom (too much DIY), NetworkX (no persistence, slow)

### ADR-002: Nanobot as Starting Point
- **Status**: ACCEPTED (research validated)
- **Context**: Need a minimal agent codebase to build on
- **Decision**: Fork HKUDS/nanobot. Keep: agent loop, tools (incl MCP), bus, providers, channels. Replace: memory, skills, CLI, templates.
- **Research findings**: ~4K core lines confirmed. Clean MessageBus architecture. MCP integration is self-contained (~200 lines). LiteLLM gives 100+ provider support. Session JSONL storage works. Memory system (MEMORY.md + HISTORY.md) is the weakest part -- exactly what we replace.
- **Alternatives rejected**: mcp-agent (library, no channels), PydanticAI (library, no channels), OpenClaw (300-500K lines, overkill)

### ADR-003: Typed Data Contracts for Dependencies
- **Status**: ACCEPTED
- **Context**: Nodes need to pass specific outputs to dependent nodes
- **Decision**: Every TaskNode declares `provides` and `consumes`. Use frozen dataclasses as typed contracts in shared PipelineContext dict.
- **Research findings**: Airflow XCom pattern validates this approach. Frozen dataclasses give immutability (safe concurrent reads) with zero-copy via Python references.

### ADR-004: Rule-Based Intent Classification (No LLM)
- **Status**: ACCEPTED
- **Context**: Router/intake must be zero-cost and instant
- **Decision**: Pattern matching first (keyword lists per domain), tiny local model fallback for ambiguous cases
- **Consequences**: Limited to known patterns initially, expandable.

### ADR-005: Multi-Provider Rotation for $0 Cost
- **Status**: ACCEPTED (replaces single-provider assumption)
- **Context**: No single free tier has enough capacity for sustained usage
- **Decision**: Rotate across Groq -> Cerebras -> Google AI Studio. Per-provider rate limiters + circuit breakers.
- **Research findings**: Groq: 30 RPM, 1K req/day for 70B. Cerebras: 1M tokens/day. Google: Gemini 2.5 Pro free at 5 RPM. OpenRouter: 50 req/day free (1K with $10 credit).
- **Implementation**: `aiolimiter` for rate limiting, `aiobreaker` for circuit breaking, 3-layer concurrency control.

### ADR-006: Updated Model Selection (post-research)
- **Status**: ACCEPTED
- **Context**: Several models from original design are deprecated or unavailable
- **Decision**: Updated model mapping:

| Complexity | Model | Provider |
|-----------|-------|----------|
| 1-2 | Llama 3.1 8B / Llama 4 Scout 17B | Groq (free) |
| 3 | Llama 3.3 70B (NOT 3.1 70B - deprecated) | Groq (free) |
| 4 | Parallel 3x on different providers | Groq + Cerebras + Google |
| 5 | Gemini 2.5 Pro / DeepSeek R1 | Google AI Studio (free) / OpenRouter |
| Code tasks | Qwen3 32B (NOT Qwen 2.5 Coder - unavailable) | Groq (free) |
| Synthesis | Qwen3 235B or Gemini 2.5 Pro | Cerebras / Google |

### ADR-007: Streaming DAG Dispatch (not batch-per-level)
- **Status**: ACCEPTED
- **Context**: Need maximum throughput in the 7-stage pipeline
- **Decision**: Use `graphlib.TopologicalSorter` with streaming dispatch. When a node completes, immediately unblock dependents via `sorter.done()`. Don't wait for entire wave to finish.
- **Research findings**: Streaming dispatch is strictly superior to batch-per-level for mixed parallel/sequential DAGs. Critical path scheduling via longest-path-first priority.

### ADR-008: Graphiti for Temporal Knowledge Graph (prototype)
- **Status**: PROPOSED
- **Context**: Need temporal fact management with entity resolution
- **Decision**: Start with `graphiti-core[kuzu]` for prototype. If LLM overhead is too high with 7-8B models, extract temporal schema and reimplement lighter version with algorithmic entity resolution (exact + edit distance + BM25).
- **Research findings**: Graphiti supports Kuzu driver natively. 4 temporal fields (created_at, valid_at, invalid_at, expired_at). But makes multiple LLM calls per add_episode -- may be slow with small models.
- **Fallback**: Custom temporal layer on Kuzu with non-LLM entity resolution.

### ADR-009: Context Injection Budget
- **Status**: ACCEPTED
- **Context**: Need to know how much graph context to inject per node
- **Decision**: Max 2,000-3,000 tokens of injected context for 7-8B models. Place at beginning/end of prompt, never middle.
- **Research findings**: Quality degrades beyond ~5,000 words for 8B models. Lost-in-the-middle effect confirmed. 256-512 token chunks optimal for individual facts.

### ADR-010: Constrained Decomposition Output
- **Status**: ACCEPTED
- **Context**: Small models struggle to produce valid task trees
- **Decision**: Use constrained JSON schema enforcement for decomposition output. Limit depth to 2-3 levels. Classify subtasks into MECE types (ROMA approach: RETRIEVE, WRITE, THINK, CODE). Provide explicit examples in prompt.
- **Research findings**: No base 7B model produces valid trees reliably. ROMA uses DSPy signatures with type-specialized executors. ReAcTree provides prompt templates. Constrained decoding essential.

## Performance Targets

| Component | Operation | Target Latency | Target Tokens |
|-----------|-----------|---------------|---------------|
| Intake | Intent classification | <5ms | 0 |
| Graph | Entity resolution | <10ms | 0 |
| Graph | Context assembly (leaf) | <10ms | 0 |
| Graph | Pattern match | <5ms | 0 |
| Decomposer | Single-level decomposition | <500ms | ~400 |
| Decomposer | Pattern instantiation | <5ms | 0 |
| Scheduler | Topological sort (50 nodes) | <1ms | 0 |
| Executor | Leaf execution (8B) | <300ms | ~150 |
| Executor | Leaf execution (70B) | <800ms | ~300 |
| Forwarder | Data forwarding | <1ms | 0 |
| Aggregator | LLM synthesis | <500ms | ~200 |

### ADR-011: Pipeline Observability via LangSmith + Custom Tracing
- **Status**: ACCEPTED
- **Context**: No observability infrastructure planned. Need timing, token, and cost tracking per pipeline stage.
- **Decision**: Two-layer observability:
  1. **LangSmith** (free tier, 5K traces/month): Since Nanobot uses LiteLLM, enabling LangSmith is 2 lines (`litellm.success_callback = ["langsmith"]` + env vars). Gives automatic tracing of every LLM call with timing, tokens, cost. Free tier is sufficient for development.
  2. **Custom `@traced` decorator**: For non-LLM operations (graph queries, scheduling, forwarding). Records: stage name, duration_ms, tokens_in/out, model_used, provider, cache_hit. Stored in ExecutionTree graph nodes.
- **Phase rollout**: Enable LangSmith in Phase 1 (trivial setup). Add custom tracing in Phase 3 (when pipeline stages exist).
- **Research findings**: LiteLLM has native LangSmith callback support. LangSmith free tier = 5K traces/month, 14-day retention. Sidekick research confirms LangSmith is the fastest path to visibility for LiteLLM-based projects. P95 (not average) is the correct latency metric.

### ADR-012: Hybrid Verification (Inline + Async)
- **Status**: ACCEPTED
- **Context**: Verify stage needs to be fast for inline checks but also do quality evaluation.
- **Decision**: Two-layer verification:
  - **Inline (<5ms):** Schema validation, confidence heuristic, threshold check. Blocks response if failed.
  - **Async (fire-and-forget):** LLM-as-judge quality score using a DIFFERENT free model than the generator. Stored in graph for learning. Does not block response.
- **Research findings**: Sidekick hybrid evaluation pattern. MT-Bench paper shows LLM judges achieve 80%+ agreement with humans. Use different model as judge to avoid self-enhancement bias.

### ADR-013: Automated Quality Scoring (LLM-as-Judge)
- **Status**: ACCEPTED
- **Context**: Need automated way to grade response quality beyond latency/tokens/cost.
- **Decision**: Implement 3 evaluation modes mapped to use cases:
  1. **Binary pass/fail** (Verify stage): "Is this output correct and complete?" -- inline, every task
  2. **Reference-guided** (regression tests): Compare output to known-good reference -- `make bench`
  3. **Pairwise comparison** (North Star): "Which is better, GraphBot or Nanobot+Sonnet?" -- `scripts/compare.py`
- Use a different free provider for judge than for generation (e.g., generate on Groq, judge on Google Gemini Flash).

### ADR-014: A/B Comparison Infrastructure
- **Status**: ACCEPTED
- **Context**: Need to prove the thesis by running identical tasks on GraphBot vs Nanobot+Sonnet and comparing.
- **Decision**: Build `scripts/compare.py` that:
  1. Takes a task description as input
  2. Runs it on GraphBot (free models, full pipeline)
  3. Runs the same task on stock Nanobot with a paid model (Claude Sonnet via OpenRouter)
  4. Runs LLM-as-judge pairwise comparison
  5. Outputs side-by-side report: quality score, tokens, latency, cost
- This is the demo script for the North Star benchmark.

### ADR-015: Longitudinal Metrics Tracking
- **Status**: ACCEPTED
- **Context**: Need to track system improvement over time -- pattern cache hit rate, graph growth, query performance.
- **Decision**: Track 4 longitudinal metrics, stored in graph and appended to TESTS.md:
  1. **Pattern cache hit rate**: % of tasks that match a cached pattern. Target: 40% month 1, 60% month 3, 75% month 6.
  2. **Graph growth**: node count, edge count, memory count at each benchmark run. Plot growth curve.
  3. **Graph query performance at scale**: Run 2-hop traversal benchmark at 100, 1K, 5K, 10K nodes. Track if <5ms target holds.
  4. **Task completion quality over time**: Average LLM-as-judge score per week/month.
- Implement as `scripts/metrics.py` that queries graph stats and appends to a metrics log.

### ADR-016: Prompt Engineering Patterns for Sub-Agents
- **Status**: ACCEPTED
- **Context**: Research shows XML-tag structure, few-shot bad/good pairs, and sandwich defense improve LLM output reliability.
- **Decision**: All sub-agent prompts and the decomposition prompt must follow:
  1. Critical rules in `<rules>` XML tags
  2. Output schema in `<output_schema>` tags with JSON schema
  3. 2-3 few-shot examples with explicit bad/good pairs
  4. Sandwich defense: restate output format after task description
  5. Tool definitions follow ACI design: namespace prefixes, enum constraints, actionable error messages

## Key Revisions from Research

1. **Medprompt claim revised**: It matches specialist *fine-tuned* models, NOT reasoning models. o1-preview outperforms GPT-4+Medprompt. Core thesis still holds -- architectural intelligence compensates for model cost -- but through different mechanisms (decomposition + context, not just prompt engineering).

2. **Model landscape shifted**: Llama 3.3, Llama 4 Scout, Qwen3 series are the current free models. Original references to Llama 3.1 70B and Qwen 2.5 Coder 7B are outdated.

3. **Kuzu risk identified**: Archived repo means no security patches. LadybugDB fork is the mitigation path.

4. **Graphiti as shortcut**: Instead of building temporal KG from scratch, Graphiti + Kuzu driver gives us the temporal model for free. Evaluate LLM overhead before committing.

5. **Decomposition requires careful prompting**: Can't just throw a task at an 8B model and expect a valid tree. Need constrained output, explicit examples, MECE task types.
