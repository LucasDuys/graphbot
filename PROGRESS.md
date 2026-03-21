# GraphBot Progress

## Current Phase
**Phase 5: Integration + Quality -- COMPLETE**
All 5 phases of the core engine are done. Next: optimization, real-world usage, UI.

## Completed
- [x] Full architecture design (2026-03-20)
- [x] Created design docs: CLAUDE.md, AGENTS.md, PROGRESS.md, PLANNING.md, RESEARCH.md, TESTS.md
- [x] Cloned Nanobot (HKUDS/nanobot v0.1.4.post5) as project base
- [x] Set up GraphBot overlay: core_gb/, graph/, agents_gb/, models/, tools_gb/
- [x] Wrote core types: TaskNode, ExecutionResult, Pattern, GraphContext (core_gb/types.py)
- [x] Wrote graph schema: 10 node types, 12 edge types (graph/schema.py)
- [x] Wrote test fixtures and type tests (tests/conftest.py, tests/test_core/test_types.py)
- [x] Research: Kuzu API -- ARCHIVED but functional, sub-5ms 2-hop, LadybugDB fork exists
- [x] Research: Free LLM tiers -- Groq (Llama 3.3 70B, Qwen3 32B), Cerebras (Qwen3 235B), Google (Gemini 2.5 Pro)
- [x] Research: Nanobot audit -- keep agent loop/tools/MCP/providers/channels, replace memory/skills
- [x] Research: Recursive decomposition -- ROMA (open source), ReAcTree (2.4x improvement), NoThinking (9x latency)
- [x] Research: Temporal KG -- Graphiti supports Kuzu, context sweet spot 1.5-3K tokens for 8B models
- [x] Research: OpenClaw -- 250K stars, serial execution, $300-750/mo, no native KG
- [x] Research: Async DAG -- graphlib.TopologicalSorter + streaming dispatch + aiolimiter + aiobreaker
- [x] Updated all ADRs with research findings (PLANNING.md)
- [x] Corrected model references (Llama 3.1 70B -> 3.3 70B, Qwen 2.5 Coder -> Qwen3 32B)
- [x] Corrected Medprompt claim (matches specialists, not reasoning models)
- [x] Cross-referenced Pitchr/Sidekick research (29 docs) -- extracted 6 new findings
- [x] Added ADRs 011-016: observability, hybrid verification, LLM-as-judge, A/B comparison, longitudinal metrics, prompt engineering patterns
- [x] Integrated 4 missing testing areas into TESTS.md: quality scoring, A/B comparison, pattern cache tracking, graph growth metrics

## Phase 1 Completed (2026-03-21)
- [x] T001: Environment setup -- pyproject.toml updated, kuzu/litellm/langsmith installed, Python 3.13 verified
- [x] T002: Core types -- ExecutionResult/Pattern/GraphContext frozen, CompletionResult added, 20 tests
- [x] T003: GraphStore schema creation -- 10 node + 12 edge tables, idempotent init, 7 tests
- [x] T004: OpenRouter provider -- ModelProvider ABC, CompletionResult, error hierarchy, 10 tests
- [x] T005: Graph CRUD -- create/get/update/delete nodes + edges, parameterized Cypher, 22 tests
- [x] T006: Model router -- complexity-based selection, DEFAULT_MODEL_MAP, scale-ready, 9 tests
- [x] T007: LangSmith observability -- litellm callback, silent degradation, 8 tests
- [x] T008: Context assembly -- 2-hop traversal, token budget, active memory filtering, 13 tests
- [x] T009: Entity resolution -- 3-layer (exact/Levenshtein/BM25), >90% accuracy on 50 pairs, 13 tests
- [x] T010: Graph benchmarks -- bench_graph.py at 100-10K nodes, JSONL tracking, 9 tests
- [x] T011: SimpleExecutor -- E2E path (graph context -> LLM call -> result), 5 tests
- **Total: 111 new tests, all passing**

## Phase 2 Completed (2026-03-21)
- [x] T012: IntakeParser -- rule-based zero-cost intent classification, complexity scoring
- [x] T013: Decomposition JSON schema -- JSON schema + validator for task trees
- [x] T014: DecompositionPrompt -- prompt template for LLM-based decomposition
- [x] T015: Tree validation -- structural validation of decomposition output
- [x] T016: Decomposer -- LLM-based task decomposition with retry/fallback
- [x] T017: Orchestrator -- intake -> decompose -> execute pipeline, sequential DAG execution
- **Total: 213+ tests, all passing**

## Phase 3 Completed (2026-03-21)
- [x] T020: Rate limiting -- per-provider rate limiting with aiolimiter
- [x] T021: Circuit breaking -- per-provider circuit breaking with aiobreaker
- [x] T022: Tracing -- @traced decorator for pipeline stage timing
- [x] T023: DAGExecutor -- parallel streaming topological dispatch with semaphore-bounded concurrency
- [x] T024: ModelRouter wiring -- rate_limiter and circuit_breaker params in ModelRouter
- [x] T025/T026: Orchestrator integration -- replaced sequential _execute_dag with parallel DAGExecutor
- **Total: 757 tests, all passing**

## Phase 4 Completed (2026-03-21)
- [x] T027: PatternExtractor -- extracts reusable templates from completed trees
- [x] T028: PatternMatcher -- regex + Levenshtein matching against cached patterns
- [x] T029: PatternStore -- pattern persistence in Kuzu graph (with tree_template)
- [x] T030: GraphUpdater -- records Task/ExecutionTree nodes after execution
- [x] T031: Orchestrator wiring -- pattern cache check before decomposition, graph update after
- **Total: 282 tests, all passing**

## Phase 5 Completed (2026-03-21)
- [x] T033: Graph seed script (seed_graph.py) -- user, projects, services, memories
- [x] T033: A/B comparison script (compare.py) -- GraphBot vs baseline single LLM
- [x] T033: Canonical integration tests -- 5 test tasks with mocked provider
- [x] T033: QUICKSTART.md documentation
- [x] LangSmith EU endpoint configured, claude-agent-sdk installed
- **Total: 287 tests, all passing**

## Phase 7 Completed (2026-03-21)
- [x] T046/T047: End-to-end pipeline pattern verification
  - test_simple_task_1_call: simple task = exactly 1 LLM call, no decomposition
  - test_complex_task_1_plus_n_calls: complex task = 1 decompose + 3 leaves = 4 LLM calls, 0 aggregation
  - test_aggregated_output_has_template: template_fill produces correct slot-filled output
  - test_pattern_cache_skips_decomposition: pre-seeded PatternNode = 0 decompose + 3 leaves = 3 LLM calls
- **Total: 868 tests (864 pre-existing + 4 new), all passing**

## In Progress
_Nothing._

## Blocked
_Nothing blocked._

## Key Decisions Made
| Date | Decision | Rationale | Alternatives Considered |
|------|----------|-----------|------------------------|
| 2026-03-20 | Recursive DAG executor with knowledge graph | Combines ROMA + MiroFish + NoThinking | Flat routing, single-agent RAG |
| 2026-03-20 | Fork Nanobot | 5.3K core lines, model-agnostic, MCP, 8+ channels | OpenClaw (300-500K lines), from scratch |
| 2026-03-20 | Kuzu v0.11.3 (with LadybugDB migration plan) | Embedded, pip install, sub-5ms perf, archived but functional | DuckDB+DuckPGQ, SQLite+custom, NetworkX |
| 2026-03-20 | Multi-provider rotation | No single free tier has enough capacity | Single provider |
| 2026-03-20 | Streaming DAG dispatch | graphlib.TopologicalSorter, immediately unblock dependents | Batch-per-level |
| 2026-03-20 | Graphiti for temporal KG (prototype) | Supports Kuzu driver, temporal model for free | Custom from scratch |
| 2026-03-20 | Constrained JSON for decomposition | 8B models need structured output enforcement | Free-form generation |

## Open Questions
- [ ] Graphiti LLM overhead with local 7-8B models -- acceptable or need custom solution?
- [ ] Constrained JSON decoding: use model's native JSON mode or external library (outlines, guidance)?
- [ ] Pattern matching: Graphiti's entity resolution sufficient or need custom for task patterns?
- [ ] LangSmith free tier (5K traces/month) sufficient for sustained development?

## Metrics
| Metric | Target | Current | Notes |
|--------|--------|---------|-------|
| Lines of code (new) | <8,000 | ~3,500 | core_gb + graph + models + scripts + tests |
| Test count | -- | 868 | All passing (Phase 7 complete) |
| 2-hop query (100 nodes) | <1ms | ~13ms | Needs optimization (full table scan approach) |
| Context assembly (100 nodes) | <10ms | ~23ms | Needs indexed queries |
| Entity resolution (100 nodes) | <10ms | ~3ms | Meets target |
| Simple task latency | <1s | -- | Needs integration test with real API |
| API cost (simple) | $0.00 | -- | |

## Session Log
### 2026-03-20 -- Session 0: Design + Research + Setup
- Designed full architecture across brainstorming sessions
- Cloned HKUDS/nanobot as project base (5.3K core, 28K total)
- Created overlay: core_gb/types.py, graph/schema.py, test fixtures
- Dispatched 7 parallel research agents, all completed
- **Key research findings that changed the design:**
  - Kuzu archived -> plan for LadybugDB migration
  - Llama 3.1 70B deprecated -> use 3.3 70B
  - Qwen 2.5 Coder unavailable -> use Qwen3 32B
  - Graphiti supports Kuzu driver -> can prototype faster
  - Context sweet spot: 1.5-3K tokens for 8B models
  - Decomposition needs constrained JSON output
- Updated all ADRs and model references
- Next session: Begin Phase 1 -- install Kuzu, implement graph store, wire up single model execution

### 2026-03-21 -- Session 1: Phase 1 Foundation (complete)
- Used Forge framework for spec/plan/execute pipeline
- 11 tasks across 5 tiers, executed with full autonomy via parallel subagents
- Built bottom-up: env -> types -> graph store -> provider -> CRUD -> router/langsmith -> context/resolver -> benchmarks/executor
- Key implementation decisions:
  - OpenRouter as single provider gateway (user provides API key)
  - Raw Kuzu (no Graphiti) for maximum performance at scale
  - TaskNode stays mutable (execution state), output types frozen
  - 3-layer entity resolution: exact/Levenshtein/BM25 (no LLM calls)
  - Context assembly via explicit per-edge-type queries (Kuzu v0.11.3 variable-length path limitations)
- Benchmark results: entity resolution fast (3ms), 2-hop and context assembly need optimization (13-23ms at 100 nodes)
- Optimization path: indexed lookups, cached entity lists, batched queries
- Next session: Phase 2 -- recursive decomposition with constrained JSON output

### 2026-03-21 -- Session 2: Phase 2 + Phase 3 (complete)
- Phase 2: Recursive decomposition pipeline (T012-T017)
  - IntakeParser: rule-based complexity scoring, zero LLM calls
  - Decomposer: LLM-based task tree generation with JSON schema validation
  - Orchestrator: full intake -> decompose -> execute pipeline
- Phase 3: Parallel DAG execution (T020-T026)
  - Rate limiting (aiolimiter) and circuit breaking (aiobreaker) per provider
  - @traced decorator for pipeline stage observability
  - DAGExecutor: streaming topological dispatch with asyncio.wait + semaphore
  - Wired DAGExecutor into Orchestrator, replacing sequential _execute_dag
  - Removed ~160 lines of sequential execution code from Orchestrator
- 757 tests passing, 0 failures
- Next session: Phase 4 -- pattern learning and caching
